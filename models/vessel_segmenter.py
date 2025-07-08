# models/vessel_segmenter.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ch_branch import CHBranch
from .spatial_branch import SpatialBranch
from .fusion.attention_fusion import AttentionFusion
from .fusion.multiscale_fusion import MultiscaleFusion
from .spatial_branch.edge_enhancement import EdgeEnhancement

__all__ = ["VesselSegmenter", "DistributedVesselSegmenter"]


class VesselSegmenter(nn.Module):
	"""Three‑tier TA‑CHNet with fully adaptive channel dimensions."""
	
	def __init__(
			self,
			in_channels: int = 1,
			out_channels: int = 1,
			ch_params: Dict | None = None,
			tier_params: Dict | None = None,
	) -> None:
		super().__init__()
		
		# ------------------------------------------------------------------
		# Branches
		# ------------------------------------------------------------------
		if ch_params is None:
			ch_params = {
				"max_n": 3,
				"max_k": 4,
				"max_l": 5,
				"cylindrical_dims": (32, 36, 32),
			}
		if tier_params is None:
			tier_params = {
				0: {"max_n": 2, "max_k": 3, "max_l": 4, "r_scale": 1.0},
				1: {"max_n": 3, "max_k": 4, "max_l": 5, "r_scale": 1.5},
				2: {"max_n": 4, "max_k": 5, "max_l": 6, "r_scale": 2.0},
			}
		self.tier_params = tier_params
		self.current_tier: int | None = None
		self.edge_enhance = EdgeEnhancement(out_channels=8)
		
		self.ch_branch = CHBranch(**ch_params)
		self.spatial_branch = SpatialBranch(in_channels, 16, 16)  # 内部自己决定输出通道
		
		# ------------------------------------------------------------------
		# Fusion & head – lazy adaptive
		# ------------------------------------------------------------------
		self.attention_fusion = AttentionFusion()  # lazy inside
		self.multiscale_fusion = MultiscaleFusion()  # lazy inside
		
		self.seg_head_first: nn.Conv3d | None = None  # lazy build on first forward
		self.seg_head_tail = nn.Sequential(
			nn.InstanceNorm3d(32),
			nn.ReLU(inplace=True),
			nn.Conv3d(32, out_channels, 1),
			nn.Sigmoid() if out_channels == 1 else nn.Softmax(dim=1),
		)
		
		# cache for tier features
		self.tier_features: Dict[int, torch.Tensor] = {}
	
	# ------------------------------------------------------------------
	# Utility builders
	# ------------------------------------------------------------------
	def _build_seg_head(self, in_c: int, ref: torch.Tensor):
		"""Create first 3×3 conv to map `in_c → 32`, align device/dtype."""
		self.seg_head_first = nn.Conv3d(in_c, 32, 3, padding=1, bias=False)
		self.seg_head_first.to(ref.device, dtype=ref.dtype)
	
	# ------------------------------------------------------------------
	# Tier control helpers
	# ------------------------------------------------------------------
	def set_tier(self, tier: int):
		"""设置当前tier"""
		self.current_tier = tier
		self.ch_branch.set_tier(tier)
		self.tier_features.clear()
	
	# ------------------------------------------------------------------
	def forward(self, x: torch.Tensor, tier: int | None = None) -> torch.Tensor:
		"""
        标准前向传播 - 完整执行整个模型

        参数:
            x: 输入张量
            tier: 可选的tier设置

        返回:
            分割结果
        """
		if tier is not None:
			self.set_tier(tier)
		if self.current_tier is None:
			raise ValueError("Tier must be set before forward pass")
		
		# Tier‑specific scaling factor
		r_scale = self.tier_params.get(self.current_tier, {}).get("r_scale", 1.0)
		edge_feat = self.edge_enhance(x)
		
		# Branch forward
		ch_features = self.ch_branch(x, r_scale=r_scale)
		spatial_features = self.spatial_branch(x)
		
		# Fuse
		fused = self.attention_fusion(ch_features, spatial_features)
		self.tier_features[self.current_tier] = fused
		
		# Multi‑tier fusion if >1 tier collected
		final = (
			self.multiscale_fusion(self.tier_features)
			if len(self.tier_features) > 1
			else fused
		)
		
		# Lazy build segmentation head
		if self.seg_head_first is None:
			self._build_seg_head(final.shape[1], final)
		# ensure seg_head_first on right device/dtype even after .to()
		if next(self.seg_head_first.parameters()).device != final.device or \
				next(self.seg_head_first.parameters()).dtype != final.dtype:
			self.seg_head_first.to(final.device, dtype=final.dtype)
		
		logits = self.seg_head_tail(self.seg_head_first(final))
		return logits
	
	# ------------------------------------------------------------------
	# 为分布式执行添加的方法 - 将前向传播分解为多个阶段
	# ------------------------------------------------------------------
	def frontend_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
        前端处理 (GPU 0)
        - 包括预处理和FFT变换

        参数:
            x: 输入张量

        返回:
            tuple: (cylindrical_spectrum, x)
        """
		# 确保当前tier设置
		if self.current_tier is None:
			raise ValueError("Tier must be set before forward pass")
		
		# 执行3D FFT
		spectrum = self.ch_branch.fft_utils.fft3d(x, apply_window=True)
		
		# 执行柱坐标映射
		cylindrical_spectrum = self.ch_branch.cylindrical_mapping.cartesian_to_cylindrical(spectrum)
		
		return cylindrical_spectrum, x
	
	def ch_processing_forward(self, cylindrical_spectrum: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
		"""
        CH核心处理 (GPU 1)
        - 包括CH分解和CH系数注意力

        参数:
            cylindrical_spectrum: 从前端传来的柱坐标频谱

        返回:
            tuple: (enhanced_coeffs, (cylindrical_spectrum, ch_coeffs))
        """
		# 确保当前tier设置
		if self.current_tier is None:
			raise ValueError("Tier must be set before forward pass")
		
		# 获取tier特定的r_scale
		r_scale = self.tier_params.get(self.current_tier, {}).get("r_scale", 1.0)
		
		# 执行CH分解
		ch_coeffs = self.ch_branch.ch_transform.decompose(cylindrical_spectrum, r_scale=r_scale)
		
		# 应用CH系数注意力
		if self.current_tier is not None:
			enhanced_coeffs = self.ch_branch.ch_attention.tier_specific_enhancement(ch_coeffs, self.current_tier)
		else:
			enhanced_coeffs = self.ch_branch.ch_attention(ch_coeffs)
		
		return enhanced_coeffs, (cylindrical_spectrum, ch_coeffs)
	
	def _reconstruct_ch_features(self, enhanced_coeffs: torch.Tensor,
	                             cylindrical_spectrum: torch.Tensor) -> torch.Tensor:
		"""
        从CH系数重构特征

        参数:
            enhanced_coeffs: 增强的CH系数
            cylindrical_spectrum: 原始柱坐标频谱

        返回:
            重构的CH特征
        """
		# 获取tier特定的r_scale
		r_scale = self.tier_params.get(self.current_tier, {}).get("r_scale", 1.0)
		
		# 获取柱坐标维度
		r_samples, theta_samples, z_samples = self.ch_branch.cylindrical_dims
		
		# 执行逆CH变换
		reconstructed_cylindrical = self.ch_branch.ch_transform.reconstruct(
			enhanced_coeffs,
			(r_samples, theta_samples, z_samples),
			r_scale=r_scale
		)
		
		# 执行柱坐标逆映射
		input_shape = cylindrical_spectrum.shape[2:5]
		reconstructed_spectrum = self.ch_branch.cylindrical_mapping.cylindrical_to_cartesian(
			reconstructed_cylindrical,
			input_shape
		)
		
		# 执行3D逆FFT
		ch_features = self.ch_branch.fft_utils.ifft3d(reconstructed_spectrum)
		
		return ch_features
	
	def spatial_fusion_forward(self, data: Tuple[torch.Tensor, Tuple], input_x: torch.Tensor) -> Tuple[
		torch.Tensor, Tuple]:
		"""
        空间处理与融合 (GPU 2)
        - 包括空间分支处理和特征融合

        参数:
            data: 从CH处理传来的数据 (enhanced_coeffs, (cylindrical_spectrum, ch_coeffs))
            input_x: 原始输入

        返回:
            tuple: (fused, (ch_features, spatial_features))
        """
		enhanced_coeffs, ch_tensors = data
		cylindrical_spectrum, _ = ch_tensors
		
		# 重构CH特征
		ch_features = self._reconstruct_ch_features(enhanced_coeffs, cylindrical_spectrum)
		
		# 执行空间分支处理
		edge_feat = self.edge_enhance(input_x)
		spatial_features = self.spatial_branch(input_x)
		
		# 特征融合
		fused = self.attention_fusion(ch_features, spatial_features)
		
		return fused, (ch_features, spatial_features)
	
	def backend_forward(self, data: Tuple[torch.Tensor, Tuple]) -> torch.Tensor:
		"""
        后端处理 (GPU 3)
        - 包括多尺度融合和分割头

        参数:
            data: 从空间融合传来的数据 (fused, (ch_features, spatial_features))

        返回:
            分割结果
        """
		fused, _ = data
		
		# 保存当前tier特征
		self.tier_features[self.current_tier] = fused
		
		# 多尺度融合 (如果有多个tier)
		if len(self.tier_features) > 1:
			final = self.multiscale_fusion(self.tier_features)
		else:
			final = fused
		
		# 延迟构建分割头
		if self.seg_head_first is None:
			self._build_seg_head(final.shape[1], final)
		
		# 确保分割头在正确的设备上
		if next(self.seg_head_first.parameters()).device != final.device or \
				next(self.seg_head_first.parameters()).dtype != final.dtype:
			self.seg_head_first.to(final.device, dtype=final.dtype)
		
		# 分割头处理
		logits = self.seg_head_tail(self.seg_head_first(final))
		
		return logits


class DistributedVesselSegmenter:
	"""
    分布式版本的VesselSegmenter - 在多个GPU上协调执行
    作为原始VesselSegmenter的包装器，提供兼容接口
    """
	
	def __init__(self, model: VesselSegmenter, gpus: List[int]):
		"""
        初始化分布式模型

        参数:
            model: 原始VesselSegmenter模型
            gpus: 使用的GPU列表 [gpu0, gpu1, gpu2, gpu3]
        """
		self.model = model
		self.gpus = gpus
		self.current_tier = None
		
		# 将模型组件分布到不同GPU
		self._distribute_components()
	
	def _distribute_components(self):
		"""将模型组件分布到不同GPU"""
		# GPU 0: 前端组件
		self.model.ch_branch.fft_utils.to(f'cuda:{self.gpus[0]}')
		self.model.ch_branch.cylindrical_mapping.to(f'cuda:{self.gpus[0]}')
		
		# GPU 1: CH处理组件
		self.model.ch_branch.ch_transform.to(f'cuda:{self.gpus[1]}')
		self.model.ch_branch.ch_attention.to(f'cuda:{self.gpus[1]}')
		
		# GPU 2: 空间处理与融合组件
		self.model.spatial_branch.to(f'cuda:{self.gpus[2]}')
		self.model.attention_fusion.to(f'cuda:{self.gpus[2]}')
		self.model.edge_enhance.to(f'cuda:{self.gpus[2]}')
		
		# GPU 3: 后端组件
		self.model.multiscale_fusion.to(f'cuda:{self.gpus[3]}')
		if self.model.seg_head_first is not None:
			self.model.seg_head_first.to(f'cuda:{self.gpus[3]}')
		self.model.seg_head_tail.to(f'cuda:{self.gpus[3]}')
	
	def set_tier(self, tier: int):
		"""设置当前tier"""
		self.current_tier = tier
		self.model.set_tier(tier)
	
	def forward(self, x: torch.Tensor, tier: int = None) -> torch.Tensor:
		"""
        分布式前向传播

        参数:
            x: 输入张量
            tier: 可选的tier设置

        返回:
            分割结果
        """
		if tier is not None:
			self.set_tier(tier)
		
		# 确保输入在GPU 0上
		x = x.to(f'cuda:{self.gpus[0]}')
		
		# 阶段1: 前端处理 (GPU 0)
		cylindrical_spectrum, x_ref = self.model.frontend_forward(x)
		
		# 将结果传输到GPU 1
		cylindrical_spectrum = cylindrical_spectrum.to(f'cuda:{self.gpus[1]}')
		
		# 阶段2: CH处理 (GPU 1)
		enhanced_coeffs, ch_tensors = self.model.ch_processing_forward(cylindrical_spectrum)
		
		# 将结果传输到GPU 2
		enhanced_coeffs = enhanced_coeffs.to(f'cuda:{self.gpus[2]}')
		x_ref = x_ref.to(f'cuda:{self.gpus[2]}')
		
		# 阶段3: 空间处理与融合 (GPU 2)
		fused, fusion_tensors = self.model.spatial_fusion_forward((enhanced_coeffs, ch_tensors), x_ref)
		
		# 将结果传输到GPU 3
		fused = fused.to(f'cuda:{self.gpus[3]}')
		
		# 阶段4: 后端处理 (GPU 3)
		output = self.model.backend_forward((fused, fusion_tensors))
		
		return output
	
	def train(self, mode: bool = True):
		"""设置训练模式"""
		self.model.train(mode)
		return self
	
	def eval(self):
		"""设置评估模式"""
		self.model.eval()
		return self
	
	def to(self, device):
		"""移动模型到设备(兼容接口)"""
		# 忽略传入的device，使用分布式设置
		self._distribute_components()
		return self
	
	def parameters(self):
		"""返回模型参数(兼容接口)"""
		return self.model.parameters()
	
	def state_dict(self):
		"""返回模型状态字典"""
		return self.model.state_dict()
	
	def load_state_dict(self, state_dict):
		"""加载模型状态字典"""
		self.model.load_state_dict(state_dict)
		# 重新分布组件
		self._distribute_components()