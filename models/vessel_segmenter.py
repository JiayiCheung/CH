#!/usr/bin/env python3
"""
血管分割模型
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ch_branch import CHBranch
from .spatial_branch import SpatialBranch
from .fusion.attention_fusion import AttentionFusion
from .fusion.multiscale_fusion import MultiscaleFusion
from .spatial_branch.edge_enhancement import EdgeEnhancement

__all__ = ["VesselSegmenter"]

class VesselSegmenter(nn.Module):
	"""
	三级TA-CHNet血管分割模型
	支持多尺度采样和动态通道适配
	"""
	
	def __init__(
			self,
			in_channels: int = 1,
			out_channels: int = 1,
			ch_params: Dict | None = None,
			tier_params: Dict | None = None,
	) -> None:
		super().__init__()
		
		# 设置默认参数
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
		
		self.ch_params = ch_params
		self.tier_params = tier_params
		self.current_tier: int | None = None
		
		# 核心组件
		self.ch_branch = CHBranch(**ch_params)
		self.spatial_branch = SpatialBranch(in_channels, 16, 16)
		self.edge_enhance = EdgeEnhancement(out_channels=8)
		
		# 融合模块
		self.attention_fusion = AttentionFusion()
		self.multiscale_fusion = MultiscaleFusion()
		
		# 分割头 - 延迟初始化
		self.seg_head_first = None
		self.seg_head_tail = nn.Sequential(
			nn.InstanceNorm3d(32),
			nn.ReLU(inplace=True),
			nn.Conv3d(32, out_channels, 1),
		)
		
		# 添加最终激活函数
		if out_channels == 1:
			self.final_activation = nn.Sigmoid()
		else:
			self.final_activation = nn.Softmax(dim=1)
		
		# tier特征缓存
		self.tier_features: Dict[int, torch.Tensor] = {}
		
		# 模型状态
		self._is_distributed = False
		self._device_info = None
	
	def _build_seg_head(self, in_channels: int, reference_tensor: torch.Tensor):
		"""延迟构建分割头的第一层"""
		self.seg_head_first = nn.Conv3d(in_channels, 32, 3, padding=1, bias=False)
		
		# 移动到正确的设备和数据类型
		self.seg_head_first.to(
			device=reference_tensor.device,
			dtype=reference_tensor.dtype
		)
	
	def set_tier(self, tier: int):
		"""设置当前处理的tier"""
		if tier not in self.tier_params:
			raise ValueError(f"Invalid tier: {tier}. Available tiers: {list(self.tier_params.keys())}")
		
		self.current_tier = tier
		
		# 更新CH分支的tier参数
		if hasattr(self.ch_branch, 'set_tier_params'):
			self.ch_branch.set_tier_params(self.tier_params[tier])
	
	def get_current_tier(self) -> int | None:
		"""获取当前tier"""
		return self.current_tier
	
	def clear_tier_cache(self):
		"""清空tier特征缓存"""
		self.tier_features.clear()
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		前向传播

		参数:
			x: 输入张量 [B, C, D, H, W]

		返回:
			分割结果张量
		"""
		# 输入验证
		if x.dim() != 5:
			raise ValueError(f"Expected 5D input (B,C,D,H,W), got {x.dim()}D")
		
		if self.current_tier is None:
			raise RuntimeError("Must set tier before forward pass. Call set_tier() first.")
		
		# 内存格式优化
		x = x.to(memory_format=torch.channels_last_3d)
		
		# CH分支处理
		ch_features = self.ch_branch(x)
		
		# 空间分支处理
		spatial_features = self.spatial_branch(x)
		
		# 边缘增强
		enhanced_features = self.edge_enhance(spatial_features)
		
		# 特征融合
		fused_features = self.attention_fusion(ch_features, enhanced_features)
		
		# 保存当前tier特征
		self.tier_features[self.current_tier] = fused_features
		
		# 多尺度融合（如果有多个tier的特征）
		if len(self.tier_features) > 1:
			final_features = self.multiscale_fusion(self.tier_features)
		else:
			final_features = fused_features
		
		# 延迟构建分割头
		if self.seg_head_first is None:
			self._build_seg_head(final_features.shape[1], final_features)
		
		# 确保分割头在正确设备上
		if self.seg_head_first.training != self.training:
			self.seg_head_first.train(self.training)
		
		# 分割头处理
		x = self.seg_head_first(final_features)
		x = self.seg_head_tail(x)
		
		# 应用最终激活函数
		x = self.final_activation(x)
		
		return x
	
	def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		返回中间特征的前向传播，用于分析和可视化

		参数:
			x: 输入张量

		返回:
			(ch_features, spatial_features, fused_features)
		"""
		if self.current_tier is None:
			raise RuntimeError("Must set tier before forward pass")
		
		x = x.to(memory_format=torch.channels_last_3d)
		
		# 提取特征
		ch_features = self.ch_branch(x)
		spatial_features = self.spatial_branch(x)
		enhanced_features = self.edge_enhance(spatial_features)
		fused_features = self.attention_fusion(ch_features, enhanced_features)
		
		return ch_features, enhanced_features, fused_features
	
	def get_model_info(self) -> Dict:
		"""获取模型信息"""
		total_params = sum(p.numel() for p in self.parameters())
		trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
		
		info = {
			"model_name": "VesselSegmenter",
			"total_parameters": total_params,
			"trainable_parameters": trainable_params,
			"current_tier": self.current_tier,
			"available_tiers": list(self.tier_params.keys()),
			"ch_params": self.ch_params,
			"tier_params": self.tier_params,
			"cached_tiers": list(self.tier_features.keys()),
			"device": next(self.parameters()).device,
			"dtype": next(self.parameters()).dtype,
		}
		
		return info
	
	def estimate_memory_usage(self, input_shape: Tuple[int, ...]) -> Dict:
		"""估算内存使用量"""
		# 简单的内存估算
		batch_size, channels, depth, height, width = input_shape
		
		# 输入数据内存
		input_memory = batch_size * channels * depth * height * width * 4  # float32
		
		# 特征图内存（粗略估计）
		feature_memory = input_memory * 10  # 假设特征图总共是输入的10倍
		
		# 参数内存
		param_memory = sum(p.numel() * 4 for p in self.parameters())  # float32
		
		# 梯度内存
		grad_memory = param_memory if self.training else 0
		
		total_memory = input_memory + feature_memory + param_memory + grad_memory
		
		return {
			"input_memory_mb": input_memory / (1024 * 1024),
			"feature_memory_mb": feature_memory / (1024 * 1024),
			"parameter_memory_mb": param_memory / (1024 * 1024),
			"gradient_memory_mb": grad_memory / (1024 * 1024),
			"total_memory_mb": total_memory / (1024 * 1024),
			"total_memory_gb": total_memory / (1024 * 1024 * 1024),
		}
	
	def optimize_for_inference(self):
		"""为推理优化模型"""
		self.eval()
		
		# 清空tier缓存
		self.clear_tier_cache()
		
		# 设置为推理模式
		for module in self.modules():
			if hasattr(module, 'eval'):
				module.eval()
		
		# 禁用梯度计算
		for param in self.parameters():
			param.requires_grad_(False)
	
	def prepare_for_training(self):
		"""为训练准备模型"""
		self.train()
		
		# 启用梯度计算
		for param in self.parameters():
			param.requires_grad_(True)
		
		# 清空缓存
		self.clear_tier_cache()
	
	def save_checkpoint(self, filepath: Union[str, Path],
	                    optimizer=None, scheduler=None, epoch=None,
	                    best_dice=None, additional_info=None):
		"""
		保存检查点

		参数:
			filepath: 保存路径
			optimizer: 优化器状态
			scheduler: 调度器状态
			epoch: 当前轮数
			best_dice: 最佳Dice分数
			additional_info: 额外信息
		"""
		checkpoint = {
			'model_state_dict': self.state_dict(),
			'model_info': self.get_model_info(),
			'ch_params': self.ch_params,
			'tier_params': self.tier_params,
		}
		
		if optimizer is not None:
			checkpoint['optimizer_state_dict'] = optimizer.state_dict()
		
		if scheduler is not None:
			checkpoint['scheduler_state_dict'] = scheduler.state_dict()
		
		if epoch is not None:
			checkpoint['epoch'] = epoch
		
		if best_dice is not None:
			checkpoint['best_dice'] = best_dice
		
		if additional_info is not None:
			checkpoint['additional_info'] = additional_info
		
		torch.save(checkpoint, filepath)
	
	@classmethod
	def load_from_checkpoint(cls, filepath: Union[str, Path],
	                         device=None, strict=True):
		"""
		从检查点加载模型

		参数:
			filepath: 检查点路径
			device: 目标设备
			strict: 是否严格加载

		返回:
			(model, checkpoint_info)
		"""
		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		
		checkpoint = torch.load(filepath, map_location=device)
		
		# 提取模型参数
		ch_params = checkpoint.get('ch_params')
		tier_params = checkpoint.get('tier_params')
		
		# 创建模型
		model = cls(ch_params=ch_params, tier_params=tier_params)
		
		# 加载权重
		model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
		model.to(device)
		
		# 返回检查点信息
		checkpoint_info = {
			'epoch': checkpoint.get('epoch'),
			'best_dice': checkpoint.get('best_dice'),
			'model_info': checkpoint.get('model_info'),
			'additional_info': checkpoint.get('additional_info'),
		}
		
		return model, checkpoint_info