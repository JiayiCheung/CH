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

from .ch_branch.ch_branch import CHBranch
from .spatial_branch import SpatialBranch
from .fusion.attention_fusion import AttentionFusion
from .spatial_branch.edge_enhancement import EdgeEnhancement

__all__ = ["VesselSegmenter"]


class VesselSegmenter(nn.Module):
	"""
	血管分割模型
	基于柱谐波变换的血管分割模型，结合空间支路和频域特征
	"""
	
	def __init__(
			self,
			in_channels: int = 1,
			out_channels: int = 1,
			ch_params: Dict | None = None,
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
		
		self.ch_params = ch_params
		
		# 核心组件
		self.ch_branch = CHBranch(**ch_params)
		self.spatial_branch = SpatialBranch(in_channels, 16, 16)
		
		
		# 融合模块
		self.attention_fusion = AttentionFusion()
		
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
		
		# 内存格式优化
		x = x.to(memory_format=torch.channels_last_3d)
		
		# CH分支处理
		ch_features = self.ch_branch(x)
		
		# 空间分支处理
		spatial_features = self.spatial_branch(x)
		
		
		
		# 特征融合
		fused_features = self.attention_fusion(ch_features, spatial_features)
		
		# 延迟构建分割头
		if self.seg_head_first is None:
			self._build_seg_head(fused_features.shape[1], fused_features)
		
		# 确保分割头在正确设备上
		if self.seg_head_first.training != self.training:
			self.seg_head_first.train(self.training)
		
		# 分割头处理
		x = self.seg_head_first(fused_features)
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
		x = x.to(memory_format=torch.channels_last_3d)
		
		# 提取特征
		ch_features = self.ch_branch(x)
		spatial_features = self.spatial_branch(x)
		
		fused_features = self.attention_fusion(ch_features, spatial_features)
		
		return ch_features, spatial_features, fused_features
	
	def get_model_info(self) -> Dict:
		"""获取模型信息"""
		total_params = sum(p.numel() for p in self.parameters())
		trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
		
		info = {
			"model_name": "VesselSegmenter",
			"total_parameters": total_params,
			"trainable_parameters": trainable_params,
			"ch_params": self.ch_params,
			"device": next(self.parameters()).device,
			"dtype": next(self.parameters()).dtype,
		}
		
		return info
	
	def estimate_memory_usage(self, input_shape: Tuple[int, ...]) -> Dict:
		"""估算内存使用量"""
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
		
		# 创建模型
		model = cls(ch_params=ch_params)
		
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