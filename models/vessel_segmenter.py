#!/usr/bin/env python3
"""
血管分割模型
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Union, Tuple

import torch
import torch.nn as nn


from .fd_branch.FDBranch import FDBranch
from .spatial_branch import SpatialBranch
from .fusion.attention_fusion import FreqGuidedCrossAttn
from .fusion.seg import Seg


__all__ = ["VesselSegmenter"]


class VesselSegmenter(nn.Module):
	"""
	血管分割模型
	基于频域特征增强的血管分割模型，结合空间支路和频域特征
	"""
	
	def __init__(
			self,
			in_channels: int = 1,
			out_channels: int = 1,
			expansion_ratio: int = 4,
			config: Dict = None,
	) -> None:
		super().__init__()
		
		# 默认配置
		if config is None:
			config = {}
		
		# 获取FDBranch配置参数
	
		
		
		# 核心组件
		self.fd_branch = FDBranch(
			in_channels=in_channels,
			out_channels=4,
			expansion_ratio=expansion_ratio
		)
		
		self.spatial_branch = SpatialBranch(in_channels, 16, 16)
		
		# 融合模块
		self.attention_fusion = FreqGuidedCrossAttn(
			fd_channels=4,
			sp_channels=16,
			out_channels=20,
			num_bands=4,
			num_heads=4,
			dropout=0.1,
			fdbranch=self.fd_branch
		)
		
		self.seg_head = Seg(
			in_channels=20,
			out_channels=out_channels,
			depth=5,
			base_features=32,
			feature_scale=2,
			bilinear=True
		)
		
		if out_channels == 1:
			self.final_activation = nn.Sigmoid()
		else:
			self.final_activation = nn.Softmax(dim=1)
	
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
		
		# 频域分支处理
		fd_features = self.fd_branch(x)
		
		# 空间分支处理
		spatial_features = self.spatial_branch(x)
		
		# 特征融合
		fused_features = self.attention_fusion(fd_features, spatial_features)
		
		# 分割头处理
		x = self.seg_head(fused_features)
		
		
		
		return x
	
	def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		返回中间特征的前向传播，用于分析和可视化

		参数:
			x: 输入张量

		返回:
			(fd_features, spatial_features, fused_features)
		"""
		x = x.to(memory_format=torch.channels_last_3d)
		
		# 提取特征
		fd_features = self.fd_branch(x)
		spatial_features = self.spatial_branch(x)
		
		fused_features = self.attention_fusion(fd_features, spatial_features)
		
		return fd_features, spatial_features, fused_features
	
	def get_model_info(self) -> Dict:
		"""获取模型信息"""
		total_params = sum(p.numel() for p in self.parameters())
		trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
		
		info = {
			"model_name": "VesselSegmenter",
			"total_parameters": total_params,
			"trainable_parameters": trainable_params,
			"fd_branch_expansion_ratio": getattr(self.fd_branch, 'expansion_ratio', 2),
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
			'fd_branch_expansion_ratio': getattr(self.fd_branch, 'expansion_ratio', 2),
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
		
		checkpoint = torch.load(filepath, map_location=device, weights_only=False)
		
		# 创建配置
		config = {
			'fd_branch': {
				'expansion_ratio': checkpoint.get('fd_branch_expansion_ratio', 2)
			}
		}
		
		# 创建模型
		model = cls(config=config)
		
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