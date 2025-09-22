from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ch_branch.ch_branch import CHBranch
from .spatial_branch import SpatialBranch
from .fusion.attention_fusion import AttentionFusion
from .fusion.seg import Seg
from .spatial_branch.edge_enhancement import EdgeEnhancement

__all__ = ["VesselSegmenter"]






class VesselSegmenterAblation(nn.Module):
	"""
	血管分割模型 - 消融实验版本
	移除CH分支和空间分支，使用简单的编码器-解码器结构
	"""
	
	def __init__(self, in_channels=1, out_channels=1):
		super().__init__()
		
		# 简单的编码器
		self.encoder = nn.Sequential(
			nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
			nn.InstanceNorm3d(16),
			nn.ReLU(inplace=True),
			
			nn.Conv3d(16, 32, kernel_size=3, padding=1),
			nn.InstanceNorm3d(32),
			nn.ReLU(inplace=True),
			
			nn.Conv3d(32, 64, kernel_size=3, padding=1),
			nn.InstanceNorm3d(64),
			nn.ReLU(inplace=True)
		)
		
		# 使用与原始模型相同的分割头
		self.seg_head = Seg(
			in_channels=64,  # 编码器输出通道数
			out_channels=out_channels,
			features=[32, 64, 128, 256],
			bilinear=True
		)
		
		if out_channels == 1:
			self.final_activation = nn.Sigmoid()
		else:
			self.final_activation = nn.Softmax(dim=1)
	
	def forward(self, x):
		# 输入验证
		if x.dim() != 5:
			raise ValueError(f"Expected 5D input (B,C,D,H,W), got {x.dim()}D")
		
		# 内存格式优化
		x = x.to(memory_format=torch.channels_last_3d)
		
		# 直接编码
		features = self.encoder(x)
		
		# 分割头处理
		x = self.seg_head(features)
		
		# 应用激活函数
		x = self.final_activation(x)
		
		return x