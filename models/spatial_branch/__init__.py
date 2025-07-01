import torch
import torch.nn as nn

from .feature_extractor import SpatialFeatureExtractor, DepthwiseSeparableConv3d
from .edge_enhancement import EdgeEnhancement


class SpatialBranch(nn.Module):
	"""轻量级空间支路"""
	
	def __init__(self, in_channels=1, mid_channels=16, out_channels=16):
		"""
		初始化空间支路

		参数:
			in_channels: 输入通道数
			mid_channels: 中间通道数
			out_channels: 输出通道数
		"""
		super().__init__()
		
		# 边缘增强模块
		self.edge_enhancement = EdgeEnhancement(in_channels, mid_channels // 2)
		
		# 特征提取模块
		self.feature_extractor = SpatialFeatureExtractor(in_channels, mid_channels, mid_channels)
		
		# 特征融合
		self.fusion = nn.Sequential(
			nn.Conv3d(mid_channels + mid_channels // 2, out_channels, kernel_size=3, padding=1),
			nn.InstanceNorm3d(out_channels),
			nn.ReLU(inplace=True)
		)
	
	def forward(self, x):
		"""
		前向传播

		参数:
			x: 输入体积 [B, C, D, H, W]

		返回:
			空间特征 [B, out_channels, D, H, W]
		"""
		# 边缘增强
		edge_features = self.edge_enhancement(x)
		
		# 特征提取
		spatial_features = self.feature_extractor(x)
		
		# 特征融合
		combined = torch.cat([spatial_features, edge_features], dim=1)
		output = self.fusion(combined)
		
		return output


__all__ = ['SpatialBranch', 'SpatialFeatureExtractor', 'EdgeEnhancement', 'DepthwiseSeparableConv3d']