import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv3d(nn.Module):
	"""轻量化的深度可分离3D卷积模块"""
	
	def __init__(self, in_ch, out_ch, k=3, stride=1, pad=None):
		super().__init__()
		pad = k // 2 if pad is None else pad
		# 深度卷积 - 每个通道独立卷积
		self.dw = nn.Conv3d(in_ch, in_ch, k, stride, pad, groups=in_ch, bias=False)
		# 逐点卷积 - 1x1x1卷积混合通道信息
		self.pw = nn.Conv3d(in_ch, out_ch, 1, bias=False)
	
	def forward(self, x):
		return self.pw(self.dw(x))


class SpatialFeatureExtractor(nn.Module):
	"""轻量级空间特征提取器"""
	
	def __init__(self, in_channels=1, mid_channels=16, out_channels=16):
		super().__init__()
		
		# 使用GroupNorm替代BatchNorm，更适合3D医学图像
		def _gn(c):
			return nn.GroupNorm(num_groups=max(4, c // 4), num_channels=c, affine=True)
		
		# 第一层深度可分离卷积块
		self.encoder = nn.Sequential(
			nn.Conv3d(in_channels, mid_channels, 3, padding=1, bias=False),
			_gn(mid_channels), nn.ReLU(inplace=True),
			
			DepthwiseSeparableConv3d(mid_channels, out_channels),
			_gn(out_channels), nn.ReLU(inplace=True),
		)
		
		# 残差连接
		self.shortcut = (
			nn.Conv3d(in_channels, out_channels, 1, bias=False)
			if in_channels != out_channels else nn.Identity()
		)
	
	def forward(self, x):
		feat = self.encoder(x)
		res = self.shortcut(x)
		
		# 空间形状检查，确保可以相加
		if res.shape[2:] != feat.shape[2:]:
			res = F.interpolate(res, size=feat.shape[2:], mode="trilinear", align_corners=True)
		
		return feat + res


class EdgeEnhancement(nn.Module):
	"""边缘增强模块"""
	
	def __init__(self, in_channels=1, out_channels=8):
		super().__init__()
		
		# 简化的边缘检测卷积
		self.edge_conv = nn.Sequential(
			nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
			nn.GroupNorm(max(4, out_channels // 4), out_channels),
			nn.ReLU(inplace=True)
		)
	
	def forward(self, x):
		return self.edge_conv(x)


class SpatialBranch(nn.Module):
	"""轻量级空间支路，结合边缘增强和特征提取"""
	
	def __init__(self, in_channels=1, mid_channels=16, out_channels=16):
		super().__init__()
		
		# 边缘增强模块
		self.edge_enhancement = EdgeEnhancement(in_channels, mid_channels // 2)
		
		# 特征提取模块
		self.feature_extractor = SpatialFeatureExtractor(
			in_channels,
			mid_channels,
			mid_channels
		)
		
		# 特征融合模块
		self.fusion = nn.Sequential(
			nn.Conv3d(
				mid_channels + mid_channels // 2,  # 输入: 特征提取 + 边缘增强
				out_channels,  # 输出: 最终通道数
				kernel_size=3,
				padding=1
			),
			nn.InstanceNorm3d(out_channels),
			nn.ReLU(inplace=True)
		)
	
	def forward(self, x):
		# 1. 边缘增强处理 - 提取边缘特征
		edge_features = self.edge_enhancement(x)
		
		# 2. 空间特征提取
		spatial_features = self.feature_extractor(x)
		
		# 3. 特征融合 - 拼接后使用卷积整合
		combined = torch.cat([spatial_features, edge_features], dim=1)
		output = self.fusion(combined)
		
		return output

__all__ = ['SpatialBranch', 'SpatialFeatureExtractor', 'EdgeEnhancement', 'DepthwiseSeparableConv3d']