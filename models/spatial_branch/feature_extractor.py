import torch
import torch.nn as nn


class DepthwiseSeparableConv3d(nn.Module):
	"""深度可分离3D卷积"""
	
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
		"""
		初始化深度可分离3D卷积

		参数:
			in_channels: 输入通道数
			out_channels: 输出通道数
			kernel_size: 卷积核大小
			stride: 步长
			padding: 填充
		"""
		super().__init__()
		
		# 深度卷积
		self.depthwise = nn.Conv3d(
			in_channels, in_channels, kernel_size=kernel_size,
			stride=stride, padding=padding, groups=in_channels
		)
		
		# 逐点卷积
		self.pointwise = nn.Conv3d(
			in_channels, out_channels, kernel_size=1
		)
	
	def forward(self, x):
		"""前向传播"""
		x = self.depthwise(x)
		x = self.pointwise(x)
		return x


class SpatialFeatureExtractor(nn.Module):
	"""轻量级空间特征提取器"""
	
	def __init__(self, in_channels=1, mid_channels=16, out_channels=16):
		"""
		初始化空间特征提取器

		参数:
			in_channels: 输入通道数
			mid_channels: 中间通道数
			out_channels: 输出通道数
		"""
		super().__init__()
		
		# 特征提取网络
		self.encoder = nn.Sequential(
			# 第一层: 标准3D卷积
			nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
			nn.InstanceNorm3d(mid_channels),
			nn.ReLU(inplace=True),
			
			# 第二层: 深度可分离卷积
			DepthwiseSeparableConv3d(mid_channels, mid_channels * 2, kernel_size=3),
			nn.InstanceNorm3d(mid_channels * 2),
			nn.ReLU(inplace=True),
			
			# 第三层: 深度可分离卷积
			DepthwiseSeparableConv3d(mid_channels * 2, out_channels, kernel_size=3),
			nn.InstanceNorm3d(out_channels),
			nn.ReLU(inplace=True)
		)
		
		# 残差连接
		self.shortcut = None
		if in_channels != out_channels:
			self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
	
	def forward(self, x):
		"""
		前向传播

		参数:
			x: 输入体积 [B, C, D, H, W]

		返回:
			提取的特征 [B, out_channels, D, H, W]
		"""
		# 提取特征
		features = self.encoder(x)
		
		# 添加残差连接
		if self.shortcut:
			residual = self.shortcut(x)
		else:
			residual = x
		
		# 防止尺寸不匹配
		if residual.shape[2:] != features.shape[2:]:
			residual = nn.functional.interpolate(
				residual, size=features.shape[2:], mode='trilinear', align_corners=True
			)
		
		# 残差连接
		output = features + residual
		
		return output