import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
	"""通道注意力模块"""
	
	def __init__(self, channels, reduction_ratio=16):
		"""
		初始化通道注意力

		参数:
			channels: 输入通道数
			reduction_ratio: 降维比例
		"""
		super().__init__()
		
		# 确保降维后至少有8个通道
		reduced_channels = max(8, channels // reduction_ratio)
		
		# 全局平均池化后的特征变换
		self.avg_pool = nn.AdaptiveAvgPool3d(1)
		self.max_pool = nn.AdaptiveMaxPool3d(1)
		
		# 共享MLP
		self.mlp = nn.Sequential(
			nn.Conv3d(channels, reduced_channels, kernel_size=1, bias=False),
			nn.ReLU(inplace=True),
			nn.Conv3d(reduced_channels, channels, kernel_size=1, bias=False)
		)
		
		self.sigmoid = nn.Sigmoid()
	
	def forward(self, x):
		"""
		前向传播

		参数:
			x: 输入特征 [B, C, D, H, W]

		返回:
			注意力权重 [B, C, 1, 1, 1]
		"""
		# 平均池化分支
		avg_out = self.mlp(self.avg_pool(x))
		
		# 最大池化分支
		max_out = self.mlp(self.max_pool(x))
		
		# 合并两个分支
		out = avg_out + max_out
		
		return self.sigmoid(out)


class SpatialAttention(nn.Module):
	"""空间注意力模块"""
	
	def __init__(self, kernel_size=7):
		"""
		初始化空间注意力

		参数:
			kernel_size: 卷积核大小
		"""
		super().__init__()
		
		# 确保kernel_size是奇数
		assert kernel_size % 2 == 1, "Kernel size must be odd"
		
		padding = kernel_size // 2
		
		# 通道聚合+卷积
		self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding)
		self.sigmoid = nn.Sigmoid()
	
	def forward(self, x):
		"""
		前向传播

		参数:
			x: 输入特征 [B, C, D, H, W]

		返回:
			注意力权重 [B, 1, D, H, W]
		"""
		# 通道维度上的平均池化和最大池化
		avg_out = torch.mean(x, dim=1, keepdim=True)
		max_out, _ = torch.max(x, dim=1, keepdim=True)
		
		# 合并池化结果
		out = torch.cat([avg_out, max_out], dim=1)
		
		# 应用卷积和激活
		out = self.conv(out)
		
		return self.sigmoid(out)


class AttentionFusion(nn.Module):
	"""双重注意力特征融合模块"""
	
	def __init__(self, ch_channels, spatial_channels):
		"""
		初始化特征融合模块

		参数:
			ch_channels: CH支路特征通道数
			spatial_channels: 空间支路特征通道数
		"""
		super().__init__()
		
		# 合并后的通道数
		self.combined_channels = ch_channels + spatial_channels
		
		# 通道注意力
		self.channel_attention = ChannelAttention(self.combined_channels)
		
		# 空间注意力
		self.spatial_attention = SpatialAttention(kernel_size=7)
		
		# 特征融合卷积
		self.fusion_conv = nn.Sequential(
			nn.Conv3d(self.combined_channels, self.combined_channels, kernel_size=3, padding=1),
			nn.InstanceNorm3d(self.combined_channels),
			nn.ReLU(inplace=True)
		)
	
	def forward(self, ch_features, spatial_features):
		"""
		前向传播

		参数:
			ch_features: CH支路特征 [B, ch_channels, D, H, W]
			spatial_features: 空间支路特征 [B, spatial_channels, D, H, W]

		返回:
			融合特征 [B, ch_channels+spatial_channels, D, H, W]
		"""
		# 确保空间维度匹配
		if ch_features.shape[2:] != spatial_features.shape[2:]:
			spatial_features = F.interpolate(
				spatial_features, size=ch_features.shape[2:],
				mode='trilinear', align_corners=True
			)
		
		# 合并特征
		combined = torch.cat([ch_features, spatial_features], dim=1)
		
		# 应用通道注意力
		channel_weights = self.channel_attention(combined)
		channel_refined = combined * channel_weights
		
		# 应用空间注意力
		spatial_weights = self.spatial_attention(channel_refined)
		spatial_refined = channel_refined * spatial_weights
		
		# 残差连接
		refined = spatial_refined + combined
		
		# 应用融合卷积
		output = self.fusion_conv(refined)
		
		return output