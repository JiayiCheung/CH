import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiscaleFusion(nn.Module):
	"""多尺度融合模块，用于合并不同tier的结果"""
	
	def __init__(self, in_channels):
		"""
		初始化多尺度融合模块

		参数:
			in_channels: 输入特征通道数
		"""
		super().__init__()
		
		# 注意力权重网络
		self.attention = nn.Sequential(
			nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
			nn.InstanceNorm3d(in_channels // 2),
			nn.ReLU(inplace=True),
			nn.Conv3d(in_channels // 2, 3, kernel_size=1),  # 输出3个权重，对应三个tier
			nn.Softmax(dim=1)  # 确保权重和为1
		)
	
	def forward(self, tier_features, target_shape=None):
		"""
		融合不同tier的特征

		参数:
			tier_features: 字典，包含不同tier的特征 {0: tensor, 1: tensor, 2: tensor}
			target_shape: 目标形状，默认使用Tier-0的形状

		返回:
			融合后的特征
		"""
		# 检查输入
		if not tier_features:
			raise ValueError("No tier features provided")
		
		# 获取所有可用的tier
		available_tiers = sorted(tier_features.keys())
		
		# 确定目标形状
		if target_shape is None:
			if 0 in tier_features:
				target_shape = tier_features[0].shape[2:]
			else:
				target_shape = tier_features[available_tiers[0]].shape[2:]
		
		# 调整所有tier的特征到相同的形状
		aligned_features = {}
		for tier in available_tiers:
			features = tier_features[tier]
			if features.shape[2:] != target_shape:
				aligned_features[tier] = F.interpolate(
					features, size=target_shape,
					mode='trilinear', align_corners=True
				)
			else:
				aligned_features[tier] = features
		
		# 当只有一个tier时，直接返回
		if len(aligned_features) == 1:
			return aligned_features[available_tiers[0]]
		
		# 创建输入张量
		# 对于缺失的tier，使用零张量
		all_features = []
		B, C = aligned_features[available_tiers[0]].shape[:2]
		
		for tier in range(3):  # 总是处理所有3个tier
			if tier in aligned_features:
				all_features.append(aligned_features[tier])
			else:
				dummy = torch.zeros(
					(B, C, *target_shape),
					dtype=aligned_features[available_tiers[0]].dtype,
					device=aligned_features[available_tiers[0]].device
				)
				all_features.append(dummy)
		
		# 拼接特征用于计算注意力权重
		stacked = torch.stack(all_features, dim=1)  # [B, 3, C, D, H, W]
		
		# 计算权重
		# 首先调整形状以适应attention网络
		B, T, C, D, H, W = stacked.shape
		reshaped = stacked.permute(0, 2, 1, 3, 4, 5).reshape(B, C, T * D, H, W)
		
		# 应用注意力获取权重
		attention_map = self.attention(reshaped)  # [B, 3, T*D, H, W]
		
		# 调整回原始形状
		attention_map = attention_map.reshape(B, 3, T, D, H, W).permute(0, 2, 1, 3, 4, 5)  # [B, T, 3, D, H, W]
		
		# 应用权重并求和
		weighted_sum = torch.sum(stacked * attention_map, dim=1)  # [B, C, D, H, W]
		
		return weighted_sum