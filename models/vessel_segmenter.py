import torch
import torch.nn as nn
import torch.nn.functional as F

from .ch_branch import CHBranch
from .spatial_branch import SpatialBranch
from .fusion import AttentionFusion, MultiscaleFusion


class VesselSegmenter(nn.Module):
	"""肝脏血管分割模型"""
	
	def __init__(self, in_channels=1, out_channels=1, ch_params=None, tier_params=None):
		"""
		初始化分割模型

		参数:
			in_channels: 输入通道数
			out_channels: 输出通道数 (1用于二分类，>1用于多分类)
			ch_params: CH支路参数 {max_n, max_k, max_l, cylindrical_dims}
			tier_params: 不同tier的特定参数 {0: {}, 1: {}, 2: {}}
		"""
		super().__init__()
		
		# 默认CH参数
		if ch_params is None:
			ch_params = {
				'max_n': 3,
				'max_k': 4,
				'max_l': 5,
				'cylindrical_dims': (32, 36, 32)
			}
		
		# 默认tier参数
		if tier_params is None:
			tier_params = {
				0: {'max_n': 2, 'max_k': 3, 'max_l': 4},  # 器官级别
				1: {'max_n': 3, 'max_k': 4, 'max_l': 5},  # 结构级别
				2: {'max_n': 4, 'max_k': 5, 'max_l': 6}  # 细节级别
			}
		
		self.tier_params = tier_params
		self.current_tier = None
		
		# 特征通道数
		ch_channels = 16
		spatial_channels = 16
		
		# 初始化分支
		self.ch_branch = CHBranch(**ch_params)
		self.spatial_branch = SpatialBranch(in_channels, 16, spatial_channels)
		
		# 特征融合
		self.attention_fusion = AttentionFusion(ch_channels, spatial_channels)
		
		# 多尺度融合
		self.multiscale_fusion = MultiscaleFusion(ch_channels + spatial_channels)
		
		# 分割头
		self.segmentation_head = nn.Sequential(
			nn.Conv3d(ch_channels + spatial_channels, 32, kernel_size=3, padding=1),
			nn.InstanceNorm3d(32),
			nn.ReLU(inplace=True),
			nn.Conv3d(32, out_channels, kernel_size=1),
			nn.Sigmoid() if out_channels == 1 else nn.Softmax(dim=1)
		)
		
		# 缓存不同tier的特征
		self.tier_features = {}
	
	def set_tier(self, tier):
		"""设置当前tier"""
		self.current_tier = tier
		self.ch_branch.set_tier(tier)
		
		# 清除缓存的特征
		self.tier_features = {}
	
	def forward(self, x, tier=None):
		"""
		前向传播

		参数:
			x: 输入体积 [B, C, D, H, W]
			tier: 当前tier，如果不是None则更新current_tier

		返回:
			分割结果 [B, out_channels, D, H, W]
		"""
		# 更新tier
		if tier is not None:
			self.set_tier(tier)
		
		# 确保已设置tier
		if self.current_tier is None:
			raise ValueError("Tier must be set before forward pass")
		
		# 获取tier特定参数
		tier_ch_params = self.tier_params.get(self.current_tier, {})
		r_scale = tier_ch_params.get('r_scale', 1.0)
		
		# CH支路
		ch_features = self.ch_branch(x, r_scale=r_scale)
		
		# 空间支路
		spatial_features = self.spatial_branch(x)
		
		# 特征融合
		fused_features = self.attention_fusion(ch_features, spatial_features)
		
		# 缓存当前tier的特征
		self.tier_features[self.current_tier] = fused_features
		
		# 如果所有tier都已处理，执行多尺度融合
		if len(self.tier_features) > 1:
			final_features = self.multiscale_fusion(self.tier_features)
		else:
			final_features = fused_features
		
		# 分割头
		output = self.segmentation_head(final_features)
		
		return output