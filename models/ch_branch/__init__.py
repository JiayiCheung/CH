from .fft_utils import FFTUtils
from .cylindrical_mapping import CylindricalMapping
from .ch_transform import CHTransform
from .ch_attention import CHAttention


class CHBranch(nn.Module):
	"""柱谐波支路的完整实现"""
	
	def __init__(self, max_n=3, max_k=4, max_l=5, cylindrical_dims=(32, 36, 32)):
		"""
		初始化CH支路

		参数:
			max_n: 最大角谐波阶数
			max_k: 最大径向阶数
			max_l: 最大轴向阶数
			cylindrical_dims: 柱坐标维度 (r_samples, theta_samples, z_samples)
		"""
		super().__init__()
		
		# CH参数
		self.max_n = max_n
		self.max_k = max_k
		self.max_l = max_l
		
		# 柱坐标维度
		self.r_samples, self.theta_samples, self.z_samples = cylindrical_dims
		
		# 组件初始化
		self.fft_utils = FFTUtils()
		self.cylindrical_mapping = CylindricalMapping(*cylindrical_dims)
		self.ch_transform = CHTransform(max_n, max_k, max_l)
		self.ch_attention = CHAttention(max_n, max_k, max_l)
		
		# 当前tier设置
		self.current_tier = None
	
	def set_tier(self, tier):
		"""设置当前tier"""
		self.current_tier = tier
	
	def forward(self, x, r_scale=1.0):
		"""
		前向传播

		参数:
			x: 输入体积 [B, C, D, H, W]
			r_scale: 径向尺度因子

		返回:
			处理后的体积 [B, C, D, H, W]
		"""
		# 1. 3D FFT变换
		spectrum = self.fft_utils.fft3d(x, apply_window=True)
		
		# 2. 柱坐标映射
		cylindrical_spectrum = self.cylindrical_mapping.cartesian_to_cylindrical(spectrum)
		
		# 3. CH分解
		ch_coeffs = self.ch_transform.decompose(cylindrical_spectrum, r_scale=r_scale)
		
		# 4. CH系数注意力
		if self.current_tier is not None:
			enhanced_coeffs = self.ch_attention.tier_specific_enhancement(ch_coeffs, self.current_tier)
		else:
			enhanced_coeffs = self.ch_attention(ch_coeffs)
		
		# 5. 逆CH变换
		reconstructed_cylindrical = self.ch_transform.reconstruct(
			enhanced_coeffs,
			(self.r_samples, self.theta_samples, self.z_samples),
			r_scale=r_scale
		)
		
		# 6. 柱坐标逆映射
		reconstructed_spectrum = self.cylindrical_mapping.cylindrical_to_cartesian(
			reconstructed_cylindrical,
			(x.shape[2], x.shape[3], x.shape[4])
		)
		
		# 7. 3D逆FFT变换
		output = self.fft_utils.ifft3d(reconstructed_spectrum, output_shape=x.shape[2:])
		
		return output


import torch.nn as nn

__all__ = ['FFTUtils', 'CylindricalMapping', 'CHTransform', 'CHAttention', 'CHBranch']