import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class CHTransform(nn.Module):
	"""柱谐波变换实现"""
	
	def __init__(self, max_n=3, max_k=4, max_l=5):
		"""
		初始化柱谐波变换

		参数:
			max_n: 最大角谐波阶数
			max_k: 最大径向阶数
			max_l: 最大轴向阶数
		"""
		super().__init__()
		self.max_n = max_n
		self.max_k = max_k
		self.max_l = max_l
		
		# 预计算Bessel函数零点
		self.bessel_zeros = self._precompute_bessel_zeros()
	
	def _precompute_bessel_zeros(self):
		"""
		预计算Bessel函数零点

		返回:
			Bessel函数零点的字典 {n: [zeros]}
		"""
		# 注意：这里使用近似值，对于更高精度可以使用查表
		bessel_zeros = {}
		
		# 前几个Bessel函数J_n的第k个零点的近似值
		# 这些值可以从数学表或scipy.special.jn_zeros获取
		
		# J_0的零点
		bessel_zeros[0] = [2.4048, 5.5201, 8.6537, 11.7915, 14.9309]
		
		# J_1的零点
		bessel_zeros[1] = [3.8317, 7.0156, 10.1735, 13.3237, 16.4706]
		
		# J_2的零点
		bessel_zeros[2] = [5.1356, 8.4172, 11.6198, 14.7960, 18.0155]
		
		# J_3的零点
		bessel_zeros[3] = [6.3802, 9.7610, 13.0152, 16.2235, 19.4094]
		
		# J_4的零点
		bessel_zeros[4] = [7.5883, 11.0647, 14.3725, 17.6160, 20.8269]
		
		# J_5的零点 (如果需要)
		bessel_zeros[5] = [8.7715, 12.3386, 15.7002, 18.9801, 22.2178]
		
		return bessel_zeros
	
	def _get_bessel_zero(self, n, k):
		"""
		获取Bessel函数J_n的第k个零点

		参数:
			n: 角谐波阶数
			k: 径向阶数（从1开始）

		返回:
			对应的Bessel零点
		"""
		if n < len(self.bessel_zeros) and k <= len(self.bessel_zeros[n]):
			return self.bessel_zeros[n][k - 1]  # k是从1开始的
		else:
			# 对于未预计算的零点，使用近似公式
			if k == 1:
				# 第一个零点的近似
				return n + 1.85575 * n ** (1 / 3) + 1.033150 * n ** (-1 / 3)
			else:
				# 更高阶零点的近似
				return (k + 0.5 * n - 0.25) * np.pi
	
	def _bessel_j_approx(self, n, x):
		"""
		Bessel函数的近似计算

		参数:
			n: 阶数
			x: 输入值（张量）

		返回:
			Bessel函数值的近似
		"""
		# 实现简化的Bessel函数近似
		# 对于实际实现，应该使用更精确的方法
		
		if n == 0:
			# J_0(x)的近似
			small_x = (x.abs() < 1.0)
			medium_x = ((x.abs() >= 1.0) & (x.abs() < 5.0))
			large_x = (x.abs() >= 5.0)
			
			result = torch.zeros_like(x)
			
			# 小x近似
			if small_x.any():
				x_small = x[small_x]
				result[small_x] = 1 - (x_small ** 2) / 4 + (x_small ** 4) / 64
			
			# 中等x近似
			if medium_x.any():
				x_med = x[medium_x]
				result[medium_x] = torch.cos(x_med - torch.pi / 4) / torch.sqrt(x_med)
			
			# 大x近似
			if large_x.any():
				x_large = x[large_x]
				factor = torch.sqrt(2 / (torch.pi * x_large))
				phase = x_large - torch.pi / 4
				result[large_x] = factor * torch.cos(phase)
			
			return result
		
		elif n == 1:
			# J_1(x)的近似
			small_x = (x.abs() < 1.0)
			medium_x = ((x.abs() >= 1.0) & (x.abs() < 5.0))
			large_x = (x.abs() >= 5.0)
			
			result = torch.zeros_like(x)
			
			# 小x近似
			if small_x.any():
				x_small = x[small_x]
				result[small_x] = x_small / 2 - (x_small ** 3) / 16
			
			# 中等x近似
			if medium_x.any():
				x_med = x[medium_x]
				result[medium_x] = torch.sin(x_med - torch.pi / 4) / torch.sqrt(x_med)
			
			# 大x近似
			if large_x.any():
				x_large = x[large_x]
				factor = torch.sqrt(2 / (torch.pi * x_large))
				phase = x_large - 3 * torch.pi / 4
				result[large_x] = factor * torch.cos(phase)
			
			return result
		
		else:
			# 更高阶的近似
			small_x = (x.abs() < 0.1 * n)
			large_x = ~small_x
			
			result = torch.zeros_like(x)
			
			# 小x近似
			if small_x.any():
				x_small = x[small_x]
				# J_n(x) ≈ (x/2)^n / n! for small x
				log_term = n * torch.log(x_small / 2) - torch.tensor(sum(math.log(i) for i in range(1, n + 1)),
				                                                     device=x.device)
				result[small_x] = torch.exp(log_term)
			
			# 大x近似
			if large_x.any():
				x_large = x[large_x]
				factor = torch.sqrt(2 / (torch.pi * x_large))
				phase = x_large - (2 * n + 1) * torch.pi / 4
				result[large_x] = factor * torch.cos(phase)
			
			return result
	
	def decompose(self, cylindrical_volume, r_scale=1.0):
		"""
		执行柱谐波分解

		参数:
			cylindrical_volume: 柱坐标表示的体积 [B, C, r, theta, z]
			r_scale: 径向尺度因子

		返回:
			CH系数 [B, C, 2*max_n+1, max_k, 2*max_l+1]
		"""
		B, C, R, T, Z = cylindrical_volume.shape
		
		# 创建结果张量
		ch_coeffs = torch.zeros(
			(B, C, 2 * self.max_n + 1, self.max_k, 2 * self.max_l + 1),
			dtype=torch.complex64, device=cylindrical_volume.device
		)
		
		# 创建柱坐标网格
		r = torch.linspace(0, 1, R, device=cylindrical_volume.device) * r_scale
		theta = torch.linspace(0, 2 * math.pi, T, device=cylindrical_volume.device)
		z = torch.linspace(-1, 1, Z, device=cylindrical_volume.device)
		
		r_grid, theta_grid, z_grid = torch.meshgrid(r, theta, z, indexing='ij')
		
		# 创建积分权重 (用于加权求和近似积分)
		dr = r_scale / (R - 1)
		dtheta = 2 * math.pi / T
		dz = 2.0 / (Z - 1)
		
		# r方向的权重 (包括r因子)
		r_weight = r_grid * dr
		
		# 创建角度谐波
		n_range = torch.arange(-self.max_n, self.max_n + 1, device=cylindrical_volume.device)
		angular_harmonics = []
		
		for n in n_range:
			# e^(i*n*theta) / sqrt(2*pi)
			angular_harmonic = torch.exp(1j * n * theta_grid) / math.sqrt(2 * math.pi)
			angular_harmonics.append(angular_harmonic)
		
		# 创建轴向谐波
		l_range = torch.arange(-self.max_l, self.max_l + 1, device=cylindrical_volume.device)
		axial_harmonics = []
		
		for l in l_range:
			# e^(i*l*pi*z) / sqrt(2)
			axial_harmonic = torch.exp(1j * l * math.pi * z_grid) / math.sqrt(2)
			axial_harmonics.append(axial_harmonic)
		
		# 计算系数
		for b in range(B):
			for c in range(C):
				volume_data = cylindrical_volume[b, c]
				
				for n_idx, n in enumerate(range(-self.max_n, self.max_n + 1)):
					angular_harmonic = angular_harmonics[n_idx]
					
					for k_idx, k in enumerate(range(1, self.max_k + 1)):
						# 获取Bessel零点
						r_nk = self._get_bessel_zero(abs(n), k)
						
						# 计算Bessel函数
						bessel_vals = self._bessel_j_approx(abs(n), r_nk * r_grid)
						
						# 归一化Bessel函数
						bessel_sum_squares = torch.sum((bessel_vals * r_weight) ** 2)
						if bessel_sum_squares > 1e-6:
							bessel_norm = 1.0 / torch.sqrt(bessel_sum_squares)
						else:
							bessel_norm = torch.tensor(0.0, device=bessel_vals.device)
						bessel_vals = bessel_vals * bessel_norm
						
						for l_idx, l in enumerate(range(-self.max_l, self.max_l + 1)):
							axial_harmonic = axial_harmonics[l_idx]
							
							# 创建完整的基函数
							basis = bessel_vals * angular_harmonic * axial_harmonic
							
							# 计算内积 (使用加权求和近似积分)
							weight = r_weight * dtheta * dz
							
							# 计算系数
							ch_coeffs[b, c, n_idx, k_idx, l_idx] = torch.sum(volume_data * basis * weight)
		
		return ch_coeffs
	
	def reconstruct(self, ch_coeffs, output_shape, r_scale=1.0):
		"""
		从CH系数重构体积

		参数:
			ch_coeffs: CH系数 [B, C, 2*max_n+1, max_k, 2*max_l+1]
			output_shape: 输出形状 (R, T, Z)
			r_scale: 径向尺度因子

		返回:
			重构的柱坐标体积 [B, C, R, T, Z]
		"""
		B, C = ch_coeffs.shape[:2]
		R, T, Z = output_shape
		
		# 创建输出张量
		reconstructed = torch.zeros(
			(B, C, R, T, Z),
			dtype=torch.float32, device=ch_coeffs.device
		)
		
		# 创建柱坐标网格
		r = torch.linspace(0, 1, R, device=ch_coeffs.device) * r_scale
		theta = torch.linspace(0, 2 * math.pi, T, device=ch_coeffs.device)
		z = torch.linspace(-1, 1, Z, device=ch_coeffs.device)
		
		r_grid, theta_grid, z_grid = torch.meshgrid(r, theta, z, indexing='ij')
		
		# 创建角度谐波
		n_range = torch.arange(-self.max_n, self.max_n + 1, device=ch_coeffs.device)
		angular_harmonics = []
		
		for n in n_range:
			# e^(i*n*theta) / sqrt(2*pi)
			angular_harmonic = torch.exp(1j * n * theta_grid) / math.sqrt(2 * math.pi)
			angular_harmonics.append(angular_harmonic)
		
		# 创建轴向谐波
		l_range = torch.arange(-self.max_l, self.max_l + 1, device=ch_coeffs.device)
		axial_harmonics = []
		
		for l in l_range:
			# e^(i*l*pi*z) / sqrt(2)
			axial_harmonic = torch.exp(1j * l * math.pi * z_grid) / math.sqrt(2)
			axial_harmonics.append(axial_harmonic)
		
		# 重构体积
		for b in range(B):
			for c in range(C):
				for n_idx, n in enumerate(range(-self.max_n, self.max_n + 1)):
					angular_harmonic = angular_harmonics[n_idx]
					
					for k_idx, k in enumerate(range(1, self.max_k + 1)):
						# 获取Bessel零点
						r_nk = self._get_bessel_zero(abs(n), k)
						
						# 计算Bessel函数
						bessel_vals = self._bessel_j_approx(abs(n), r_nk * r_grid)
						
						# 归一化Bessel函数
						bessel_sum_squares = torch.sum((bessel_vals * r_grid) ** 2)
						if bessel_sum_squares > 1e-6:
							bessel_norm = 1.0 / torch.sqrt(bessel_sum_squares)
						else:
							bessel_norm = torch.tensor(0.0, device=bessel_vals.device)
						bessel_vals = bessel_vals * bessel_norm
						
						for l_idx, l in enumerate(range(-self.max_l, self.max_l + 1)):
							axial_harmonic = axial_harmonics[l_idx]
							
							# 获取系数
							coeff = ch_coeffs[b, c, n_idx, k_idx, l_idx]
							
							# 创建完整的基函数
							basis = bessel_vals * angular_harmonic * axial_harmonic
							
							# 将基函数与系数相乘并累加
							reconstructed[b, c] += (coeff * basis).real
		
		return reconstructed