import torch
import torch.nn.functional as F
import numpy as np


class FFTUtils:
	"""3D FFT变换工具类"""
	
	@staticmethod
	def apply_hanning_window(volume):
		"""
		应用Hanning窗函数减少频谱泄漏

		参数:
			volume: 输入体积 [B, C, D, H, W]

		返回:
			应用窗函数后的体积
		"""
		B, C, D, H, W = volume.shape
		
		# 创建1D窗函数
		window_d = torch.hann_window(D, periodic=False, dtype=volume.dtype, device=volume.device)
		window_h = torch.hann_window(H, periodic=False, dtype=volume.dtype, device=volume.device)
		window_w = torch.hann_window(W, periodic=False, dtype=volume.dtype, device=volume.device)
		
		# 创建3D窗函数 (D, H, W)
		window_d = window_d.view(1, 1, D, 1, 1)
		window_h = window_h.view(1, 1, 1, H, 1)
		window_w = window_w.view(1, 1, 1, 1, W)
		
		# 应用窗函数
		windowed_volume = volume * window_d * window_h * window_w
		
		return windowed_volume
	
	@staticmethod
	def fft3d(volume, apply_window=True):
		"""
		执行3D FFT变换

		参数:
			volume: 输入体积 [B, C, D, H, W]
			apply_window: 是否应用窗函数

		返回:
			频谱 [B, C, D, H, W//2+1, 2]，最后一维是实部和虚部
		"""
		# 应用窗函数
		if apply_window:
			volume = FFTUtils.apply_hanning_window(volume)
		
		# 执行3D FFT
		spectrum = torch.fft.rfftn(volume, dim=(-3, -2, -1))
		
		return spectrum
	
	@staticmethod
	def ifft3d(spectrum, output_shape=None):
		"""
		执行3D逆FFT变换

		参数:
			spectrum: 输入频谱 [B, C, D, H, W//2+1]，复数张量
			output_shape: 输出形状，默认与输入相同

		返回:
			重构的体积 [B, C, D, H, W]
		"""
		# 执行3D逆FFT
		volume = torch.fft.irfftn(spectrum, s=output_shape, dim=(-3, -2, -1))
		
		return volume
	
	@staticmethod
	def spectral_whitening(spectrum, eps=1e-8):
		"""
		频谱白化（可选）

		参数:
			spectrum: 输入频谱 [B, C, D, H, W//2+1]，复数张量
			eps: 数值稳定性小常数

		返回:
			白化后的频谱
		"""
		# 计算幅度
		magnitude = torch.abs(spectrum)
		
		# 规范化幅度
		mean_magnitude = torch.mean(magnitude, dim=(-3, -2, -1), keepdim=True)
		whitened_magnitude = magnitude / (mean_magnitude + eps)
		
		# 保持相位不变
		phase = torch.angle(spectrum)
		whitened_spectrum = whitened_magnitude * torch.exp(1j * phase)
		
		return whitened_spectrum
	
	@staticmethod
	def frequency_band_enhancement(spectrum, band_range=(0.2, 0.6), enhancement_factor=1.5):
		"""
		频带增强（可选）

		参数:
			spectrum: 输入频谱 [B, C, D, H, W//2+1]，复数张量
			band_range: 增强的频率范围（归一化的半径）
			enhancement_factor: 增强因子

		返回:
			增强后的频谱
		"""
		B, C, D, H, W_half = spectrum.shape
		
		# 创建频率网格
		kz = torch.fft.fftfreq(D, device=spectrum.device)[:, None, None]
		ky = torch.fft.fftfreq(H, device=spectrum.device)[None, :, None]
		kx = torch.fft.rfftfreq(2 * (W_half - 1), device=spectrum.device)[None, None, :]
		
		# 计算频率半径
		k_radius = torch.sqrt(kz ** 2 + ky ** 2 + kx ** 2)
		
		# 创建频带掩码
		low, high = band_range
		mask = ((k_radius >= low) & (k_radius <= high)).float()
		mask = mask.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
		
		# 创建增强因子
		factor = torch.ones_like(mask) + (enhancement_factor - 1) * mask
		
		# 应用增强
		enhanced_spectrum = spectrum * factor
		
		return enhanced_spectrum