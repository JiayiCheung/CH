
import torch


def spectral_whitening(spectrum, eps=1e-8):
	"""
	复数兼容的频谱白化

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


def frequency_band_enhancement(spectrum, band_range=(0.2, 0.6), enhancement_factor=1.5):
	"""
	复数兼容的频带增强

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
	
	# 应用增强，安全处理复数
	enhanced_spectrum = spectrum * factor
	
	return enhanced_spectrum