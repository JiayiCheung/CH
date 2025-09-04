
import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceNorm3d(nn.Module):
	"""复数兼容的InstanceNorm3d"""
	
	def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False,
	             track_running_stats=False):
		super().__init__()
		self.norm_real = nn.InstanceNorm3d(
			num_features, eps, momentum, affine, track_running_stats)
		self.norm_imag = nn.InstanceNorm3d(
			num_features, eps, momentum, affine, track_running_stats)
	
	def forward(self, x):
		if torch.is_complex(x):
			real_part = self.norm_real(x.real)
			imag_part = self.norm_imag(x.imag)
			return torch.complex(real_part, imag_part)
		else:
			return self.norm_real(x)


class Conv3d(nn.Module):
	"""复数兼容的Conv3d"""
	
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
	             dilation=1, groups=1, bias=True, padding_mode='zeros'):
		super().__init__()
		self.conv_real = nn.Conv3d(in_channels, out_channels, kernel_size, stride,
		                           padding, dilation, groups, bias, padding_mode)
		self.conv_imag = nn.Conv3d(in_channels, out_channels, kernel_size, stride,
		                           padding, dilation, groups, bias, padding_mode)
	
	def forward(self, x):
		if torch.is_complex(x):
			# 完整复数卷积：(a+bi)(c+di) = (ac-bd) + (ad+bc)i
			real_real = self.conv_real(x.real)
			imag_imag = self.conv_imag(x.imag)
			real_imag = self.conv_real(x.imag)
			imag_real = self.conv_imag(x.real)
			
			real_part = real_real - imag_imag
			imag_part = real_imag + imag_real
			
			return torch.complex(real_part, imag_part)
		else:
			return self.conv_real(x)