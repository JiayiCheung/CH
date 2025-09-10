
import torch
import torch.nn.functional as F


def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None):
	"""
	复数兼容的grid_sample实现

	参数:
		与torch.nn.functional.grid_sample相同
	"""
	if torch.is_complex(input):
		real_part = F.grid_sample(input.real, grid, mode, padding_mode, align_corners)
		imag_part = F.grid_sample(input.imag, grid, mode, padding_mode, align_corners)
		return torch.complex(real_part, imag_part)
	else:
		return F.grid_sample(input, grid, mode, padding_mode, align_corners)


def interpolate(input, size=None, scale_factor=None, mode='nearest',
                align_corners=None, recompute_scale_factor=None):
	"""
	复数兼容的interpolate实现

	参数:
		与torch.nn.functional.interpolate相同
	"""
	if torch.is_complex(input):
		real_part = F.interpolate(input.real, size, scale_factor, mode,
		                          align_corners, recompute_scale_factor)
		imag_part = F.interpolate(input.imag, size, scale_factor, mode,
		                          align_corners, recompute_scale_factor)
		return torch.complex(real_part, imag_part)
	else:
		return F.interpolate(input, size, scale_factor, mode,
		                     align_corners, recompute_scale_factor)