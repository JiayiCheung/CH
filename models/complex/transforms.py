
import torch
from .functional import grid_sample


def cylindrical_mapping(volume, grid, mode='bilinear', align_corners=True):
	"""
	复数兼容的柱坐标映射

	参数:
		volume: 输入体积 [B, C, D, H, W]
		grid: 采样网格
		mode: 插值模式
		align_corners: 对齐角点

	返回:
		映射后的体积
	"""
	volume_reshaped = volume.permute(0, 1, 4, 3, 2)  # [B, C, W, H, D]
	
	# 使用复数兼容的grid_sample
	sampled = grid_sample(
		volume_reshaped, grid,
		mode=mode, align_corners=align_corners
	)
	
	return sampled


def cylindrical_to_cartesian(cylindrical_volume, grid, output_shape,
                             mode='bilinear', align_corners=True):
	"""
	复数兼容的柱坐标到笛卡尔坐标映射

	参数:
		cylindrical_volume: 柱坐标体积
		grid: 采样网格
		output_shape: 输出形状
		mode: 插值模式
		align_corners: 对齐角点

	返回:
		笛卡尔坐标体积
	"""
	cylindrical_reshaped = cylindrical_volume.permute(0, 1, 4, 3, 2)
	
	# 使用复数兼容的grid_sample
	sampled = grid_sample(
		cylindrical_reshaped, grid,
		mode=mode, align_corners=align_corners
	)
	
	return sampled