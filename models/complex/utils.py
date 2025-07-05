
import torch
import functools
from torch import view_as_real, view_as_complex

def complex_to_real(x):
	"""
	将复数张量转换为实数表示

	参数:
		x: 复数张量 [..., C, ...]

	返回:
		实数张量 [..., 2*C, ...] (实部和虚部交错)
	"""
	if not torch.is_complex(x):
		return x
	
	real_part = x.real
	imag_part = x.imag
	
	# 创建形状信息
	shape = list(x.shape)
	channel_dim = 1  # 假设通道维度是1
	
	# 扩展通道维度
	shape[channel_dim] *= 2
	
	# 创建结果张量
	result = torch.zeros(shape, dtype=real_part.dtype, device=x.device)
	
	# 填充实部和虚部
	result[:, 0::2] = real_part
	result[:, 1::2] = imag_part
	
	return result


def real_to_complex(x):
	"""
	将实数表示转换回复数张量

	参数:
		x: 实数张量 [..., 2*C, ...] (实部和虚部交错)

	返回:
		复数张量 [..., C, ...]
	"""
	if torch.is_complex(x):
		return x
	
	# 创建形状信息
	shape = list(x.shape)
	channel_dim = 1  # 假设通道维度是1
	
	if shape[channel_dim] % 2 != 0:
		raise ValueError("通道数必须是偶数才能转换为复数")
	
	# 缩小通道维度
	shape[channel_dim] //= 2
	
	# 提取实部和虚部
	real_part = x[:, 0::2]
	imag_part = x[:, 1::2]
	
	# 创建复数张量
	return torch.complex(real_part, imag_part)


def apply_to_complex(func):
	"""
	装饰器: 使任何函数兼容复数张量

	参数:
		func: 要装饰的函数

	返回:
		兼容复数的函数
	"""
	
	@functools.wraps(func)
	def wrapper(input, *args, **kwargs):
		if torch.is_complex(input):
			real_result = func(input.real, *args, **kwargs)
			imag_result = func(input.imag, *args, **kwargs)
			return torch.complex(real_result, imag_result)
		else:
			return func(input, *args, **kwargs)
	
	return wrapper