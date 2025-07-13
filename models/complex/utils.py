#complex.utils.py
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




def magnitude(x: torch.Tensor) -> torch.Tensor:
    """
    返回张量的幅值图：

    - 若 `x` 是复数 (cfloat/cdouble)，返回 |x| = sqrt(re² + im²)
    - 若 `x` 本身就是实数，直接按原样返回（无额外开销）

    参数
    ----
    x : torch.Tensor
        任意实数或复数张量

    返回
    ----
    torch.Tensor
        实数张量，与 `x` 的 shape 相同，dtype 与 x.real 保持一致
    """
    return x.abs() if torch.is_complex(x) else x


def apply_real_weight(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    将 **实数权重张量 `w`** 施加到输入特征 `x` 上：

    - 若 `x` 为复数：同时对实部、虚部各自乘以 `w`
    - 若 `x` 为实数：直接 `x * w`

    两个张量的 shape 需可广播；通常 `w` 已经是
    (B,C,1,1,1) 或 (B,1,D,H,W) 之类的注意力权重。

    参数
    ----
    x : torch.Tensor
        输入特征，可以是实数或复数
    w : torch.Tensor
        实数权重，dtype 必须是浮点型

    返回
    ----
    torch.Tensor
        与 `x` dtype/shape 对齐的张量
    """
    if torch.is_complex(x):
        # 注意：w 必须能广播到 x.real 的 shape
        return torch.complex(x.real * w, x.imag * w)
    else:
        return x * w
