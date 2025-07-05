import torch
import torch.nn.functional as F
import math
from utils.complex_ops import complex_grid_sample_3d


class CylindricalMapping:
	"""笛卡尔坐标到柱坐标的可微分映射"""
	
	def __init__(self, r_samples=32, theta_samples=36, z_samples=32):
		"""
		初始化柱坐标映射器

		参数:
			r_samples: 径向采样点数
			theta_samples: 角度采样点数
			z_samples: 轴向采样点数
		"""
		self.r_samples = r_samples
		self.theta_samples = theta_samples
		self.z_samples = z_samples
		
		# 缓存网格
		self.grid_cache = {}
	
	def _create_sampling_grid(self, batch_size, depth, height, width, device):
		"""
		创建用于网格采样的柱坐标网格

		参数:
			batch_size: 批次大小
			depth, height, width: 输入体积的形状
			device: 计算设备

		返回:
			用于grid_sample的采样网格
		"""
		# 检查缓存
		cache_key = (batch_size, depth, height, width, device)
		if cache_key in self.grid_cache:
			return self.grid_cache[cache_key]
		
		# 创建柱坐标网格
		r = torch.linspace(0, 1, self.r_samples, device=device)
		theta = torch.linspace(0, 2 * math.pi, self.theta_samples, device=device)
		z = torch.linspace(-1, 1, self.z_samples, device=device)
		
		# 网格化
		r_grid, theta_grid, z_grid = torch.meshgrid(r, theta, z, indexing='ij')
		
		# 转换为笛卡尔坐标 (归一化到[-1, 1])
		x_grid = r_grid * torch.cos(theta_grid)
		y_grid = r_grid * torch.sin(theta_grid)
		
		# 创建网格采样点 [r_samples, theta_samples, z_samples, 3]
		grid = torch.stack([z_grid, y_grid, x_grid], dim=-1)
		
		# 重塑为grid_sample所需的形状 [batch_size, r_samples, theta_samples, z_samples, 3]
		grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
		
		# 缓存网格
		self.grid_cache[cache_key] = grid
		
		return grid
	
	def cartesian_to_cylindrical(self, volume):
		"""
		将笛卡尔坐标体积转换为柱坐标表示

		参数:
			volume: 输入体积 [B, C, D, H, W]

		返回:
			柱坐标表示 [B, C, r_samples, theta_samples, z_samples]
		"""
		B, C, D, H, W = volume.shape
		
		# 创建采样网格
		grid = self._create_sampling_grid(B, D, H, W, volume.device)
		
		# 重塑为grid_sample所需的格式
		volume_reshaped = volume.permute(0, 1, 4, 3, 2)  # [B, C, W, H, D]
		
		if torch.is_complex(volume):
			sampled = complex_grid_sample_3d(volume_reshaped, grid)
		else:
			sampled = F.grid_sample(volume_reshaped, grid, ...)
		
		
		# 处理 r=0 处的奇异性
		r_zero_mask = (grid[..., 1] ** 2 + grid[..., 2] ** 2 < 1e-6)
		for b in range(B):
			for c in range(C):
				for z in range(self.z_samples):
					# 如果r=0处有点，使用同一z平面上所有角度的平均值
					if r_zero_mask[b, 0, :, z].any():
						avg_value = cylindrical_volume[b, c, 0, :, z].mean()
						cylindrical_volume[b, c, 0, :, z] = avg_value
		
		return cylindrical_volume
	
	def cylindrical_to_cartesian(self, cylindrical_volume, output_shape):
		"""
		将柱坐标表示转换回笛卡尔坐标体积

		参数:
			cylindrical_volume: 柱坐标表示 [B, C, r_samples, theta_samples, z_samples]
			output_shape: 输出形状 (D, H, W)

		返回:
			笛卡尔坐标体积 [B, C, D, H, W]
		"""
		B, C = cylindrical_volume.shape[:2]
		D, H, W = output_shape
		
		# 创建反向映射网格
		# 这部分更复杂，需要反向计算柱坐标
		# 为简化实现，我们使用一个近似方法
		
		# 创建笛卡尔坐标网格
		z = torch.linspace(-1, 1, D, device=cylindrical_volume.device)
		y = torch.linspace(-1, 1, H, device=cylindrical_volume.device)
		x = torch.linspace(-1, 1, W, device=cylindrical_volume.device)
		
		z_grid, y_grid, x_grid = torch.meshgrid(z, y, x, indexing='ij')
		
		# 转换为柱坐标
		r_grid = torch.sqrt(x_grid ** 2 + y_grid ** 2)
		theta_grid = torch.atan2(y_grid, x_grid) % (2 * math.pi)
		
		# 归一化到采样索引范围
		r_norm = r_grid  # 已经在 [0, 1] 范围内
		theta_norm = theta_grid / (2 * math.pi)  # [0, 1]
		z_norm = (z_grid + 1) / 2  # [-1, 1] -> [0, 1]
		
		# 创建采样网格
		grid = torch.stack([
			2 * z_norm - 1,  # [0, 1] -> [-1, 1]
			2 * theta_norm - 1,  # [0, 1] -> [-1, 1]
			2 * r_norm - 1  # [0, 1] -> [-1, 1]
		], dim=-1)
		
		# 调整形状并重复以匹配批次大小
		grid = grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)
		
		# 重塑柱坐标体积以适应grid_sample
		cylindrical_reshaped = cylindrical_volume.permute(0, 1, 4, 3, 2)  # [B, C, z, theta, r]
		
		# 对每个通道执行网格采样
		cartesian_volumes = []
		for c in range(C):
			# 网格采样
			sampled = F.grid_sample(
				cylindrical_reshaped[:, c:c + 1], grid,
				mode='bilinear', align_corners=True
			)
			cartesian_volumes.append(sampled)
		
		# 合并通道
		cartesian_volume = torch.cat(cartesian_volumes, dim=1)
		
		return cartesian_volume