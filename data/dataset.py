import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.filters import frangi
from scipy.ndimage import gaussian_filter


class FrangiSampler:
	"""
	基于Frangi滤波器的血管增强采样器
	用于从CT体积中提取血管可能性较高的区域进行补丁采样
	"""
	
	def __init__(self, patch_size=64, scales=range(1, 5), contrast_enhancement=True):
		"""
		初始化Frangi采样器

		参数:
			patch_size: 提取补丁的大小
			scales: Frangi滤波器的尺度范围
			contrast_enhancement: 是否增强对比度
		"""
		self.patch_size = patch_size
		self.scales = scales
		self.contrast_enhancement = contrast_enhancement
	
	def create_vessel_density_map(self, ct_volume):
		"""
		创建血管密度图，用于引导采样

		参数:
			ct_volume: 输入CT体积

		返回:
			vessel_map: 血管密度图，值范围[0,1]
		"""
		# 标准化CT数据
		ct_norm = self._normalize_ct(ct_volume)
		
		# 应用Frangi滤波器(多尺度)
		vessel_map = np.zeros_like(ct_norm)
		for scale in self.scales:
			response = frangi(
				ct_norm,
				scale_range=(scale, scale),
				scale_step=1,
				black_ridges=False  # CT中血管通常为亮区域
			)
			vessel_map = np.maximum(vessel_map, response)
		
		# 移除低响应区域
		threshold = 0.05
		vessel_map[vessel_map < threshold] = 0
		
		# 对比度增强(可选)
		if self.contrast_enhancement:
			vessel_map = vessel_map ** 1.5
		
		# 重新归一化
		if vessel_map.max() > 0:
			vessel_map = vessel_map / vessel_map.max()
		
		return vessel_map
	
	def generate_sample_points(self, density_map, num_samples):
		"""
		基于密度图生成采样点

		参数:
			density_map: 血管密度图
			num_samples: 采样点数量

		返回:
			points: 采样点坐标列表 [(z, y, x), ...]
		"""
		half_size = self.patch_size // 2
		
		# 创建有效区域掩码(处理边界)
		valid_mask = np.zeros_like(density_map, dtype=bool)
		valid_mask[half_size:-half_size,
		half_size:-half_size,
		half_size:-half_size] = True
		
		# 将密度图限制在有效区域
		masked_density = density_map * valid_mask
		
		# 转换为概率分布
		flat_density = masked_density.ravel()
		
		if flat_density.sum() > 0:
			probability = flat_density / flat_density.sum()
			
			# 基于密度采样点
			indices = np.random.choice(
				len(flat_density),
				size=min(num_samples, np.count_nonzero(masked_density)),
				replace=False,
				p=probability
			)
			
			# 转换为3D坐标
			points = [np.unravel_index(idx, density_map.shape) for idx in indices]
			return points
		else:
			# 如果密度图全为0，进行均匀随机采样
			valid_indices = np.where(valid_mask)
			random_indices = np.random.choice(
				len(valid_indices[0]),
				size=min(num_samples, len(valid_indices[0])),
				replace=False
			)
			return [(valid_indices[0][i], valid_indices[1][i], valid_indices[2][i])
			        for i in random_indices]
	
	def extract_patches(self, volume, points):
		"""
		从体积中提取补丁

		参数:
			volume: 输入体积
			points: 采样点坐标列表

		返回:
			patches: 提取的补丁数组
		"""
		patches = []
		half_size = self.patch_size // 2
		
		for z, y, x in points:
			patch = volume[z - half_size:z + half_size,
			        y - half_size:y + half_size,
			        x - half_size:x + half_size]
			
			# 确保补丁大小正确
			if patch.shape == (self.patch_size, self.patch_size, self.patch_size):
				patches.append(patch)
		
		return np.array(patches) if patches else None
	
	def _normalize_ct(self, ct_volume):
		"""
		标准化CT数据

		参数:
			ct_volume: 输入CT体积

		返回:
			normalized_ct: 标准化后的CT体积
		"""
		# 鲁棒线性标准化到[0,1]
		# 使用百分位数避免离群值影响
		p_low = np.percentile(ct_volume, 0.5)
		p_high = np.percentile(ct_volume, 99.5)
		
		if p_high > p_low:
			normalized = (ct_volume - p_low) / (p_high - p_low)
			return np.clip(normalized, 0, 1)
		return np.zeros_like(ct_volume)


class VesselSegDataset(Dataset):
	"""
	血管分割数据集类
	使用Frangi采样器从CT体积中提取补丁
	"""
	
	def __init__(self, volumes, labels=None, patch_size=64, samples_per_volume=30, transform=None):
		"""
		初始化数据集

		参数:
			volumes: CT体积列表
			labels: 分割标签列表(可选)
			patch_size: 补丁大小
			samples_per_volume: 每个体积采样的补丁数量
			transform: 数据增强变换
		"""
		self.volumes = volumes
		self.labels = labels
		self.patch_size = patch_size
		self.samples_per_volume = samples_per_volume
		self.transform = transform
		self.sampler = FrangiSampler(patch_size=patch_size)
		
		# 预计算所有密度图
		self.density_maps = []
		for vol in volumes:
			self.density_maps.append(self.sampler.create_vessel_density_map(vol))
		
		# 生成所有采样点和补丁
		self.all_volume_patches = []
		self.all_label_patches = []
		
		for i, vol in enumerate(volumes):
			points = self.sampler.generate_sample_points(
				self.density_maps[i], self.samples_per_volume)
			
			vol_patches = self.sampler.extract_patches(vol, points)
			
			if vol_patches is not None:
				self.all_volume_patches.append(vol_patches)
				
				if self.labels is not None:
					label_patches = self.sampler.extract_patches(self.labels[i], points)
					if label_patches is not None:
						self.all_label_patches.append(label_patches)
		
		# 将补丁列表展平
		if self.all_volume_patches:
			self.all_volume_patches = np.vstack(self.all_volume_patches)
			if self.labels is not None and self.all_label_patches:
				self.all_label_patches = np.vstack(self.all_label_patches)
		else:
			raise ValueError("No valid patches extracted from volumes")
	
	def __len__(self):
		return len(self.all_volume_patches)
	
	def __getitem__(self, idx):
		vol_patch = self.all_volume_patches[idx]
		
		# 添加通道维度
		vol_patch = vol_patch[np.newaxis, ...]
		
		if self.labels is not None:
			label_patch = self.all_label_patches[idx]
			label_patch = label_patch[np.newaxis, ...]
			
			if self.transform:
				vol_patch, label_patch = self.transform(vol_patch, label_patch)
			
			return torch.from_numpy(vol_patch).float(), torch.from_numpy(label_patch).float()
		else:
			if self.transform:
				vol_patch = self.transform(vol_patch)
			
			return torch.from_numpy(vol_patch).float()