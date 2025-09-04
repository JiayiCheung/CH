import os
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
	惰性加载实现，支持使用预计算的Frangi响应
	"""
	
	def __init__(self, volumes, labels=None, patch_size=64, samples_per_volume=30, transform=None,
	             preprocessed_dir=None):
		"""
		初始化数据集

		参数:
			volumes: CT体积列表或体积文件路径列表
			labels: 分割标签列表或标签文件路径列表(可选)
			patch_size: 补丁大小
			samples_per_volume: 每个体积采样的补丁数量
			transform: 数据增强变换
			preprocessed_dir: 包含预计算Frangi响应的目录(可选)
		"""
		self.patch_size = patch_size
		self.samples_per_volume = samples_per_volume
		self.transform = transform
		self.preprocessed_dir = preprocessed_dir
		self.sampler = FrangiSampler(patch_size=patch_size)
		
		# 检测输入类型（路径列表或已加载体积）
		self.is_path_input = False
		if volumes is not None and isinstance(volumes, list):
			if len(volumes) > 0 and isinstance(volumes[0], str):
				self.is_path_input = True
		
		# 存储体积/标签或路径
		self.volumes = volumes
		self.labels = labels
		
		# 预计算每个体积的样本偏移
		self._compute_sample_offsets()
		
		# 缓存最近使用的数据
		self.cache = {}
		self.max_cache_size = 3  # 只缓存最近3个体积的数据
	
	def _compute_sample_offsets(self):
		"""计算每个体积的样本偏移量，用于定位特定索引"""
		self.offsets = [0]
		total = 0
		for _ in range(len(self.volumes)):
			total += self.samples_per_volume
			self.offsets.append(total)
	
	def __len__(self):
		"""数据集的总样本数"""
		return self.offsets[-1]
	
	def _get_case_id(self, idx):
		"""根据体积或路径获取案例ID"""
		if self.is_path_input:
			path = self.volumes[idx]
			return os.path.splitext(os.path.splitext(os.path.basename(path))[0])[0]
		else:
			# 如果是实际体积，使用索引作为ID
			return f"volume_{idx}"
	
	def _load_volume(self, idx):
		"""惰性加载单个体积"""
		if self.is_path_input:
			# 加载文件
			import nibabel as nib
			volume = nib.load(self.volumes[idx]).get_fdata().astype(np.float32)
			return volume
		else:
			# 直接返回已加载的体积
			return self.volumes[idx]
	
	def _load_label(self, idx):
		"""惰性加载单个标签"""
		if self.labels is None:
			return None
		
		if self.is_path_input:
			# 加载文件
			import nibabel as nib
			label = nib.load(self.labels[idx]).get_fdata().astype(np.float32)
			return label
		else:
			# 直接返回已加载的标签
			return self.labels[idx]
	
	def _get_density_map(self, idx):
		"""获取密度图，优先使用预计算的Frangi响应"""
		case_id = self._get_case_id(idx)
		
		# 检查是否有预计算的Frangi响应
		if self.preprocessed_dir is not None and self.is_path_input:
			frangi_path = os.path.join(self.preprocessed_dir, f"{case_id}_frangi.npy")
			if os.path.exists(frangi_path):
				return np.load(frangi_path)
		
		# 如果没有预计算，则实时计算
		if idx in self.cache and 'volume' in self.cache[idx]:
			volume = self.cache[idx]['volume']
		else:
			volume = self._load_volume(idx)
			if idx not in self.cache:
				self.cache[idx] = {}
			self.cache[idx]['volume'] = volume
		
		# 使用FrangiSampler计算密度图
		return self.sampler.create_vessel_density_map(volume)
	
	def __getitem__(self, idx):
		"""按需加载并返回单个补丁"""
		# 确定体积索引
		vol_idx = 0
		while vol_idx + 1 < len(self.offsets) and self.offsets[vol_idx + 1] <= idx:
			vol_idx += 1
		
		# 确定这个体积内的样本索引
		sample_idx = idx - self.offsets[vol_idx]
		
		# 为了一致性，对每个索引使用固定的随机种子
		np.random.seed(idx)
		
		# 获取或计算密度图
		if vol_idx in self.cache and 'density_map' in self.cache[vol_idx]:
			density_map = self.cache[vol_idx]['density_map']
		else:
			density_map = self._get_density_map(vol_idx)
			
			# 更新缓存
			if vol_idx not in self.cache:
				self.cache[vol_idx] = {}
			self.cache[vol_idx]['density_map'] = density_map
			
			# 限制缓存大小
			if len(self.cache) > self.max_cache_size:
				oldest_key = next(iter(self.cache))
				del self.cache[oldest_key]
		
		# 使用原始方法生成采样点
		points = self.sampler.generate_sample_points(density_map, self.samples_per_volume)
		
		if sample_idx < len(points):
			point = points[sample_idx]
			
			# 按需加载体积
			if vol_idx in self.cache and 'volume' in self.cache[vol_idx]:
				volume = self.cache[vol_idx]['volume']
			else:
				volume = self._load_volume(vol_idx)
				if vol_idx not in self.cache:
					self.cache[vol_idx] = {}
				self.cache[vol_idx]['volume'] = volume
			
			# 提取补丁
			half_size = self.patch_size // 2
			z, y, x = point
			vol_patch = volume[z - half_size:z + half_size,
			            y - half_size:y + half_size,
			            x - half_size:x + half_size]
			
			# 按需加载标签
			if self.labels is not None:
				if vol_idx in self.cache and 'label' in self.cache[vol_idx]:
					label = self.cache[vol_idx]['label']
				else:
					label = self._load_label(vol_idx)
					if label is not None and vol_idx in self.cache:
						self.cache[vol_idx]['label'] = label
				
				label_patch = label[z - half_size:z + half_size,
				              y - half_size:y + half_size,
				              x - half_size:x + half_size]
			else:
				label_patch = None
			
			# 添加通道维度
			vol_patch = vol_patch[np.newaxis, ...]
			if label_patch is not None:
				label_patch = label_patch[np.newaxis, ...]
			
			# 应用数据增强
			if self.transform:
				if label_patch is not None:
					vol_patch, label_patch = self.transform(vol_patch, label_patch)
				else:
					vol_patch = self.transform(vol_patch)
			
			if label_patch is not None:
				return torch.from_numpy(vol_patch).float(), torch.from_numpy(label_patch).float()
			else:
				return torch.from_numpy(vol_patch).float()
