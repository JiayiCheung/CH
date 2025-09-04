import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from functools import lru_cache



class CHDataset(Dataset):
	"""
	血管分割数据集类 - 惰性加载版
	需要预计算的Frangi响应，不支持实时计算
	"""
	
	def __init__(self, volume_paths, label_paths, patch_size=64,
	             samples_per_volume=30, transform=None,
	             preprocessed_dir=None):
		"""
		初始化数据集

		参数:
			volume_paths: CT体积文件路径列表
			label_paths: 分割标签文件路径列表
			patch_size: 补丁大小
			samples_per_volume: 每个体积采样的补丁数量
			transform: 数据增强变换
			preprocessed_dir: 包含预计算Frangi响应的目录(必需)
		"""
		self.volume_paths = volume_paths
		self.label_paths = label_paths
		self.patch_size = patch_size
		self.samples_per_volume = samples_per_volume
		self.transform = transform
		
		# 检查预处理目录是否存在
		if preprocessed_dir is None or not os.path.exists(preprocessed_dir):
			raise ValueError("必须提供有效的预处理目录，包含预计算的Frangi响应")
		self.preprocessed_dir = preprocessed_dir
		
		# 从原始模块导入FrangiSampler用于采样（但不用于计算）
		from data.dataset import FrangiSampler
		self.sampler = FrangiSampler(patch_size=patch_size)
		
		# 预计算每个体积的样本偏移
		self._compute_sample_offsets()
	
	def _compute_sample_offsets(self):
		"""计算每个体积的样本偏移量，用于定位特定索引"""
		self.offsets = [0]
		total = 0
		for _ in range(len(self.volume_paths)):
			total += self.samples_per_volume
			self.offsets.append(total)
	
	def __len__(self):
		"""数据集的总样本数"""
		return self.offsets[-1]
	
	def _get_case_id(self, idx):
		"""根据路径获取案例ID"""
		path = self.volume_paths[idx]
		return os.path.splitext(os.path.splitext(os.path.basename(path))[0])[0]
	
	def _load_volume(self, idx):
		"""惰性加载单个体积"""
		import nibabel as nib
		volume = nib.load(self.volume_paths[idx]).get_fdata().astype(np.float32)
		return volume
	
	def _load_label(self, idx):
		"""惰性加载单个标签"""
		import nibabel as nib
		label = nib.load(self.label_paths[idx]).get_fdata().astype(np.float32)
		return label
	
	def _get_density_map(self, idx):
		"""获取预计算的Frangi响应"""
		case_id = self._get_case_id(idx)
		frangi_path = os.path.join(self.preprocessed_dir, f"{case_id}_frangi.npy")
		
		if not os.path.exists(frangi_path):
			raise FileNotFoundError(f"未找到预计算的Frangi响应：{frangi_path}")
		
		return np.load(frangi_path)
	
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
		
		# 获取密度图
		density_map = self._get_density_map(vol_idx)
		
		# 使用原始方法生成采样点
		points = self.sampler.generate_sample_points(density_map, self.samples_per_volume)
		
		if sample_idx < len(points):
			point = points[sample_idx]
			
			# 加载体积
			volume = self._load_volume(vol_idx)
			
			# 提取补丁
			half_size = self.patch_size // 2
			z, y, x = point
			vol_patch = volume[z - half_size:z + half_size,
			            y - half_size:y + half_size,
			            x - half_size:x + half_size]
			
			# 加载标签
			label = self._load_label(vol_idx)
			label_patch = label[z - half_size:z + half_size,
			              y - half_size:y + half_size,
			              x - half_size:x + half_size]
			
			# 添加通道维度
			vol_patch = vol_patch[np.newaxis, ...]
			label_patch = label_patch[np.newaxis, ...]
			
			# 应用数据增强
			if self.transform:
				vol_patch, label_patch = self.transform(vol_patch, label_patch)
			
			return torch.from_numpy(vol_patch).float(), torch.from_numpy(label_patch).float()
		else:
			# 应急情况：返回随机补丁
			return self.__getitem__(np.random.randint(len(self)))