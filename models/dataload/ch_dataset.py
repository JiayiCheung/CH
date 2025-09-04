import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from functools import lru_cache


class CHDataset(Dataset):
	"""
	血管分割数据集类 - nnU-Net风格实现
	使用预计算的采样点和全局索引映射
	"""
	
	def __init__(self, volume_paths, label_paths, patch_size=64,
	             transform=None, preprocessed_dir=None):
		"""
		初始化数据集

		参数:
			volume_paths: CT体积文件路径列表
			label_paths: 分割标签文件路径列表
			patch_size: 补丁大小
			transform: 数据增强变换
			preprocessed_dir: 包含预计算Frangi响应和采样点的目录(必需)
		"""
		self.volume_paths = volume_paths
		self.label_paths = label_paths
		self.patch_size = patch_size
		self.transform = transform
		
		# 检查预处理目录是否存在
		if preprocessed_dir is None or not os.path.exists(preprocessed_dir):
			raise ValueError("必须提供有效的预处理目录，包含预计算的Frangi响应")
		self.preprocessed_dir = preprocessed_dir
		
		# 加载全局索引映射
		index_mapping_file = os.path.join(preprocessed_dir, "index_mapping.npy")
		if not os.path.exists(index_mapping_file):
			raise FileNotFoundError(f"未找到索引映射文件：{index_mapping_file}，请先运行预处理")
		
		self.index_mapping = np.load(index_mapping_file, allow_pickle=True)
		
		# 为每个体积加载有效点
		self.valid_points = []
		for vol_idx, vol_path in enumerate(volume_paths):
			case_id = self._get_case_id(vol_idx)
			points_file = os.path.join(preprocessed_dir, f"{case_id}_valid_points.npy")
			
			if not os.path.exists(points_file):
				raise FileNotFoundError(f"未找到有效点文件：{points_file}，请先运行预处理")
			
			points = np.load(points_file, allow_pickle=True)
			self.valid_points.append(points)
	
	def _get_case_id(self, idx):
		"""根据路径获取案例ID"""
		path = self.volume_paths[idx]
		return os.path.splitext(os.path.splitext(os.path.basename(path))[0])[0]
	
	def __len__(self):
		"""数据集的总样本数"""
		return len(self.index_mapping)
	
	def _load_patch(self, file_path, center_point):
		"""仅加载必要的补丁区域"""
		import nibabel as nib
		
		z, y, x = center_point
		half_size = self.patch_size // 2
		
		# 加载文件头但不加载全部数据
		img = nib.load(file_path, mmap=True)
		
		# 计算边界
		z_start = max(0, z - half_size)
		z_end = min(img.shape[0], z + half_size)
		y_start = max(0, y - half_size)
		y_end = min(img.shape[1], y + half_size)
		x_start = max(0, x - half_size)
		x_end = min(img.shape[2], x + half_size)
		
		# 只加载所需区域
		patch = np.array(img.dataobj[z_start:z_end, y_start:y_end, x_start:x_end], dtype=np.float32)
		
		# 处理边界填充
		if patch.shape != (self.patch_size, self.patch_size, self.patch_size):
			# 创建零补丁
			full_patch = np.zeros((self.patch_size, self.patch_size, self.patch_size), dtype=np.float32)
			
			# 计算偏移
			z_offset = max(0, half_size - z)
			y_offset = max(0, half_size - y)
			x_offset = max(0, half_size - x)
			
			# 填充补丁
			p_z_end = z_offset + (z_end - z_start)
			p_y_end = y_offset + (y_end - y_start)
			p_x_end = x_offset + (x_end - x_start)
			
			full_patch[z_offset:p_z_end, y_offset:p_y_end, x_offset:p_x_end] = patch
			patch = full_patch
		
		return patch
	
	def __getitem__(self, idx):
		"""获取一个样本，使用预计算的索引和点"""
		# 检查索引是否有效
		if idx >= len(self.index_mapping):
			raise IndexError(f"索引 {idx} 超出范围 (0-{len(self.index_mapping) - 1})")
		
		# 使用预计算的映射获取体积和点索引
		vol_idx, point_idx = self.index_mapping[idx]
		
		# 获取预验证的有效点
		if vol_idx >= len(self.valid_points) or point_idx >= len(self.valid_points[vol_idx]):
			raise IndexError(f"无效的体积/点索引: vol_idx={vol_idx}, point_idx={point_idx}")
		
		point = self.valid_points[vol_idx][point_idx]
		
		# 加载补丁
		vol_patch = self._load_patch(self.volume_paths[vol_idx], point)
		label_patch = self._load_patch(self.label_paths[vol_idx], point)
		
		# 添加通道维度
		vol_patch = vol_patch[np.newaxis, ...]
		label_patch = label_patch[np.newaxis, ...]
		
		# 应用数据增强
		if self.transform:
			vol_patch, label_patch = self.transform(vol_patch, label_patch)
		
		return torch.from_numpy(vol_patch).float(), torch.from_numpy(label_patch).float()