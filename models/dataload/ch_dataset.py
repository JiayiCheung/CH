#/models/dataload/ch_dataset.py


import os
import pickle
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class VesselDataset(Dataset):
	def __init__(self,
	             data_dir,
	             props_dir,
	             split_file,
	             split='train',
	             patch_size=(64, 128, 128),
	             oversample_foreground_percent=0.33,
	             mode='train'):
		"""
		血管分割数据集

		参数:
			data_dir: 处理后数据目录，如 preprocess/processed/data
			props_dir: 属性文件目录，如 preprocess/processed/properties
			split_file: 数据集划分文件，如 preprocess/splits.json
			split: 'train' 或 'val'
			patch_size: 补丁大小，如 [64, 128, 128]
			oversample_foreground_percent: 前景过采样比例
			mode: 'train', 'val' 或 'test'
		"""
		self.data_dir = data_dir
		self.props_dir = props_dir
		self.split = split
		self.patch_size = patch_size
		self.oversample_foreground_percent = oversample_foreground_percent
		self.mode = mode
		
		# 加载数据集划分
		with open(split_file, 'r') as f:
			splits = json.load(f)
		
		# 获取当前划分的样本ID
		self.case_ids = splits[split]
		
		# 预加载属性文件（较小，可全部加载到内存）
		self.properties = {}
		for case_id in self.case_ids:
			prop_file = os.path.join(props_dir, f"{case_id}.pkl")
			if os.path.exists(prop_file):
				with open(prop_file, 'rb') as f:
					self.properties[case_id] = pickle.load(f)
		
		print(f"加载了 {len(self.case_ids)} 个 {split} 样本")
	
	def __len__(self):
		return len(self.case_ids)
	
	def _get_do_oversample(self):
		"""决定是否进行前景过采样"""
		return np.random.random() < self.oversample_foreground_percent
	
	def _get_random_center_point(self, shape):
		"""获取随机中心点"""
		center_point = []
		for i, (img_size, patch_size) in enumerate(zip(shape, self.patch_size)):
			# 确保补丁不会超出图像边界
			if img_size <= patch_size:
				# 如果图像尺寸小于补丁尺寸，中心点就是图像中心
				center_point.append(img_size // 2)
			else:
				# 在有效范围内随机选择
				low = patch_size // 2
				high = img_size - patch_size // 2
				center_point.append(np.random.randint(low, high))
		return center_point
	
	def _get_center_point(self, shape):
		"""获取图像中心点"""
		return [sz // 2 for sz in shape]
	
	def _extract_patch(self, data, center_point):
		"""
		从数据中提取补丁

		参数:
			data: 形状为 [C, D, H, W] 的数据
			center_point: 补丁中心点 [z, y, x]

		返回:
			补丁数据
		"""
		# 初始化输出补丁
		patch = np.zeros((data.shape[0], *self.patch_size), dtype=data.dtype)
		
		# 计算源和目标的区域
		src_slices = []
		dst_slices = []
		
		for i, center in enumerate(center_point):
			# 在原始数据中的范围
			src_start = max(0, center - self.patch_size[i] // 2)
			src_end = min(data.shape[i + 1], center + self.patch_size[i] // 2 + self.patch_size[i] % 2)
			src_slices.append(slice(src_start, src_end))
			
			# 在目标补丁中的范围
			dst_start = max(0, self.patch_size[i] // 2 - (center - src_start))
			dst_end = dst_start + (src_end - src_start)
			dst_slices.append(slice(dst_start, dst_end))
		
		# 复制数据
		patch[:, dst_slices[0], dst_slices[1], dst_slices[2]] = data[:, src_slices[0], src_slices[1], src_slices[2]]
		
		return patch
	
	def __getitem__(self, idx):
		"""获取数据样本"""
		case_id = self.case_ids[idx]
		
		# 加载数据
		data_file = os.path.join(self.data_dir, f"{case_id}.npy")
		data = np.load(data_file)  # [2, D, H, W] - 索引0是图像，索引1是标签
		
		# 根据模式决定采样策略
		if self.mode == 'train':
			# 决定是否进行前景过采样
			do_oversample = self._get_do_oversample()
			
			if do_oversample and case_id in self.properties and 'class_locations' in self.properties[case_id] and \
					self.properties[case_id]['class_locations'] is not None:
				class_locations = self.properties[case_id]['class_locations']
				
				# 前景采样
				if 'foreground' in class_locations and len(class_locations['foreground']) > 0:
					point_idx = np.random.randint(0, len(class_locations['foreground']))
					center_point = class_locations['foreground'][point_idx]
				# 如果没有前景点，退化为随机采样
				else:
					center_point = self._get_random_center_point(data.shape[1:])
			else:
				# 随机采样
				center_point = self._get_random_center_point(data.shape[1:])
		
		else:  # 验证或测试模式
			# 中心裁剪
			center_point = self._get_center_point(data.shape[1:])
		
		# 确保中心点是整数
		center_point = [int(p) for p in center_point]
		
		# 提取补丁
		patch = self._extract_patch(data, center_point)
		
		# 转换为Tensor
		image = torch.from_numpy(patch[0:1]).float()  # [1, D, H, W]
		label = torch.from_numpy(patch[1:2]).long()  # [1, D, H, W]
		
		return image, label