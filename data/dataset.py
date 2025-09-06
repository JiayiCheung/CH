# data/dataset.py

import os
import numpy as np
import pickle
import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional, Union
import random

class HepaticVesselDataset(Dataset):
	"""肝脏血管分割数据集，使用nnUNet的采样策略"""
	
	def __init__(self, data_dir: str, properties_dir: str,
	             patch_size: Tuple[int, int, int] = (64, 64, 64),
	             mode: str = 'train',
	             splits_file: Optional[str] = None,
	             oversample_foreground_percent: float = 0.33,
	             num_samples_per_volume: int = 30,
	             transform=None):
		"""
		初始化数据集

		参数:
			data_dir: 预处理数据目录
			properties_dir: 属性文件目录
			patch_size: 补丁大小
			mode: 'train'或'val'
			splits_file: 训练验证集划分文件
			oversample_foreground_percent: 前景过采样比例
			num_samples_per_volume: 每个体积采样的补丁数量
			transform: 数据增强转换
		"""
		self.data_dir = data_dir
		self.properties_dir = properties_dir
		self.patch_size = patch_size
		self.mode = mode
		self.oversample_foreground_percent = oversample_foreground_percent
		self.num_samples_per_volume = num_samples_per_volume
		self.transform = transform
		
		# 加载划分
		if splits_file is not None and os.path.exists(splits_file):
			with open(splits_file, 'r') as f:
				splits = json.load(f)
			
			if mode == 'train':
				self.case_ids = splits['train']
			elif mode == 'val':
				self.case_ids = splits['val']
			else:
				raise ValueError(f"无效的模式: {mode}, 必须是'train'或'val'")
		else:
			# 如果没有划分文件，使用目录中的所有文件
			self.case_ids = [os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith('.npz')]
		
		# 加载属性
		self.properties = {}
		self.class_locations = {}
		
		for case_id in self.case_ids:
			prop_file = os.path.join(properties_dir, f"{case_id}.pkl")
			if os.path.exists(prop_file):
				with open(prop_file, 'rb') as f:
					prop = pickle.load(f)
					self.properties[case_id] = prop
					if 'class_locations' in prop:
						self.class_locations[case_id] = prop['class_locations']
		
		# 验证模式下固定采样索引
		if mode == 'val':
			self.fixed_samples = self._precompute_samples()
	
	def _precompute_samples(self) -> List:
		"""为验证集预计算采样点"""
		samples = []
		
		for case_idx, case_id in enumerate(self.case_ids):
			# 固定随机种子确保可重复性
			random.seed(case_idx)
			
			# 为每个案例生成固定的采样点
			for sample_idx in range(self.num_samples_per_volume):
				samples.append((case_idx, sample_idx))
		
		return samples
	
	def __len__(self) -> int:
		"""数据集长度"""
		if self.mode == 'val' and hasattr(self, 'fixed_samples'):
			return len(self.fixed_samples)
		return len(self.case_ids) * self.num_samples_per_volume
	
	def _get_data(self, case_id: str) -> Tuple[np.ndarray, np.ndarray]:
		"""加载数据"""
		data_file = os.path.join(self.data_dir, f"{case_id}.npz")
		data = np.load(data_file)['data']
		
		# 分离图像和标签
		if data.shape[0] > 1:  # 有标签
			image_data = data[0:1]
			label_data = data[1:2]
		else:  # 只有图像
			image_data = data
			label_data = None
		
		return image_data, label_data
	
	def _get_random_patch_coords(self, case_id: str,
	                             do_oversample_foreground: bool) -> Tuple[int, int, int]:
		"""
		获取随机补丁坐标，支持前景过采样

		参数:
			case_id: 案例ID
			do_oversample_foreground: 是否过采样前景

		返回:
			(z, y, x) 补丁中心坐标
		"""
		if case_id not in self.properties:
			# 如果没有属性信息，返回随机坐标
			return (np.random.randint(0, 100), np.random.randint(0, 100), np.random.randint(0, 100))
		
		shape = self.properties[case_id]['size_after_resampling']
		
		# 计算有效范围
		half_patch_size = np.array(self.patch_size) // 2
		lb_z, lb_y, lb_x = half_patch_size
		ub_z = shape[0] - half_patch_size[0]
		ub_y = shape[1] - half_patch_size[1]
		ub_x = shape[2] - half_patch_size[2]
		
		# 修复可能的边界问题
		if ub_z <= lb_z:
			lb_z = 0
			ub_z = shape[0]
		if ub_y <= lb_y:
			lb_y = 0
			ub_y = shape[1]
		if ub_x <= lb_x:
			lb_x = 0
			ub_x = shape[2]
		
		if do_oversample_foreground and case_id in self.class_locations and 1 in self.class_locations[case_id]:
			# 前景过采样
			foreground_locs = self.class_locations[case_id][1]
			
			if len(foreground_locs) > 0:
				# 随机选择一个前景位置
				selected_idx = np.random.choice(len(foreground_locs))
				z, y, x = foreground_locs[selected_idx]
				
				# 确保补丁在图像边界内
				z = max(lb_z, min(z, ub_z))
				y = max(lb_y, min(y, ub_y))
				x = max(lb_x, min(x, ub_x))
				
				return (z, y, x)
		
		# 如果不进行前景过采样或没有前景位置，则随机选择
		z = np.random.randint(lb_z, ub_z + 1)
		y = np.random.randint(lb_y, ub_y + 1)
		x = np.random.randint(lb_x, ub_x + 1)
		
		return (z, y, x)
	
	def _extract_patch(self, data: np.ndarray, center: Tuple[int, int, int]) -> np.ndarray:
		"""
		从数据中提取补丁

		参数:
			data: 输入数据 [c, z, y, x]
			center: 补丁中心坐标 (z, y, x)

		返回:
			提取的补丁
		"""
		z, y, x = center
		half_size = np.array(self.patch_size) // 2
		
		# 计算切片范围
		z_slice = slice(z - half_size[0], z + half_size[0] + (self.patch_size[0] % 2))
		y_slice = slice(y - half_size[1], y + half_size[1] + (self.patch_size[1] % 2))
		x_slice = slice(x - half_size[2], x + half_size[2] + (self.patch_size[2] % 2))
		
		# 提取补丁
		patch = data[:, z_slice, y_slice, x_slice]
		
		# 检查补丁形状，如果不匹配，则进行填充或裁剪
		if patch.shape[1:] != self.patch_size:
			# 创建空补丁
			result_patch = np.zeros((data.shape[0], *self.patch_size), dtype=data.dtype)
			
			# 计算有效区域
			valid_z_range = (0, min(patch.shape[1], self.patch_size[0]))
			valid_y_range = (0, min(patch.shape[2], self.patch_size[1]))
			valid_x_range = (0, min(patch.shape[3], self.patch_size[2]))
			
			# 复制有效区域
			result_patch[
			:,
			valid_z_range[0]:valid_z_range[1],
			valid_y_range[0]:valid_y_range[1],
			valid_x_range[0]:valid_x_range[1]
			] = patch[
			    :,
			    :valid_z_range[1]-valid_z_range[0],
			    :valid_y_range[1]-valid_y_range[0],
			    :valid_x_range[1]-valid_x_range[0]
			    ]
			
			return result_patch
		
		return patch
	
	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		"""获取数据样本"""
		if self.mode == 'val' and hasattr(self, 'fixed_samples'):
			# 验证模式使用预计算的样本
			case_idx, sample_idx = self.fixed_samples[idx]
			case_id = self.case_ids[case_idx]
			# 固定随机种子
			random.seed(case_idx * 100 + sample_idx)
			np.random.seed(case_idx * 100 + sample_idx)
		else:
			# 训练模式
			case_idx = idx // self.num_samples_per_volume
			sample_idx = idx % self.num_samples_per_volume
			case_id = self.case_ids[case_idx]
		
		# 决定是否过采样前景
		do_oversample = np.random.random() < self.oversample_foreground_percent
		
		# 加载数据
		image_data, label_data = self._get_data(case_id)
		
		# 获取随机补丁坐标
		center = self._get_random_patch_coords(case_id, do_oversample)
		
		# 提取补丁
		image_patch = self._extract_patch(image_data, center)
		
		if label_data is not None:
			label_patch = self._extract_patch(label_data, center)
		else:
			# 如果没有标签，创建全零标签
			label_patch = np.zeros((1, *self.patch_size), dtype=np.float32)
		
		# 应用数据增强
		if self.transform:
			image_patch, label_patch = self.transform(image_patch, label_patch)
		
		# 转换为张量
		image_tensor = torch.from_numpy(image_patch).float()
		label_tensor = torch.from_numpy(label_patch).float()
		
		return image_tensor, label_tensor# data/dataset.py

import os
import numpy as np
import pickle
import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional, Union
import random

class HepaticVesselDataset(Dataset):
    """肝脏血管分割数据集，使用nnUNet的采样策略"""
    
    def __init__(self, data_dir: str, properties_dir: str,
                 patch_size: Tuple[int, int, int] = (64, 64, 64),
                 mode: str = 'train',
                 splits_file: Optional[str] = None,
                 oversample_foreground_percent: float = 0.33,
                 num_samples_per_volume: int = 30,
                 transform=None):
        """
        初始化数据集
        
        参数:
            data_dir: 预处理数据目录
            properties_dir: 属性文件目录
            patch_size: 补丁大小
            mode: 'train'或'val'
            splits_file: 训练验证集划分文件
            oversample_foreground_percent: 前景过采样比例
            num_samples_per_volume: 每个体积采样的补丁数量
            transform: 数据增强转换
        """
        self.data_dir = data_dir
        self.properties_dir = properties_dir
        self.patch_size = patch_size
        self.mode = mode
        self.oversample_foreground_percent = oversample_foreground_percent
        self.num_samples_per_volume = num_samples_per_volume
        self.transform = transform
        
        # 加载划分
        if splits_file is not None and os.path.exists(splits_file):
            with open(splits_file, 'r') as f:
                splits = json.load(f)
            
            if mode == 'train':
                self.case_ids = splits['train']
            elif mode == 'val':
                self.case_ids = splits['val']
            else:
                raise ValueError(f"无效的模式: {mode}, 必须是'train'或'val'")
        else:
            # 如果没有划分文件，使用目录中的所有文件
            self.case_ids = [os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith('.npz')]
        
        # 加载属性
        self.properties = {}
        self.class_locations = {}
        
        for case_id in self.case_ids:
            prop_file = os.path.join(properties_dir, f"{case_id}.pkl")
            if os.path.exists(prop_file):
                with open(prop_file, 'rb') as f:
                    prop = pickle.load(f)
                    self.properties[case_id] = prop
                    if 'class_locations' in prop:
                        self.class_locations[case_id] = prop['class_locations']
        
        # 验证模式下固定采样索引
        if mode == 'val':
            self.fixed_samples = self._precompute_samples()
    
    def _precompute_samples(self) -> List:
        """为验证集预计算采样点"""
        samples = []
        
        for case_idx, case_id in enumerate(self.case_ids):
            # 固定随机种子确保可重复性
            random.seed(case_idx)
            
            # 为每个案例生成固定的采样点
            for sample_idx in range(self.num_samples_per_volume):
                samples.append((case_idx, sample_idx))
        
        return samples
    
    def __len__(self) -> int:
        """数据集长度"""
        if self.mode == 'val' and hasattr(self, 'fixed_samples'):
            return len(self.fixed_samples)
        return len(self.case_ids) * self.num_samples_per_volume
    
    def _get_data(self, case_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载数据"""
        data_file = os.path.join(self.data_dir, f"{case_id}.npz")
        data = np.load(data_file)['data']
        
        # 分离图像和标签
        if data.shape[0] > 1:  # 有标签
            image_data = data[0:1]
            label_data = data[1:2]
        else:  # 只有图像
            image_data = data
            label_data = None
            
        return image_data, label_data
    
    def _get_random_patch_coords(self, case_id: str,
                               do_oversample_foreground: bool) -> Tuple[int, int, int]:
        """
        获取随机补丁坐标，支持前景过采样
        
        参数:
            case_id: 案例ID
            do_oversample_foreground: 是否过采样前景
        
        返回:
            (z, y, x) 补丁中心坐标
        """
        if case_id not in self.properties:
            # 如果没有属性信息，返回随机坐标
            return (np.random.randint(0, 100), np.random.randint(0, 100), np.random.randint(0, 100))
        
        shape = self.properties[case_id]['size_after_resampling']
        
        # 计算有效范围
        half_patch_size = np.array(self.patch_size) // 2
        lb_z, lb_y, lb_x = half_patch_size
        ub_z = shape[0] - half_patch_size[0]
        ub_y = shape[1] - half_patch_size[1]
        ub_x = shape[2] - half_patch_size[2]
        
        # 修复可能的边界问题
        if ub_z <= lb_z:
            lb_z = 0
            ub_z = shape[0]
        if ub_y <= lb_y:
            lb_y = 0
            ub_y = shape[1]
        if ub_x <= lb_x:
            lb_x = 0
            ub_x = shape[2]
        
        if do_oversample_foreground and case_id in self.class_locations and 1 in self.class_locations[case_id]:
            # 前景过采样
            foreground_locs = self.class_locations[case_id][1]
            
            if len(foreground_locs) > 0:
                # 随机选择一个前景位置
                selected_idx = np.random.choice(len(foreground_locs))
                z, y, x = foreground_locs[selected_idx]
                
                # 确保补丁在图像边界内
                z = max(lb_z, min(z, ub_z))
                y = max(lb_y, min(y, ub_y))
                x = max(lb_x, min(x, ub_x))
                
                return (z, y, x)
        
        # 如果不进行前景过采样或没有前景位置，则随机选择
        z = np.random.randint(lb_z, ub_z + 1)
        y = np.random.randint(lb_y, ub_y + 1)
        x = np.random.randint(lb_x, ub_x + 1)
        
        return (z, y, x)
    
    def _extract_patch(self, data: np.ndarray, center: Tuple[int, int, int]) -> np.ndarray:
        """
        从数据中提取补丁
        
        参数:
            data: 输入数据 [c, z, y, x]
            center: 补丁中心坐标 (z, y, x)
            
        返回:
            提取的补丁
        """
        z, y, x = center
        half_size = np.array(self.patch_size) // 2
        
        # 计算切片范围
        z_slice = slice(z - half_size[0], z + half_size[0] + (self.patch_size[0] % 2))
        y_slice = slice(y - half_size[1], y + half_size[1] + (self.patch_size[1] % 2))
        x_slice = slice(x - half_size[2], x + half_size[2] + (self.patch_size[2] % 2))
        
        # 提取补丁
        patch = data[:, z_slice, y_slice, x_slice]
        
        # 检查补丁形状，如果不匹配，则进行填充或裁剪
        if patch.shape[1:] != self.patch_size:
            # 创建空补丁
            result_patch = np.zeros((data.shape[0], *self.patch_size), dtype=data.dtype)
            
            # 计算有效区域
            valid_z_range = (0, min(patch.shape[1], self.patch_size[0]))
            valid_y_range = (0, min(patch.shape[2], self.patch_size[1]))
            valid_x_range = (0, min(patch.shape[3], self.patch_size[2]))
            
            # 复制有效区域
            result_patch[
                :,
                valid_z_range[0]:valid_z_range[1],
                valid_y_range[0]:valid_y_range[1],
                valid_x_range[0]:valid_x_range[1]
            ] = patch[
                :,
                :valid_z_range[1]-valid_z_range[0],
                :valid_y_range[1]-valid_y_range[0],
                :valid_x_range[1]-valid_x_range[0]
            ]
            
            return result_patch
        
        return patch
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取数据样本"""
        if self.mode == 'val' and hasattr(self, 'fixed_samples'):
            # 验证模式使用预计算的样本
            case_idx, sample_idx = self.fixed_samples[idx]
            case_id = self.case_ids[case_idx]
            # 固定随机种子
            random.seed(case_idx * 100 + sample_idx)
            np.random.seed(case_idx * 100 + sample_idx)
        else:
            # 训练模式
            case_idx = idx // self.num_samples_per_volume
            sample_idx = idx % self.num_samples_per_volume
            case_id = self.case_ids[case_idx]
        
        # 决定是否过采样前景
        do_oversample = np.random.random() < self.oversample_foreground_percent
        
        # 加载数据
        image_data, label_data = self._get_data(case_id)
        
        # 获取随机补丁坐标
        center = self._get_random_patch_coords(case_id, do_oversample)
        
        # 提取补丁
        image_patch = self._extract_patch(image_data, center)
        
        if label_data is not None:
            label_patch = self._extract_patch(label_data, center)
        else:
            # 如果没有标签，创建全零标签
            label_patch = np.zeros((1, *self.patch_size), dtype=np.float32)
        
        # 应用数据增强
        if self.transform:
            image_patch, label_patch = self.transform(image_patch, label_patch)
        
        # 转换为张量
        image_tensor = torch.from_numpy(image_patch).float()
        label_tensor = torch.from_numpy(label_patch).float()
        
        return image_tensor, label_tensor