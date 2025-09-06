# data/preprocess.py

import os
import numpy as np
import nibabel as nib
import pickle
from typing import Tuple, List, Dict, Union, Optional
from skimage.transform import resize
from multiprocessing.pool import Pool
import json
import argparse
import yaml


class HepaticVesselPreprocessor:
	"""肝脏血管数据预处理器，基于nnUNet的处理策略，针对血管分割优化"""
	
	def __init__(self,
	             target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
	             intensity_bounds: Tuple[float, float] = (-100, 300),
	             use_nonzero_mask: bool = True,
	             resample_separate_z_anisotropy_threshold: float = 3.0,
	             order_data: int = 3,
	             order_seg: int = 0,
	             output_dir: str = "/home/bingxing2/home/scx7as9/run/CSUqx/CH/data/preprocessed"):
		"""
		初始化预处理器

		参数:
			target_spacing: 目标体素间距 (z, y, x)
			intensity_bounds: CT强度窗口 (min, max)
			use_nonzero_mask: 是否使用非零掩码进行归一化
			resample_separate_z_anisotropy_threshold: z轴单独重采样的阈值
			order_data: 图像重采样的插值阶数
			order_seg: 分割标签重采样的插值阶数
			output_dir: 输出目录
		"""
		self.target_spacing = target_spacing
		self.intensity_bounds = intensity_bounds
		self.use_nonzero_mask = use_nonzero_mask
		self.resample_separate_z_anisotropy_threshold = resample_separate_z_anisotropy_threshold
		self.order_data = order_data
		self.order_seg = order_seg
		self.output_dir = output_dir
		
		# 创建目录结构
		self.data_dir = os.path.join(output_dir, "data")
		self.properties_dir = os.path.join(output_dir, "properties")
		
		os.makedirs(self.data_dir, exist_ok=True)
		os.makedirs(self.properties_dir, exist_ok=True)
	
	def preprocess_case(self, image_file: str, label_file: Optional[str] = None) -> Tuple:
		"""
		处理单个案例

		参数:
			image_file: 图像文件路径(.nii.gz)
			label_file: 标签文件路径(.nii.gz)，可选

		返回:
			处理后的图像和标签数据，以及元数据
		"""
		# 提取案例ID
		case_id = os.path.splitext(os.path.basename(image_file))[0]
		if case_id.endswith('.nii'):  # 处理.nii.gz文件
			case_id = os.path.splitext(case_id)[0]
		
		# 加载图像
		img_nii = nib.load(image_file)
		img_data = img_nii.get_fdata().astype(np.float32)
		original_spacing = img_nii.header.get_zooms()
		
		# 转换为nnUNet期望的格式 [c, z, y, x]
		img_data = img_data[np.newaxis, ...]
		
		# 加载标签(如果有)
		if label_file is not None and os.path.exists(label_file):
			label_nii = nib.load(label_file)
			label_data = label_nii.get_fdata().astype(np.float32)
			label_data = label_data[np.newaxis, ...]
		else:
			label_data = None
		
		# 重采样到目标间距
		img_data, label_data = self.resample_data(
			img_data, label_data, original_spacing, self.target_spacing
		)
		
		# 归一化
		img_data = self.normalize_ct(img_data, label_data)
		
		# 保存元数据
		properties = {
			'original_spacing': original_spacing,
			'spacing_after_resampling': self.target_spacing,
			'size_after_resampling': img_data.shape[1:],
			'intensity_properties': {
				'min': np.min(img_data),
				'max': np.max(img_data),
				'mean': np.mean(img_data),
				'std': np.std(img_data),
				'percentile_00_5': np.percentile(img_data, 0.5),
				'percentile_99_5': np.percentile(img_data, 99.5)
			}
		}
		
		# 提取前景类位置信息
		if label_data is not None:
			properties['class_locations'] = self._get_class_locations(label_data)
		
		# 保存到磁盘
		self._save_to_disk(case_id, img_data, label_data, properties)
		
		return img_data, label_data, properties
	
	def _get_class_locations(self, seg: np.ndarray) -> Dict:
		"""
		提取前景类位置信息

		参数:
			seg: 分割标签 [c, z, y, x]

		返回:
			各类别的位置字典 {class_id: coordinates}
		"""
		class_locations = {}
		
		# 获取唯一类别
		seg_arr = seg[0]  # 取第一个通道
		unique_classes = np.unique(seg_arr)
		
		# 提取每个类别的位置
		for c in unique_classes:
			if c < 0:  # 跳过负值标签
				continue
			
			# 找到该类别的所有位置
			class_locs = np.argwhere(seg_arr == c)
			
			# 存储位置坐标
			if len(class_locs) > 0:
				class_locations[int(c)] = class_locs
			else:
				class_locations[int(c)] = []
		
		return class_locations
	
	def _save_to_disk(self, case_id: str, img_data: np.ndarray,
	                  label_data: Optional[np.ndarray], properties: Dict) -> None:
		"""
		保存预处理结果到磁盘

		参数:
			case_id: 案例标识符
			img_data: 预处理后的图像数据
			label_data: 预处理后的标签数据
			properties: 元数据属性
		"""
		# 合并数据
		if label_data is not None:
			all_data = np.vstack((img_data, label_data)).astype(np.float32)
		else:
			all_data = img_data.astype(np.float32)
		
		# 保存.npz文件
		np.savez_compressed(
			os.path.join(self.data_dir, f"{case_id}.npz"),
			data=all_data
		)
		
		# 保存属性文件(.pkl)
		with open(os.path.join(self.properties_dir, f"{case_id}.pkl"), 'wb') as f:
			pickle.dump(properties, f)
		
		print(f"已保存预处理结果: {case_id}")
	
	def normalize_ct(self, data: np.ndarray, seg: Optional[np.ndarray] = None) -> np.ndarray:
		"""
		CT特定的归一化，包括窗宽窗位和标准化

		参数:
			data: 输入图像数据 [c, z, y, x]
			seg: 输入分割标签 [c, z, y, x]

		返回:
			归一化后的图像数据
		"""
		# 去除NaN
		data[np.isnan(data)] = 0
		
		# 应用窗宽窗位，适合肝脏血管的值
		lower_bound, upper_bound = self.intensity_bounds
		data = np.clip(data, lower_bound, upper_bound)
		
		# 使用均值和标准差归一化
		if self.use_nonzero_mask and seg is not None:
			mask = seg[-1] >= 0  # 使用标签掩码
		else:
			mask = np.ones(data.shape[1:], dtype=bool)  # 使用全图
		
		# 均值标准差归一化
		mean = data[0][mask].mean()
		std = data[0][mask].std()
		data[0] = (data[0] - mean) / (std + 1e-8)
		
		# 将背景区域置零(可选)
		if self.use_nonzero_mask and seg is not None:
			data[0][mask == 0] = 0
		
		return data
	
	def resample_data(self, data: np.ndarray, seg: Optional[np.ndarray],
	                  original_spacing: Tuple[float, float, float],
	                  target_spacing: Tuple[float, float, float]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
		"""
		重采样数据到目标间距

		参数:
			data: 输入图像数据 [c, z, y, x]
			seg: 输入分割标签 [c, z, y, x]
			original_spacing: 原始体素间距 (z, y, x)
			target_spacing: 目标体素间距 (z, y, x)

		返回:
			重采样后的图像和标签数据
		"""
		original_spacing = np.array(original_spacing)
		target_spacing = np.array(target_spacing)
		
		# 计算新的形状
		shape = np.array(data[0].shape)
		new_shape = np.round((original_spacing / target_spacing) * shape).astype(int)
		
		# 检查是否需要单独处理z轴
		do_separate_z = self._get_do_separate_z(original_spacing, target_spacing)
		
		if do_separate_z:
			# 各向异性数据处理
			axis = self._get_lowres_axis(original_spacing)
			if axis is not None:
				# 分别处理低分辨率轴
				data_reshaped, seg_reshaped = self._resample_along_axis(
					data, seg, axis, new_shape, original_spacing, target_spacing
				)
			else:
				# 所有轴都是各向同性的情况
				data_reshaped, seg_reshaped = self._resample_isotropic(
					data, seg, new_shape
				)
		else:
			# 直接进行3D重采样
			data_reshaped, seg_reshaped = self._resample_isotropic(
				data, seg, new_shape
			)
		
		# 确保分割标签值合法
		if seg_reshaped is not None:
			seg_reshaped[seg_reshaped < -1] = 0
		
		return data_reshaped, seg_reshaped
	
	def _get_do_separate_z(self, original_spacing: np.ndarray, target_spacing: np.ndarray) -> bool:
		"""
		检查是否需要单独处理z轴
		"""
		spacing_ratio = np.max(original_spacing) / np.min(original_spacing)
		return spacing_ratio > self.resample_separate_z_anisotropy_threshold
	
	def _get_lowres_axis(self, spacing: np.ndarray) -> Optional[int]:
		"""
		获取低分辨率轴
		"""
		max_spacing = np.max(spacing)
		axis = np.where(spacing == max_spacing)[0]
		return axis[0] if len(axis) == 1 else None
	
	def _resample_along_axis(self, data: np.ndarray, seg: Optional[np.ndarray],
	                         axis: int, new_shape: np.ndarray,
	                         original_spacing: np.ndarray, target_spacing: np.ndarray) -> Tuple:
		"""
		单独处理某一轴的重采样
		"""
		# 沿指定轴重采样
		if axis == 0:
			# 处理z轴
			new_shape_2d = new_shape[1:]
			data_reshaped = []
			
			for c in range(data.shape[0]):
				data_c = []
				for z in range(data.shape[1]):
					slice_2d = resize(data[c, z], new_shape_2d, order=self.order_data, mode='edge', anti_aliasing=False)
					data_c.append(slice_2d)
				data_c = np.stack(data_c, axis=0)
				
				# 沿z轴重采样
				if data.shape[1] != new_shape[0]:
					data_c = resize(data_c, new_shape, order=0, mode='edge', anti_aliasing=False)
				
				data_reshaped.append(data_c[None])
			
			data_reshaped = np.vstack(data_reshaped)
			
			if seg is not None:
				seg_reshaped = []
				for c in range(seg.shape[0]):
					seg_c = []
					for z in range(seg.shape[1]):
						slice_2d = resize(seg[c, z], new_shape_2d, order=self.order_seg, mode='constant')
						seg_c.append(slice_2d)
					seg_c = np.stack(seg_c, axis=0)
					
					# 沿z轴重采样
					if seg.shape[1] != new_shape[0]:
						seg_c = resize(seg_c, new_shape, order=0, mode='constant')
					
					seg_reshaped.append(seg_c[None])
				
				seg_reshaped = np.vstack(seg_reshaped)
			else:
				seg_reshaped = None
		else:
			# 处理其他轴
			# 为简洁起见，这里使用isotropic方式处理，实际应该根据axis适配
			data_reshaped, seg_reshaped = self._resample_isotropic(data, seg, new_shape)
		
		return data_reshaped, seg_reshaped
	
	def _resample_isotropic(self, data: np.ndarray, seg: Optional[np.ndarray],
	                        new_shape: np.ndarray) -> Tuple:
		"""
		各向同性重采样
		"""
		data_reshaped = []
		for c in range(data.shape[0]):
			data_c = resize(data[c], new_shape, order=self.order_data, mode='edge', anti_aliasing=False)
			data_reshaped.append(data_c[None])
		data_reshaped = np.vstack(data_reshaped)
		
		if seg is not None:
			seg_reshaped = []
			for c in range(seg.shape[0]):
				seg_c = resize(seg[c], new_shape, order=self.order_seg, mode='constant')
				seg_reshaped.append(seg_c[None])
			seg_reshaped = np.vstack(seg_reshaped)
		else:
			seg_reshaped = None
		
		return data_reshaped, seg_reshaped
	
	def preprocess_batch(self, image_files: List[str], label_files: Optional[List[str]] = None,
	                     num_workers: int = 8) -> None:
		"""
		批量处理多个案例

		参数:
			image_files: 图像文件路径列表
			label_files: 标签文件路径列表(可选)
			num_workers: 并行处理的工作进程数
		"""
		if label_files is None:
			label_files = [None] * len(image_files)
		
		print(f"开始预处理 {len(image_files)} 个案例，结果将保存到: {self.output_dir}")
		
		if num_workers > 1:
			# 并行处理
			with Pool(num_workers) as p:
				p.starmap(self.preprocess_case, zip(image_files, label_files))
		else:
			# 串行处理
			for img_file, lbl_file in zip(image_files, label_files):
				self.preprocess_case(img_file, lbl_file)
		
		print(f"预处理完成，结果已保存到: {self.output_dir}")
		
		# 保存数据集全局属性
		self.save_dataset_properties(image_files)
		
		# 创建训练验证集划分
		self.create_split(image_files)
	
	def save_dataset_properties(self, image_files: List[str]) -> None:
		"""
		保存数据集全局属性

		参数:
			image_files: 图像文件路径列表
		"""
		# 收集所有类别
		all_classes = set()
		
		# 读取所有预处理后的属性文件
		for img_file in image_files:
			case_id = os.path.splitext(os.path.basename(img_file))[0]
			if case_id.endswith('.nii'):  # 处理.nii.gz文件
				case_id = os.path.splitext(case_id)[0]
			
			pkl_file = os.path.join(self.properties_dir, f"{case_id}.pkl")
			if os.path.exists(pkl_file):
				with open(pkl_file, 'rb') as f:
					properties = pickle.load(f)
					if 'class_locations' in properties:
						all_classes.update(properties['class_locations'].keys())
		
		# 保存数据集属性
		dataset_properties = {
			'all_classes': sorted(list(all_classes)),
			'spacing': self.target_spacing,
			'intensityproperties': {
				'0': {  # CT通道
					'mean': 0,
					'sd': 1,
					'percentile_00_5': self.intensity_bounds[0],
					'percentile_99_5': self.intensity_bounds[1]
				}
			}
		}
		
		# 保存为pickle文件
		with open(os.path.join(self.output_dir, "dataset_properties.pkl"), 'wb') as f:
			pickle.dump(dataset_properties, f)
		
		print(f"数据集属性已保存: {os.path.join(self.output_dir, 'dataset_properties.pkl')}")
	
	def create_split(self, image_files: List[str], val_percent: float = 0.2, seed: int = 42) -> None:
		"""
		创建训练验证集划分

		参数:
			image_files: 图像文件路径列表
			val_percent: 验证集比例
			seed: 随机种子
		"""
		np.random.seed(seed)
		
		# 提取所有案例ID
		case_ids = []
		for img_file in image_files:
			case_id = os.path.splitext(os.path.basename(img_file))[0]
			if case_id.endswith('.nii'):  # 处理.nii.gz文件
				case_id = os.path.splitext(case_id)[0]
			case_ids.append(case_id)
		
		# 随机打乱
		indices = np.arange(len(case_ids))
		np.random.shuffle(indices)
		
		# 划分训练集和验证集
		n_val = int(len(case_ids) * val_percent)
		train_indices = indices[n_val:]
		val_indices = indices[:n_val]
		
		train_cases = [case_ids[i] for i in train_indices]
		val_cases = [case_ids[i] for i in val_indices]
		
		# 创建划分文件
		splits = {
			"train": train_cases,
			"val": val_cases
		}
		
		# 保存为JSON文件
		with open(os.path.join(self.output_dir, "splits.json"), 'w') as f:
			json.dump(splits, f, indent=4)
		
		print(f"训练验证集划分已保存: {os.path.join(self.output_dir, 'splits.json')}")


def main():
	parser = argparse.ArgumentParser(description='预处理肝脏血管数据')
	parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
	args = parser.parse_args()
	
	# 加载配置
	with open(args.config, 'r', encoding='utf-8') as f:
		config = yaml.safe_load(f)
		
	image_dir = config['data']['data_dir']
	label_dir = config['data']['label_dir']
	output_dir = config['data']['preprocessed_dir']
	
	# 获取所有图像和标签文件
	image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
	label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.nii.gz')])
	
	# 初始化预处理器
	preprocessor = HepaticVesselPreprocessor(
		target_spacing=(1.0, 1.0, 1.0),
		intensity_bounds=(-100, 300),
		output_dir=output_dir
	)
	
	# 批量处理所有案例
	preprocessor.preprocess_batch(image_files, label_files, num_workers=8)


if __name__ == "__main__":
	main()