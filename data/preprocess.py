# data/preprocess.py

import os
import gc
import json
import pickle
import numpy as np
from typing import Tuple, List, Dict, Optional
from multiprocessing import Pool
from scipy.ndimage import binary_erosion, zoom
from pathlib import Path


def resample_data(data: np.ndarray, seg: Optional[np.ndarray],
                  original_spacing: Tuple[float, float, float],
                  target_spacing: Tuple[float, float, float]) -> Tuple:
	"""
	重采样数据到目标间距

	参数:
		data: 图像数据
		seg: 标签数据(可选)
		original_spacing: 原始体素间距
		target_spacing: 目标体素间距

	返回:
		重采样后的图像, 重采样后的标签
	"""
	# 检查是否需要重采样
	if np.allclose(original_spacing, target_spacing):
		return data, seg
	
	# 计算缩放因子
	scale_factors = np.array(original_spacing) / np.array(target_spacing)
	
	# 处理z轴各向异性（如果需要）
	do_separate_z = False
	if scale_factors[0] > 2 or scale_factors[0] < 0.5:
		do_separate_z = True
	
	if do_separate_z:
		# 单独处理z轴
		# 先处理xy平面
		new_shape_xy = np.round(np.array(data.shape[1:]) * scale_factors[1:]).astype(int)
		resampled_data_xy = np.zeros((data.shape[0], new_shape_xy[0], new_shape_xy[1]), dtype=data.dtype)
		
		for z in range(data.shape[0]):
			resampled_data_xy[z] = zoom(data[z], scale_factors[1:], order=3, mode='nearest')
		
		# 再处理z轴
		new_shape_z = np.round(data.shape[0] * scale_factors[0]).astype(int)
		resampled_data = zoom(resampled_data_xy, (scale_factors[0], 1, 1), order=0, mode='nearest')
		
		# 标签处理类似，但使用最近邻插值
		if seg is not None:
			resampled_seg_xy = np.zeros((seg.shape[0], new_shape_xy[0], new_shape_xy[1]), dtype=seg.dtype)
			for z in range(seg.shape[0]):
				resampled_seg_xy[z] = zoom(seg[z], scale_factors[1:], order=0, mode='nearest')
			
			resampled_seg = zoom(resampled_seg_xy, (scale_factors[0], 1, 1), order=0, mode='nearest')
		else:
			resampled_seg = None
	else:
		# 直接3D重采样
		new_shape = np.round(np.array(data.shape) * scale_factors).astype(int)
		resampled_data = zoom(data, scale_factors, order=3, mode='nearest')
		
		if seg is not None:
			resampled_seg = zoom(seg, scale_factors, order=0, mode='nearest')
		else:
			resampled_seg = None
	
	return resampled_data, resampled_seg


def normalize_ct(data: np.ndarray, clip_window: Tuple[float, float] = (-200, 200)) -> Tuple:
	"""
	血管CT数据标准化

	参数:
		data: 图像数据
		clip_window: 裁剪窗口

	返回:
		归一化后的数据, 归一化参数
	"""
	# 裁剪到合适的HU值范围
	data_clipped = np.clip(data, clip_window[0], clip_window[1])
	
	# Z-score归一化
	mean = np.mean(data_clipped)
	std = np.std(data_clipped)
	normalized = (data_clipped - mean) / (std + 1e-8)
	
	return normalized, {'mean': mean, 'std': std, 'clip_window': clip_window}


def extract_foreground_locations(seg: np.ndarray, max_points: int = 10000) -> Dict:
	"""
	提取前景位置，用于过采样

	参数:
		seg: 分割标签
		max_points: 最大点数限制

	返回:
		前景位置字典
	"""
	# 获取前景点
	foreground_mask = seg > 0
	foreground_points = np.argwhere(foreground_mask)
	
	# 提取边界和内部区域
	eroded = binary_erosion(foreground_mask, iterations=2)
	boundary = foreground_mask & (~eroded)
	interior = eroded
	
	# 获取位置点
	boundary_points = np.argwhere(boundary)
	interior_points = np.argwhere(interior)
	background_points = np.argwhere(~foreground_mask)
	
	# 如果点太多，进行下采样
	if len(boundary_points) > max_points // 3:
		indices = np.random.choice(len(boundary_points), max_points // 3, replace=False)
		boundary_points = boundary_points[indices]
	
	if len(interior_points) > max_points // 3:
		indices = np.random.choice(len(interior_points), max_points // 3, replace=False)
		interior_points = interior_points[indices]
	
	if len(background_points) > max_points // 3:
		indices = np.random.choice(len(background_points), max_points // 3, replace=False)
		background_points = background_points[indices]
	
	return {
		'foreground': foreground_points[:max_points] if len(foreground_points) > max_points else foreground_points,
		'boundary': boundary_points,
		'interior': interior_points,
		'background': background_points
	}


def process_case(args: Tuple) -> None:
	"""
	处理单个案例的重采样和归一化

	参数:
		args: (cropped_image_file, cropped_label_file, properties_file, output_dir, target_spacing, case_id)
	"""
	cropped_image_file, cropped_label_file, properties_file, output_dir, target_spacing, case_id = args
	
	try:
		# 创建输出目录
		data_dir = os.path.join(output_dir, 'processed', 'data')
		props_dir = os.path.join(output_dir, 'processed', 'properties')
		
		os.makedirs(data_dir, exist_ok=True)
		os.makedirs(props_dir, exist_ok=True)
		
		# 1. 加载裁剪后的数据
		cropped_image = np.load(cropped_image_file)
		if cropped_label_file and os.path.exists(cropped_label_file):
			cropped_label = np.load(cropped_label_file)
		else:
			cropped_label = None
		
		# 2. 加载属性
		with open(properties_file, 'rb') as f:
			properties = pickle.load(f)
		
		original_spacing = properties['original_spacing']
		
		# 3. 重采样到目标间距
		resampled_image, resampled_label = resample_data(
			cropped_image, cropped_label, original_spacing, target_spacing
		)
		
		# 4. 图像归一化
		normalized_image, norm_params = normalize_ct(resampled_image)
		
		# 5. 提取前景位置（用于训练时的过采样）
		class_locations = None
		if resampled_label is not None:
			class_locations = extract_foreground_locations(resampled_label)
		
		# 6. 更新属性
		process_props = properties.copy()
		process_props.update({
			'spacing_after_resampling': target_spacing,
			'shape_after_resampling': normalized_image.shape,
			'normalization_params': norm_params,
			'class_locations': class_locations
		})
		
		# 7. 保存处理后的数据
		if resampled_label is not None:
			combined_data = np.stack([normalized_image, resampled_label])
			np.save(os.path.join(data_dir, f"{case_id}.npy"), combined_data)
		else:
			np.save(os.path.join(data_dir, f"{case_id}.npy"), normalized_image[np.newaxis])
		
		# 8. 保存属性
		with open(os.path.join(props_dir, f"{case_id}.pkl"), 'wb') as f:
			pickle.dump(process_props, f)
		
		# 9. 清理内存
		del cropped_image, normalized_image, resampled_image
		if cropped_label is not None:
			del cropped_label, resampled_label
		gc.collect()
		
		print(f"案例 {case_id} 处理完成")
	
	except Exception as e:
		print(f"处理案例 {case_id} 时出错: {e}")
		import traceback
		traceback.print_exc()


def create_dataset_properties(output_dir: str) -> None:
	"""
	创建全局数据集属性文件

	参数:
		output_dir: 输出目录
	"""
	props_dir = os.path.join(output_dir, 'processed', 'properties')
	
	# 收集所有案例的属性
	all_spacings = []
	all_shapes = []
	all_classes = set()
	intensity_stats = {'mean': [], 'std': []}
	
	for props_file in os.listdir(props_dir):
		if props_file.endswith('.pkl'):
			with open(os.path.join(props_dir, props_file), 'rb') as f:
				props = pickle.load(f)
				
				all_spacings.append(props['spacing_after_resampling'])
				all_shapes.append(props['shape_after_resampling'])
				
				if 'class_locations' in props and props['class_locations'] is not None:
					for loc_type in props['class_locations']:
						if len(props['class_locations'][loc_type]) > 0:
							all_classes.add(1)  # 血管分割中通常只有一个前景类
				
				if 'normalization_params' in props:
					intensity_stats['mean'].append(props['normalization_params']['mean'])
					intensity_stats['std'].append(props['normalization_params']['std'])
	
	# 计算全局属性
	dataset_properties = {
		'spacing_median': np.median(all_spacings, axis=0).tolist(),
		'median_shape': np.median(all_shapes, axis=0).astype(int).tolist(),
		'classes': sorted(list(all_classes) + [0]),  # 添加背景类
		'intensity_stats': {
			'mean': float(np.mean(intensity_stats['mean'])),
			'std': float(np.mean(intensity_stats['std']))
		}
	}
	
	# 保存全局属性
	with open(os.path.join(output_dir, 'dataset_properties.pkl'), 'wb') as f:
		pickle.dump(dataset_properties, f)
	
	print(f"数据集属性已保存至 {os.path.join(output_dir, 'dataset_properties.pkl')}")


def create_splits(output_dir: str, val_percent: float = 0.2, seed: int = 42) -> None:
	"""
	创建训练/验证集划分

	参数:
		output_dir: 输出目录
		val_percent: 验证集比例
		seed: 随机种子
	"""
	data_dir = os.path.join(output_dir, 'processed', 'data')
	
	# 获取所有案例ID
	case_ids = [os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith('.npy')]
	case_ids.sort()
	
	# 设置随机种子
	np.random.seed(seed)
	np.random.shuffle(case_ids)
	
	# 分割训练集和验证集
	val_size = max(1, int(len(case_ids) * val_percent))
	train_ids = case_ids[val_size:]
	val_ids = case_ids[:val_size]
	
	# 创建划分
	splits = {
		'train': train_ids,
		'val': val_ids
	}
	
	# 保存划分
	with open(os.path.join(output_dir, 'splits.json'), 'w') as f:
		json.dump(splits, f, indent=4)
	
	print(f"数据集划分已保存至 {os.path.join(output_dir, 'splits.json')}")
	print(f"训练集: {len(train_ids)} 案例, 验证集: {len(val_ids)} 案例")


def run_preprocessing(cropped_dir: str, output_dir: str, target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                      num_workers: int = 8) -> None:
	"""
	执行重采样和归一化

	参数:
		cropped_dir: 裁剪结果目录
		output_dir: 输出目录
		target_spacing: 目标体素间距
		num_workers: 工作进程数
	"""
	# 准备参数
	args_list = []
	
	# 获取所有裁剪后的案例
	images_dir = os.path.join(cropped_dir, 'images')
	labels_dir = os.path.join(cropped_dir, 'labels')
	props_dir = os.path.join(cropped_dir, 'properties')
	
	for img_file in os.listdir(images_dir):
		if img_file.endswith('.npy'):
			case_id = os.path.splitext(img_file)[0]
			
			cropped_image_file = os.path.join(images_dir, img_file)
			cropped_label_file = os.path.join(labels_dir, img_file)
			properties_file = os.path.join(props_dir, f"{case_id}.pkl")
			
			if not os.path.exists(cropped_label_file):
				cropped_label_file = None
			
			args_list.append((
				cropped_image_file,
				cropped_label_file,
				properties_file,
				output_dir,
				target_spacing,
				case_id
			))
	
	# 多进程执行预处理
	if num_workers > 1:
		with Pool(num_workers) as p:
			p.map(process_case, args_list)
	else:
		# 串行处理
		for args in args_list:
			process_case(args)
	
	# 创建全局数据集属性
	create_dataset_properties(output_dir)
	
	# 创建训练/验证集划分
	create_splits(output_dir)
	
	print(f"所有案例处理完成，结果保存在 {os.path.join(output_dir, 'processed')}")


if __name__ == "__main__":
	import argparse
	
	parser = argparse.ArgumentParser(description='执行重采样和归一化')
	parser.add_argument('--cropped_dir', type=str, required=True, help='裁剪结果目录')
	parser.add_argument('--output_dir', type=str, default='preprocess', help='输出目录')
	parser.add_argument('--target_spacing', type=float, nargs=3, default=[1.0, 1.0, 1.0], help='目标体素间距')
	parser.add_argument('--num_workers', type=int, default=8, help='工作进程数')
	
	args = parser.parse_args()
	
	# 执行预处理
	run_preprocessing(
		args.cropped_dir,
		args.output_dir,
		tuple(args.target_spacing),
		args.num_workers
	)