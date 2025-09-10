# data/crop.py

import os
import gc
import json
import pickle
import numpy as np
import nibabel as nib
from typing import Tuple, List, Dict, Optional
from multiprocessing import Pool
from pathlib import Path


def crop_to_nonzero(data: np.ndarray, seg: Optional[np.ndarray] = None, margin: int = 10) -> Tuple:
	"""
	裁剪到非零区域，保留边距

	参数:
		data: 图像数据
		seg: 标签数据(可选)
		margin: 边距大小(像素)

	返回:
		裁剪后的图像, 裁剪后的标签, 边界框信息
	"""
	# 创建前景掩码
	if seg is not None:
		mask = seg > 0  # 使用标签确定前景
	else:
		mask = data > np.min(data)  # 使用图像确定非零区域
	
	# 检查是否有前景
	if not np.any(mask):
		print("警告: 未检测到前景区域，返回原始数据")
		return data, seg, None
	
	# 获取边界框
	nonzero_indices = np.nonzero(mask)
	min_idx = [max(0, np.min(idx) - margin) for idx in nonzero_indices]
	max_idx = [min(mask.shape[i], np.max(nonzero_indices[i]) + margin) for i in range(len(nonzero_indices))]
	bbox = (min_idx, max_idx)
	
	# 裁剪数据
	slices = tuple(slice(min_idx[i], max_idx[i]) for i in range(len(min_idx)))
	cropped_data = data[slices]
	
	if seg is not None:
		cropped_seg = seg[slices]
		return cropped_data, cropped_seg, bbox
	else:
		return cropped_data, None, bbox


def process_case(args: Tuple) -> None:
	"""
	处理单个案例的裁剪

	参数:
		args: 包含(image_file, label_file, output_dir, case_id)的元组
	"""
	image_file, label_file, output_dir, case_id = args
	
	try:
		# 创建输出目录
		images_dir = os.path.join(output_dir, 'cropped', 'images')
		labels_dir = os.path.join(output_dir, 'cropped', 'labels')
		props_dir = os.path.join(output_dir, 'cropped', 'properties')
		
		os.makedirs(images_dir, exist_ok=True)
		os.makedirs(labels_dir, exist_ok=True)
		os.makedirs(props_dir, exist_ok=True)
		
		# 1. 加载数据
		if image_file.endswith('.nii.gz'):
			image_nii = nib.load(image_file)
			image_data = image_nii.get_fdata().astype(np.float32)
			original_spacing = image_nii.header.get_zooms()
		else:  # 假设是npy文件
			image_data = np.load(image_file)
			# 如果是npy文件，可能需要外部提供spacing
			original_spacing = (1.0, 1.0, 1.0)  # 默认值
		
		if label_file and os.path.exists(label_file):
			if label_file.endswith('.nii.gz'):
				label_nii = nib.load(label_file)
				label_data = label_nii.get_fdata().astype(np.float32)
			else:  # 假设是npy文件
				label_data = np.load(label_file)
		else:
			label_data = None
		
		# 2. 执行裁剪
		cropped_image, cropped_label, bbox = crop_to_nonzero(image_data, label_data, margin=10)
		
		# 3. 保存裁剪后的数据
		np.save(os.path.join(images_dir, f"{case_id}.npy"), cropped_image)
		if cropped_label is not None:
			np.save(os.path.join(labels_dir, f"{case_id}.npy"), cropped_label)
		
		# 4. 保存属性
		properties = {
			'original_spacing': original_spacing,
			'original_shape': image_data.shape,
			'bbox': bbox,
			'case_id': case_id
		}
		
		with open(os.path.join(props_dir, f"{case_id}.pkl"), 'wb') as f:
			pickle.dump(properties, f)
		
		# 5. 清理内存
		del image_data, cropped_image
		if label_data is not None:
			del label_data, cropped_label
		gc.collect()
		
		print(f"案例 {case_id} 裁剪完成")
	
	except Exception as e:
		print(f"处理案例 {case_id} 时出错: {e}")
		import traceback
		traceback.print_exc()


def run_cropping(image_files: List[str], label_files: Optional[List[str]] = None,
                 output_dir: str = 'preprocess', num_workers: int = 8) -> None:
	"""
	批量执行非零裁剪

	参数:
		image_files: 图像文件路径列表
		label_files: 标签文件路径列表(可选)
		output_dir: 输出目录
		num_workers: 并行处理的工作进程数
	"""
	if label_files is None:
		label_files = [None] * len(image_files)
	
	# 准备参数
	args_list = []
	for i, img_file in enumerate(image_files):
		# 提取案例ID
		case_id = os.path.splitext(os.path.basename(img_file))[0].replace('.nii', '')
		
		args_list.append((img_file, label_files[i], output_dir, case_id))
	
	# 多进程执行裁剪
	if num_workers > 1:
		with Pool(num_workers) as p:
			p.map(process_case, args_list)
	else:
		# 串行处理
		for args in args_list:
			process_case(args)
	
	print(f"所有案例裁剪完成，结果保存在 {output_dir}/cropped/")


if __name__ == "__main__":
	import argparse
	
	parser = argparse.ArgumentParser(description='执行非零裁剪')
	parser.add_argument('--images_dir', type=str, required=True, help='图像目录')
	parser.add_argument('--labels_dir', type=str, required=True, help='标签目录')
	parser.add_argument('--output_dir', type=str, default='preprocess', help='输出目录')
	parser.add_argument('--num_workers', type=int, default=8, help='工作进程数')
	
	args = parser.parse_args()
	
	# 获取所有图像和标签文件
	image_files = sorted([os.path.join(args.images_dir, f) for f in os.listdir(args.images_dir)
	                      if f.endswith('.nii.gz') or f.endswith('.npy')])
	
	label_files = sorted([os.path.join(args.labels_dir, f) for f in os.listdir(args.labels_dir)
	                      if f.endswith('.nii.gz') or f.endswith('.npy')])
	
	# 执行裁剪
	run_cropping(image_files, label_files, args.output_dir, args.num_workers)