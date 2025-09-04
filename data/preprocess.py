# data/preprocess.py
import os
import numpy as np
import nibabel as nib
from skimage.filters import frangi
import pickle
from tqdm import tqdm
import argparse


def preprocess_dataset(images_dir, labels_dir, output_dir, scales=range(1, 5), patch_size=64, samples_per_volume=30):
	"""预处理数据集，计算并存储Frangi响应以及有效采样点"""
	# 创建输出目录
	os.makedirs(output_dir, exist_ok=True)
	
	# 获取所有文件路径
	volume_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
	label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.nii.gz')])
	
	print(f"找到 {len(volume_files)} 个体积文件和 {len(label_files)} 个标签文件")
	
	# 导入FrangiSampler用于生成采样点
	from data.dataset import FrangiSampler
	sampler = FrangiSampler(patch_size=patch_size)
	
	# 全局索引映射
	global_index_mapping = []
	half_size = patch_size // 2
	
	# 处理每个案例
	for vol_idx, (vol_file, lab_file) in enumerate(tqdm(zip(volume_files, label_files), total=len(volume_files))):
		case_id = os.path.splitext(os.path.splitext(vol_file)[0])[0]  # 移除.nii.gz
		
		# 加载数据
		vol_path = os.path.join(images_dir, vol_file)
		lab_path = os.path.join(labels_dir, lab_file)
		
		vol_nii = nib.load(vol_path)
		lab_nii = nib.load(lab_path)
		
		vol_data = vol_nii.get_fdata().astype(np.float32)
		lab_data = lab_nii.get_fdata().astype(np.float32)
		
		# 标准化CT数据
		p_low = np.percentile(vol_data, 0.5)
		p_high = np.percentile(vol_data, 99.5)
		if p_high > p_low:
			vol_norm = np.clip((vol_data - p_low) / (p_high - p_low), 0, 1)
		else:
			vol_norm = np.zeros_like(vol_data)
		
		# 计算Frangi响应
		vessel_map = np.zeros_like(vol_norm)
		for scale in scales:
			response = frangi(
				vol_norm,
				scale_range=(scale, scale),
				scale_step=1,
				black_ridges=False
			)
			vessel_map = np.maximum(vessel_map, response)
		
		# 移除低响应区域
		threshold = 0.05
		vessel_map[vessel_map < threshold] = 0
		
		# 对比度增强
		vessel_map = vessel_map ** 1.5
		
		# 重新归一化
		if vessel_map.max() > 0:
			vessel_map = vessel_map / vessel_map.max()
		
		# 保存预处理结果和元数据
		metadata = {
			'volume_path': vol_path,
			'label_path': lab_path,
			'shape': vol_data.shape
		}
		
		# 保存Frangi响应和元数据
		np.save(os.path.join(output_dir, f"{case_id}_frangi.npy"), vessel_map)
		with open(os.path.join(output_dir, f"{case_id}_metadata.pkl"), 'wb') as f:
			pickle.dump(metadata, f)
		
		# 生成采样点
		points = sampler.generate_sample_points(vessel_map, samples_per_volume)
		
		# 验证点的有效性
		valid_points = []
		for point in points:
			z, y, x = point
			if (z >= half_size and z < vol_data.shape[0] - half_size and
					y >= half_size and y < vol_data.shape[1] - half_size and
					x >= half_size and x < vol_data.shape[2] - half_size):
				valid_points.append(point)
		
		# 保存有效点
		np.save(os.path.join(output_dir, f"{case_id}_valid_points.npy"), valid_points)
		
		# 添加到全局索引映射
		for point_idx in range(len(valid_points)):
			global_index_mapping.append((vol_idx, point_idx))
		
		print(f"案例 {case_id}: {len(valid_points)}/{len(points)} 个有效点")
	
	# 保存全局索引映射
	np.save(os.path.join(output_dir, "index_mapping.npy"), global_index_mapping)
	print(f"预处理完成，结果保存在 {output_dir}，共有 {len(global_index_mapping)} 个全局样本")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='预处理CT数据集')
	parser.add_argument('--images_dir', type=str, required=True, help='CT图像目录')
	parser.add_argument('--labels_dir', type=str, required=True, help='分割标签目录')
	parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
	parser.add_argument('--patch_size', type=int, default=64, help='补丁大小')
	parser.add_argument('--samples_per_volume', type=int, default=30, help='每个体积的采样点数')
	args = parser.parse_args()
	
	preprocess_dataset(
		args.images_dir,
		args.labels_dir,
		args.output_dir,
		patch_size=args.patch_size,
		samples_per_volume=args.samples_per_volume
	)