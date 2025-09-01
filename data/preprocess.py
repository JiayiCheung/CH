# data/preprocess.py
import os
import numpy as np
import nibabel as nib
from skimage.filters import frangi
import pickle
from tqdm import tqdm
import argparse


def preprocess_dataset(images_dir, labels_dir, output_dir, scales=range(1, 5)):
	"""预处理数据集，计算并存储Frangi响应"""
	# 创建输出目录
	os.makedirs(output_dir, exist_ok=True)
	
	# 获取所有文件路径
	volume_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
	label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.nii.gz')])
	
	print(f"找到 {len(volume_files)} 个体积文件和 {len(label_files)} 个标签文件")
	
	# 处理每个案例
	for vol_file, lab_file in tqdm(zip(volume_files, label_files), total=len(volume_files)):
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
		
		# 保存
		np.save(os.path.join(output_dir, f"{case_id}_frangi.npy"), vessel_map)
		with open(os.path.join(output_dir, f"{case_id}_metadata.pkl"), 'wb') as f:
			pickle.dump(metadata, f)
	
	print(f"预处理完成，结果保存在 {output_dir}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='预处理CT数据集')
	parser.add_argument('--images_dir', type=str, required=True, help='CT图像目录')
	parser.add_argument('--labels_dir', type=str, required=True, help='分割标签目录')
	parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
	args = parser.parse_args()
	
	preprocess_dataset(args.images_dir, args.labels_dir, args.output_dir)