import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
import json
import matplotlib.pyplot as plt
import nibabel as nib  # 需要安装nibabel库处理医学影像
from datetime import datetime

from config import MODEL_CONFIG, DATA_CONFIG, INFERENCE_CONFIG
from models.model import VesselSegModel
from data.dataset import FrangiSampler


def parse_args():
	"""解析命令行参数"""
	parser = argparse.ArgumentParser(description='Inference for Vessel Segmentation Model')
	
	# 数据参数
	parser.add_argument('--input_path', type=str, required=True, help='输入CT体积路径')
	parser.add_argument('--output_dir', type=str, default='results', help='输出目录路径')
	parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
	
	# 配置参数
	parser.add_argument('--config', type=str, default='', help='自定义配置文件路径(JSON格式)')
	
	return parser.parse_args()


def load_custom_config(config_path):
	"""加载自定义配置"""
	if not config_path or not os.path.exists(config_path):
		return {}
	
	with open(config_path, 'r') as f:
		return json.load(f)


def update_config(base_config, custom_config):
	"""更新配置"""
	for key, value in custom_config.items():
		if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):
			update_config(base_config[key], value)
		else:
			base_config[key] = value


def load_volume(file_path):
	"""
	加载体积数据

	参数:
		file_path: 文件路径

	返回:
		volume: 体积数据
	"""
	# 支持不同的文件格式
	if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
		# 加载NIfTI格式
		nii = nib.load(file_path)
		volume = nii.get_fdata().astype(np.float32)
		return volume, nii.affine
	elif file_path.endswith('.npy'):
		# 加载NumPy格式
		volume = np.load(file_path).astype(np.float32)
		return volume, None
	else:
		raise ValueError(f"不支持的文件格式: {file_path}")


def save_segmentation(segmentation, output_path, affine=None):
	"""
	保存分割结果

	参数:
		segmentation: 分割结果
		output_path: 输出路径
		affine: NIfTI仿射矩阵
	"""
	if output_path.endswith('.nii') or output_path.endswith('.nii.gz'):
		# 保存为NIfTI格式
		if affine is None:
			affine = np.eye(4)
		nib.save(nib.Nifti1Image(segmentation.astype(np.uint8), affine), output_path)
	elif output_path.endswith('.npy'):
		# 保存为NumPy格式
		np.save(output_path, segmentation.astype(np.uint8))
	else:
		# 默认保存为NIfTI格式
		if affine is None:
			affine = np.eye(4)
		nib.save(nib.Nifti1Image(segmentation.astype(np.uint8), affine), output_path + '.nii.gz')


def create_gaussian_weight_kernel(size):
	"""
	创建高斯权重核，中心权重高，边缘权重低

	参数:
		size: 核大小

	返回:
		kernel: 高斯权重核
	"""
	kernel = np.zeros((size, size, size), dtype=np.float32)
	center = size // 2
	
	for z in range(size):
		for y in range(size):
			for x in range(size):
				# 计算到中心的距离
				dist = np.sqrt((z - center) ** 2 + (y - center) ** 2 + (x - center) ** 2)
				# 高斯权重
				kernel[z, y, x] = np.exp(-(dist ** 2) / (2 * (size / 4) ** 2))
	
	return kernel


def frangi_guided_inference(model, ct_volume, config):
	"""
	使用Frangi滤波器引导的滑动窗口推理

	参数:
		model: 模型
		ct_volume: CT体积
		config: 配置字典

	返回:
		segmentation: 分割结果
	"""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)
	model.eval()
	
	# 参数
	patch_size = config['data']['patch_size']
	overlap = config['inference']['overlap']
	batch_size = config['inference']['batch_size']
	threshold = config['inference']['threshold']
	
	# 创建Frangi采样器
	sampler = FrangiSampler(
		patch_size=patch_size,
		scales=config['data']['frangi_scales'],
		contrast_enhancement=config['data']['contrast_enhancement']
	)
	
	# 生成密度图
	print("生成血管密度图...")
	density_map = sampler.create_vessel_density_map(ct_volume)
	
	# 计算步长
	stride = int(patch_size * (1 - overlap))
	
	# 创建输出体积和权重累积器
	output_volume = np.zeros_like(ct_volume, dtype=np.float32)
	weight_volume = np.zeros_like(ct_volume, dtype=np.float32)
	
	# 创建高斯权重核(中心权重高，边缘权重低)
	weight_kernel = create_gaussian_weight_kernel(patch_size)
	
	# 计算所有可能的窗口中心
	print("计算采样点...")
	centers = []
	for z in range(patch_size // 2, ct_volume.shape[0] - patch_size // 2, stride):
		for y in range(patch_size // 2, ct_volume.shape[1] - patch_size // 2, stride):
			for x in range(patch_size // 2, ct_volume.shape[2] - patch_size // 2, stride):
				centers.append((z, y, x, density_map[z, y, x]))
	
	# 按密度值排序，高密度区域优先处理
	centers.sort(key=lambda x: x[3], reverse=True)
	
	# 批量处理窗口
	print(f"开始推理，共{len(centers)}个补丁...")
	for i in tqdm(range(0, len(centers), batch_size)):
		batch_centers = centers[i:i + batch_size]
		
		# 提取补丁
		batch_patches = []
		for z, y, x, _ in batch_centers:
			patch = ct_volume[z - patch_size // 2:z + patch_size // 2,
			        y - patch_size // 2:y + patch_size // 2,
			        x - patch_size // 2:x + patch_size // 2]
			
			# 添加通道维度
			patch = patch[np.newaxis, ...]
			batch_patches.append(patch)
		
		# 转换为tensor
		batch_tensor = torch.from_numpy(np.array(batch_patches)).float().to(device)
		
		# 预测
		with torch.no_grad():
			batch_output = model(batch_tensor).cpu().numpy()
		
		# 将预测结果放回原位置
		for j, (z, y, x, _) in enumerate(batch_centers):
			output = batch_output[j, 0]  # 移除通道维度
			
			z_start = z - patch_size // 2
			y_start = y - patch_size // 2
			x_start = x - patch_size // 2
			
			output_volume[z_start:z_start + patch_size,
			y_start:y_start + patch_size,
			x_start:x_start + patch_size] += output * weight_kernel
			
			weight_volume[z_start:z_start + patch_size,
			y_start:y_start + patch_size,
			x_start:x_start + patch_size] += weight_kernel
	
	# 计算加权平均
	output_volume = np.divide(
		output_volume,
		weight_volume,
		out=np.zeros_like(output_volume),
		where=weight_volume > 0
	)
	
	# 二值化
	segmentation = (output_volume > threshold).astype(np.uint8)
	
	return segmentation, output_volume


def visualize_results(ct_volume, segmentation, output_dir):
	"""
	可视化结果

	参数:
		ct_volume: CT体积
		segmentation: 分割结果
		output_dir: 输出目录
	"""
	# 创建输出目录
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	
	# 选择体积中间的切片
	z_mid = ct_volume.shape[0] // 2
	
	# 创建图形
	plt.figure(figsize=(12, 6))
	
	# 显示原始CT切片
	plt.subplot(1, 2, 1)
	plt.imshow(ct_volume[z_mid], cmap='gray')
	plt.title('Original CT')
	plt.axis('off')
	
	# 显示分割结果
	plt.subplot(1, 2, 2)
	plt.imshow(ct_volume[z_mid], cmap='gray')
	plt.imshow(segmentation[z_mid], cmap='hot', alpha=0.5)
	plt.title('Segmentation')
	plt.axis('off')
	
	# 保存图形
	plt.savefig(os.path.join(output_dir, 'visualization.png'))
	plt.close()


def main():
	"""主函数"""
	# 解析命令行参数
	args = parse_args()
	
	# 加载配置
	configs = {
		'model': MODEL_CONFIG,
		'data': DATA_CONFIG,
		'inference': INFERENCE_CONFIG
	}
	
	# 加载自定义配置
	custom_config = load_custom_config(args.config)
	if custom_config:
		for key, value in custom_config.items():
			if key in configs:
				update_config(configs[key], value)
	
	# 创建输出目录
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	output_dir = os.path.join(args.output_dir, f'inference_{timestamp}')
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	
	# 保存配置
	config_save_path = os.path.join(output_dir, 'config.json')
	with open(config_save_path, 'w') as f:
		json.dump(configs, f, indent=4)
	
	# 加载CT体积
	print(f"加载体积: {args.input_path}")
	ct_volume, affine = load_volume(args.input_path)
	
	# 初始化模型
	print("初始化模型...")
	model = VesselSegModel(
		input_channels=configs['model']['input_channels'],
		output_classes=configs['model']['output_classes'],
		feature_channels=configs['model']['feature_channels'],
		max_harmonic_degree=configs['model']['max_harmonic_degree']
	)
	
	# 加载模型权重
	model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
	
	# 使用Frangi引导推理
	print("开始分割...")
	segmentation, probability_map = frangi_guided_inference(model, ct_volume, configs)
	
	# 保存结果
	print("保存结果...")
	save_segmentation(segmentation, os.path.join(output_dir, 'segmentation.nii.gz'), affine)
	save_segmentation(probability_map, os.path.join(output_dir, 'probability_map.nii.gz'), affine)
	
	# 可视化结果
	print("生成可视化...")
	visualize_results(ct_volume, segmentation, output_dir)
	
	print(f"推理完成! 结果已保存到: {output_dir}")


if __name__ == "__main__":
	main()