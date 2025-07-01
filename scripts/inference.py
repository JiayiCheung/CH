import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import nibabel as nib

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import CTPreprocessor, TierSampler
from models import VesselSegmenter
from utils import Visualizer


def parse_args():
	"""解析命令行参数"""
	parser = argparse.ArgumentParser(description='Inference with Liver Vessel Segmentation Model')
	
	# 数据参数
	parser.add_argument('--input', type=str, required=True, help='Path to input volume or directory')
	parser.add_argument('--output', type=str, default='./predictions', help='Path to output directory')
	parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
	parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
	
	# 推理参数
	parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
	parser.add_argument('--threshold', type=float, default=0.5, help='Segmentation threshold')
	parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
	parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
	
	return parser.parse_args()


def load_config(config_path):
	"""加载配置文件"""
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)
	return config


def load_volume(file_path):
	"""加载3D体积"""
	# 支持nii.gz格式
	if file_path.suffix == '.gz' or file_path.suffix == '.nii':
		nib_img = nib.load(str(file_path))
		volume = nib_img.get_fdata()
		affine = nib_img.affine
	else:
		raise ValueError(f"Unsupported file format: {file_path.suffix}")
	
	return volume, affine


def save_volume(volume, affine, output_path):
	"""保存3D体积"""
	# 创建NIfTI图像
	nib_img = nib.Nifti1Image(volume, affine)
	
	# 保存
	nib.save(nib_img, str(output_path))


def sliding_window_inference(model, volume, tier_params, window_size=64, overlap=0.5, batch_size=1, device='cuda'):
	"""
	使用滑动窗口进行推理

	参数:
		model: 模型
		volume: 输入体积 [D, H, W]
		tier_params: tier参数
		window_size: 窗口大小
		overlap: 重叠比例
		batch_size: 批处理大小
		device: 设备

	返回:
		预测体积 [D, H, W]
	"""
	# 获取体积形状
	D, H, W = volume.shape
	
	# 计算步长
	stride = int(window_size * (1 - overlap))
	
	# 计算每个维度的窗口数量
	n_d = (D - window_size) // stride + 2
	n_h = (H - window_size) // stride + 2
	n_w = (W - window_size) // stride + 2
	
	# 确保最后一个窗口覆盖到边界
	d_starts = [min(D - window_size, stride * i) for i in range(n_d)]
	h_starts = [min(H - window_size, stride * i) for i in range(n_h)]
	w_starts = [min(W - window_size, stride * i) for i in range(n_w)]
	
	# 创建输出体积和权重体积 (用于加权平均)
	output = np.zeros((D, H, W), dtype=np.float32)
	weight = np.zeros((D, H, W), dtype=np.float32)
	
	# 创建高斯权重 (中心权重高，边缘权重低)
	def gaussian_weight(window_size):
		x, y, z = np.meshgrid(
			np.linspace(-1, 1, window_size),
			np.linspace(-1, 1, window_size),
			np.linspace(-1, 1, window_size),
			indexing='ij'
		)
		
		# 计算距离中心的距离
		d = np.sqrt(x * x + y * y + z * z)
		
		# 高斯权重
		sigma = 0.5
		w = np.exp(-(d ** 2) / (2 * sigma ** 2))
		
		return w
	
	# 创建窗口权重
	window_weight = gaussian_weight(window_size)
	
	# 设置模型为评估模式
	model.eval()
	
	# 提取窗口，批量处理
	windows = []
	window_positions = []
	
	for d_start in d_starts:
		for h_start in h_starts:
			for w_start in w_starts:
				# 提取窗口
				d_end = d_start + window_size
				h_end = h_start + window_size
				w_end = w_start + window_size
				
				window = volume[d_start:d_end, h_start:h_end, w_start:w_end]
				
				# 如果窗口大小不一致，跳过
				if window.shape != (window_size, window_size, window_size):
					continue
				
				# 添加窗口
				windows.append(window)
				window_positions.append((d_start, h_start, w_start))
				
				# 当窗口数量达到批次大小时，进行处理
				if len(windows) == batch_size:
					# 处理批次
					batch_result = process_batch(model, windows, tier_params, device)
					
					# 将结果放回输出体积
					for i, (d_s, h_s, w_s) in enumerate(window_positions):
						d_e = d_s + window_size
						h_e = h_s + window_size
						w_e = w_s + window_size
						
						output[d_s:d_e, h_s:h_e, w_s:w_e] += batch_result[i] * window_weight
						weight[d_s:d_e, h_s:h_e, w_s:w_e] += window_weight
					
					# 清空窗口列表
					windows = []
					window_positions = []
	
	# 处理剩余的窗口
	if windows:
		# 处理批次
		batch_result = process_batch(model, windows, tier_params, device)
		
		# 将结果放回输出体积
		for i, (d_s, h_s, w_s) in enumerate(window_positions):
			d_e = d_s + window_size
			h_e = h_s + window_size
			w_e = w_s + window_size
			
			output[d_s:d_e, h_s:h_e, w_s:w_e] += batch_result[i] * window_weight
			weight[d_s:d_e, h_s:h_e, w_s:w_e] += window_weight
	
	# 计算加权平均
	weight = np.maximum(weight, 1e-10)  # 避免除零错误
	output = output / weight
	
	return output


def process_batch(model, windows, tier_params, device):
	"""
	处理一个批次的窗口

	参数:
		model: 模型
		windows: 窗口列表
		tier_params: tier参数
		device: 设备

	返回:
		批次预测结果
	"""
	# 转换为张量
	batch = np.stack(windows, axis=0)
	batch = torch.from_numpy(batch).float().unsqueeze(1).to(device)  # [B, 1, D, H, W]
	
	# 在每个tier上运行模型
	with torch.no_grad():
		# 初始化多tier预测
		tier_predictions = {}
		
		for tier in tier_params.keys():
			# 设置当前tier
			model.set_tier(tier)
			
			# 前向传播
			pred = model(batch)
			
			# 存储预测结果
			tier_predictions[tier] = pred
		
		# 加权平均不同tier的预测
		if len(tier_predictions) > 1:
			# 合并预测 (简单平均)
			pred = sum(tier_predictions.values()) / len(tier_predictions)
		else:
			# 只有一个tier时，直接使用
			pred = list(tier_predictions.values())[0]
		
		# 应用sigmoid
		pred = torch.sigmoid(pred)
	
	# 转换为numpy
	pred = pred.cpu().numpy()
	
	return pred


def multitier_inference(model, volume, tier_sampler, config, device='cuda'):
	"""
	使用多tier策略进行推理

	参数:
		model: 模型
		volume: 输入体积
		tier_sampler: tier采样器
		config: 配置
		device: 设备

	返回:
		预测体积
	"""
	# 创建肝脏掩码 (如果有更好的肝脏分割方法，可以替换)
	liver_mask = (volume > np.percentile(volume, 99.7)).astype(np.uint8)
	
	# 应用三级采样
	patches = tier_sampler.sample(volume, None, liver_mask)
	
	# 创建输出体积 (与输入大小相同)
	output = np.zeros_like(volume)
	
	# 设置模型为评估模式
	model.eval()
	
	# 处理每个patch
	for patch in patches:
		# 获取patch信息
		patch_img = patch['image']
		tier = patch['tier']
		
		# 转换为张量
		patch_tensor = torch.from_numpy(patch_img).float().unsqueeze(0).unsqueeze(0).to(device)
		
		# 设置当前tier
		model.set_tier(tier)
		
		# 推理
		with torch.no_grad():
			pred = model(patch_tensor)
			pred = torch.sigmoid(pred)
		
		# 转换为numpy
		pred = pred.squeeze().cpu().numpy()
	
	# TODO: 将预测结果放回原始体积 (需要记录patch位置)
	# 这部分需要额外实现，以便将patch预测结果正确放回原始体积
	
	# 由于上述TODO的复杂性，这里直接使用滑动窗口推理
	# 对于完整实现，应该使用更有效的方法将patch预测结果放回原始体积
	tier_params = config.get('tier_params', {})
	output = sliding_window_inference(
		model, volume, tier_params,
		window_size=config.get('window_size', 64),
		overlap=config.get('overlap', 0.5),
		batch_size=1,
		device=device
	)
	
	return output


def main():
	"""主函数"""
	# 解析参数
	args = parse_args()
	
	# 加载配置
	config = load_config(args.config)
	
	# 创建输出目录
	output_dir = Path(args.output)
	output_dir.mkdir(exist_ok=True, parents=True)
	
	# 设置设备
	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")
	
	# 初始化预处理器和tier采样器
	preprocessor = CTPreprocessor()
	tier_sampler = TierSampler()
	
	# 初始化可视化器 (如果需要)
	visualizer = Visualizer(output_dir / 'visualizations') if args.visualize else None
	
	# 创建模型
	model = VesselSegmenter(
		in_channels=1,
		out_channels=1,  # 二分类
		ch_params=config.get('ch_params', None),
		tier_params=config.get('tier_params', None)
	)
	
	# 加载模型权重
	checkpoint = torch.load(args.model, map_location=device)
	if 'model_state_dict' in checkpoint:
		model.load_state_dict(checkpoint['model_state_dict'])
	else:
		model.load_state_dict(checkpoint)
	
	model.to(device)
	model.eval()
	
	# 确定输入路径
	input_path = Path(args.input)
	
	# 处理单个文件或目录
	if input_path.is_file():
		# 加载体积
		volume, affine = load_volume(input_path)
		
		# 预处理
		volume_norm = preprocessor.normalize(volume)
		
		# 推理
		print(f"Processing {input_path.name}...")
		prediction = multitier_inference(model, volume_norm, tier_sampler, config, device)
		
		# 二值化
		binary_pred = (prediction > args.threshold).astype(np.float32)
		
		# 保存结果
		output_path = output_dir / f"{input_path.stem}_pred.nii.gz"
		save_volume(binary_pred, affine, output_path)
		print(f"Saved prediction to {output_path}")
		
		# 可视化 (如果需要)
		if visualizer:
			# 创建保存路径
			vis_dir = output_dir / 'visualizations'
			vis_dir.mkdir(exist_ok=True)
			
			# 获取中间切片
			slice_idx = volume.shape[0] // 2
			
			# 可视化原始图像
			visualizer.visualize_slice(
				volume,
				slice_idx=slice_idx,
				title=f"Original - {input_path.name}",
				save_path=vis_dir / f"{input_path.stem}_original.png"
			)
			
			# 可视化预测结果
			visualizer.visualize_segmentation(
				volume,
				binary_pred,
				slice_idx=slice_idx,
				title=f"Prediction - {input_path.name}",
				save_path=vis_dir / f"{input_path.stem}_prediction.png"
			)
			
			# 可视化3D结果
			visualizer.visualize_3d(
				binary_pred,
				title=f"3D Prediction - {input_path.name}",
				save_path=vis_dir / f"{input_path.stem}_3d.png"
			)
	
	elif input_path.is_dir():
		# 处理目录中的所有文件
		input_files = list(input_path.glob("*.nii.gz"))
		input_files.extend(list(input_path.glob("*.nii")))
		
		for file_path in tqdm(input_files, desc="Processing files"):
			try:
				# 加载体积
				volume, affine = load_volume(file_path)
				
				# 预处理
				volume_norm = preprocessor.normalize(volume)
				
				# 推理
				prediction = multitier_inference(model, volume_norm, tier_sampler, config, device)
				
				# 二值化
				binary_pred = (prediction > args.threshold).astype(np.float32)
				
				# 保存结果
				output_path = output_dir / f"{file_path.stem}_pred.nii.gz"
				save_volume(binary_pred, affine, output_path)
				
				# 可视化 (如果需要)
				if visualizer:
					# 创建保存路径
					vis_dir = output_dir / 'visualizations'
					vis_dir.mkdir(exist_ok=True)
					
					# 获取中间切片
					slice_idx = volume.shape[0] // 2
					
					# 可视化预测结果
					visualizer.visualize_segmentation(
						volume,
						binary_pred,
						slice_idx=slice_idx,
						title=f"Prediction - {file_path.name}",
						save_path=vis_dir / f"{file_path.stem}_prediction.png"
					)
			
			except Exception as e:
				print(f"Error processing {file_path}: {e}")
	
	else:
		print(f"Invalid input path: {input_path}")


if __name__ == '__main__':
	main()