import os
import torch
import numpy as np
import time
import nibabel as nib
import yaml
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 导入自定义模块
from models.vessel_segmenter import VesselSegmenter
from data.dataset import VesselSegDataset
from data.transforms import get_training_transforms, get_validation_transforms
from visualization.visualization import save_patch_visualization


def parse_args():
	"""解析命令行参数"""
	parser = argparse.ArgumentParser(description='Vessel Segmentation Training with DDP')
	parser.add_argument('--local_rank', type=int, default=-1,
	                    help='Local rank for distributed training')
	parser.add_argument('--config', type=str, default='config.yaml',
	                    help='Path to config file')
	return parser.parse_args()


def load_config(config_path='config.yaml'):
	"""直接从YAML文件加载配置"""
	with open(config_path, 'r', encoding='utf-8') as f:
		config = yaml.safe_load(f)
	return config





def load_data(config, is_distributed=False, rank=0):
	"""加载训练和验证数据"""
	# 使用nnU-Net数据集路径
	images_dir = config['data']['data_dir']
	labels_dir = config['data']['label_dir']
	
	if rank == 0:
	# 读取所有体积文件
		volume_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
		label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.nii.gz')])
	
	if rank == 0:
		print(f"找到 {len(volume_files)} 个体积文件和 {len(label_files)} 个标签文件")
	
	# 加载体积和标签
	volumes = []
	labels = []
	for vol_file, lab_file in zip(volume_files, label_files):
		if rank == 0:
			vol_path = os.path.join(images_dir, vol_file)
			lab_path = os.path.join(labels_dir, lab_file)
		
		# 使用nibabel加载NIfTI格式数据
		vol_nii = nib.load(vol_path)
		lab_nii = nib.load(lab_path)
		
		vol_data = vol_nii.get_fdata().astype(np.float32)
		lab_data = lab_nii.get_fdata().astype(np.float32)
		
		volumes.append(vol_data)
		labels.append(lab_data)
	
	# 划分训练集和验证集
	num_samples = len(volumes)
	np.random.seed(config['data']['random_seed'])  # 固定随机种子，确保可重现性
	indices = np.random.permutation(num_samples)
	split_idx = int(num_samples * config['data']['train_val_split'])  # 默认80%用于训练
	
	train_indices = indices[:split_idx]
	val_indices = indices[split_idx:]
	
	train_volumes = [volumes[i] for i in train_indices]
	train_labels = [labels[i] for i in train_indices]
	val_volumes = [volumes[i] for i in val_indices]
	val_labels = [labels[i] for i in val_indices]
	
	if rank == 0:
		print(f"训练集: {len(train_volumes)} 个样本, 验证集: {len(val_volumes)} 个样本")
	
	# 创建数据集
	train_dataset = VesselSegDataset(
		volumes=train_volumes,
		labels=train_labels,
		patch_size=config['data']['patch_size'],
		samples_per_volume=config['data']['samples_per_volume'],
		transform=get_training_transforms(config['aug'])
	)
	
	val_dataset = VesselSegDataset(
		volumes=val_volumes,
		labels=val_labels,
		patch_size=config['data']['patch_size'],
		samples_per_volume=config['data']['samples_per_volume'],
		transform=get_validation_transforms()
	)
	
	# 创建分布式采样器
	train_sampler = DistributedSampler(train_dataset) if is_distributed else None
	val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
	
	# 创建数据加载器
	train_loader = DataLoader(
		train_dataset,
		batch_size=config['train']['batch_size'],
		shuffle=(train_sampler is None),  # 如果使用sampler则不能shuffle
		sampler=train_sampler,
		num_workers=4,
		pin_memory=True
	)
	
	val_loader = DataLoader(
		val_dataset,
		batch_size=config['train']['batch_size'],
		shuffle=False,
		sampler=val_sampler,
		num_workers=4,
		pin_memory=True
	)
	
	return train_loader, val_loader, train_sampler, val_sampler


def dice_coefficient(pred, target):
	"""计算Dice系数"""
	smooth = 1e-5
	intersection = (pred * target).sum()
	return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def sensitivity_score(pred, target):
	"""计算敏感度/召回率"""
	smooth = 1e-5
	true_positives = (pred * target).sum()
	return (true_positives + smooth) / (target.sum() + smooth)


def precision_score(pred, target):
	"""计算精确率"""
	smooth = 1e-5
	true_positives = (pred * target).sum()
	return (true_positives + smooth) / (pred.sum() + smooth)


def validate(model, val_loader, device, epoch, vis_dir, rank=0):
	"""验证函数"""
	model.eval()
	dice_scores = []
	sensitivity_scores = []
	precision_scores = []
	
	# 可视化间隔
	vis_samples = 5
	vis_interval = max(1, len(val_loader) // vis_samples)
	
	with torch.no_grad():
		for i, (inputs, labels) in enumerate(
				tqdm(val_loader, desc=f"Validating Epoch {epoch}") if rank == 0 else val_loader):
			inputs = inputs.to(device)
			labels = labels.to(device)
			
			# 前向传播
			outputs = model(inputs)
			predictions = (outputs > 0.5).float()
			
			# 计算指标
			dice = dice_coefficient(predictions, labels)
			sensitivity = sensitivity_score(predictions, labels)
			precision = precision_score(predictions, labels)
			
			# 收集指标
			dice_scores.append(dice.item())
			sensitivity_scores.append(sensitivity.item())
			precision_scores.append(precision.item())
			
			# 选择部分样本进行可视化
			if rank == 0 and i % vis_interval == 0:
				save_patch_visualization(
					inputs=inputs,
					predictions=predictions,
					labels=labels,
					epoch=epoch,
					batch_idx=i,
					output_dir=vis_dir
				)
	
	# 如果是分布式训练，需要收集所有进程的指标
	if dist.is_available() and dist.is_initialized():
		# 收集所有进程的结果
		dice_tensor = torch.tensor(np.mean(dice_scores), device=device)
		sensitivity_tensor = torch.tensor(np.mean(sensitivity_scores), device=device)
		precision_tensor = torch.tensor(np.mean(precision_scores), device=device)
		
		# 聚合结果
		dist.all_reduce(dice_tensor)
		dist.all_reduce(sensitivity_tensor)
		dist.all_reduce(precision_tensor)
		
		# 计算平均值
		world_size = dist.get_world_size()
		dice_tensor /= world_size
		sensitivity_tensor /= world_size
		precision_tensor /= world_size
		
		return {
			"dice": dice_tensor.item(),
			"sensitivity": sensitivity_tensor.item(),
			"precision": precision_tensor.item()
		}
	else:
		# 单进程训练，直接返回平均指标
		return {
			"dice": np.mean(dice_scores),
			"sensitivity": np.mean(sensitivity_scores),
			"precision": np.mean(precision_scores)
		}


def train(config, local_rank=-1):
	"""训练函数，支持分布式训练"""
	# 分布式初始化
	is_distributed = local_rank != -1
	if is_distributed:
		torch.cuda.set_device(local_rank)
		dist.init_process_group(backend='nccl')
		world_size = dist.get_world_size()
		rank = dist.get_rank()
	else:
		rank = 0
		world_size = 1
	
	# 设置设备
	device = torch.device("cuda", local_rank) if is_distributed else torch.device(
		"cuda" if torch.cuda.is_available() else "cpu")
	
	if rank == 0:
		print(f"使用设备: {device}, 世界大小: {world_size}")
	
	# 创建输出目录
	output_dir = config['train']['output_dir']
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	save_dir = os.path.join(output_dir, f'model_{timestamp}')
	vis_dir = os.path.join(save_dir, 'visualizations')
	
	if rank == 0:
		os.makedirs(save_dir, exist_ok=True)
		os.makedirs(vis_dir, exist_ok=True)
		print(f"输出目录: {save_dir}")
		


	# 加载数据
	train_loader, val_loader, train_sampler, val_sampler = load_data(config, is_distributed, rank)
	
	# 初始化模型
	model = VesselSegmenter(
		in_channels=config['model']['input_channels'],
		out_channels=config['model']['output_classes']
	)
	model.to(device)
	
	# 如果是分布式训练，包装模型
	if is_distributed:
		model = DDP(model, device_ids=[local_rank], output_device=local_rank)
	
	# 定义损失函数和优化器
	criterion = torch.nn.BCELoss()
	optimizer = torch.optim.Adam(
		model.parameters(),
		lr=config['train']['lr'],
		weight_decay=config['train']['weight_decay']
	)
	
	# 训练跟踪
	best_dice = 0.0
	start_time = time.time()
	
	# 训练循环
	if rank == 0:
		print(f"开始训练，总轮数: {config['train']['epochs']}")
	
	for epoch in range(config['train']['epochs']):
		# 设置sampler的epoch
		if is_distributed:
			train_sampler.set_epoch(epoch)
		
		# 训练阶段
		model.train()
		epoch_loss = 0.0
		
		progress_bar = tqdm(train_loader,
		                    desc=f"Epoch {epoch + 1}/{config['train']['epochs']}") if rank == 0 else train_loader
		for batch_idx, (inputs, labels) in enumerate(progress_bar):
			inputs, labels = inputs.to(device), labels.to(device)
			
			# 前向传播
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			
			# 反向传播和优化
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			# 更新进度条
			epoch_loss += loss.item()
			if rank == 0 and isinstance(progress_bar, tqdm):
				progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
		
		# 计算平均损失
		if rank == 0:
			avg_loss = epoch_loss / len(train_loader)
			print(f"Epoch {epoch + 1}/{config['train']['epochs']} | 平均损失: {avg_loss:.4f}")
		
		# 验证阶段 - 根据配置的频率执行
		if (epoch + 1) % config['train']['validate_every'] == 0:
			if rank == 0:
				print(f"验证Epoch {epoch + 1}...")
			
			# 同步所有进程进行验证
			if is_distributed:
				dist.barrier()
			
			val_metrics = validate(model, val_loader, device, epoch + 1, vis_dir, rank)
			
			if rank == 0:
				print(f"验证指标 - Dice: {val_metrics['dice']:.4f}, "
				      f"Sensitivity: {val_metrics['sensitivity']:.4f}, "
				      f"Precision: {val_metrics['precision']:.4f}")
				
				# 保存最佳模型
				if val_metrics['dice'] > best_dice:
					best_dice = val_metrics['dice']
					# 在分布式训练中，保存模型的state_dict
					if is_distributed:
						torch.save(model.module.state_dict(), os.path.join(save_dir, "best_model.pth"))
					else:
						torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
					print(f"保存最佳模型，Dice: {best_dice:.4f}")
		
		# 定期保存检查点
		if rank == 0 and (epoch + 1) % config['train']['save_every'] == 0:
			if is_distributed:
				torch.save(
					model.module.state_dict(),
					os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth")
				)
			else:
				torch.save(
					model.state_dict(),
					os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth")
				)
	
	# 保存最终模型
	if rank == 0:
		if is_distributed:
			torch.save(model.module.state_dict(), os.path.join(save_dir, "final_model.pth"))
		else:
			torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
		
		# 训练完成，打印总时间
		total_time = time.time() - start_time
		print(f"训练完成! 总时间: {total_time // 3600}h {(total_time % 3600) // 60}m {total_time % 60:.2f}s")
		print(f"最佳验证Dice: {best_dice:.4f}")
		print(f"模型保存在: {save_dir}")
	
	# 清理分布式环境
	if is_distributed:
		dist.destroy_process_group()


if __name__ == "__main__":
	# 解析命令行参数
	args = parse_args()
	
	# 加载配置
	config = load_config(args.config)
	
	# 启动训练
	train(config, args.local_rank)