import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import LiverVesselDataset
from models import VesselSegmenter
from losses import VesselSegmentationLoss
from utils import SegmentationMetrics, Logger, Visualizer


def parse_args():
	"""解析命令行参数"""
	parser = argparse.ArgumentParser(description='Train Liver Vessel Segmentation Model')
	
	# 数据参数
	parser.add_argument('--image_dir', type=str, required=True, help='Path to image directory')
	parser.add_argument('--label_dir', type=str, required=True, help='Path to label directory')
	parser.add_argument('--output_dir', type=str, default='./output', help='Path to output directory')
	parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
	
	# 训练参数
	parser.add_argument('--tier', type=int, default=None, help='Train specific tier (0, 1, 2)')
	parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
	parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
	parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
	parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
	parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint for resuming training')
	
	# 其他参数
	parser.add_argument('--seed', type=int, default=42, help='Random seed')
	parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
	parser.add_argument('--val_interval', type=int, default=5, help='Validation interval (epochs)')
	
	return parser.parse_args()


def load_config(config_path):
	"""加载配置文件"""
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)
	return config


def set_seed(seed):
	"""设置随机种子"""
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, optimizer, device, logger, epoch):
	"""训练一个epoch"""
	model.train()
	epoch_loss = 0
	
	# 进度条
	progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")
	
	for i, batch in progress_bar:
		# 获取数据
		images = batch['image'].to(device)
		labels = batch['label'].to(device)
		tiers = batch['tier']
		
		# 清除梯度
		optimizer.zero_grad()
		
		# 对每个样本设置对应的tier
		outputs = []
		for j, tier in enumerate(tiers):
			# 设置当前tier
			model.set_tier(tier.item())
			
			# 前向传播
			output = model(images[j:j + 1])
			outputs.append(output)
		
		# 合并输出
		outputs = torch.cat(outputs, dim=0)
		
		# 计算损失
		loss = criterion(outputs, labels)
		
		# 反向传播
		loss.backward()
		
		# 更新参数
		optimizer.step()
		
		# 记录损失
		epoch_loss += loss.item()
		
		# 更新进度条
		progress_bar.set_postfix({'loss': loss.item()})
	
	# 计算平均损失
	epoch_loss /= len(dataloader)
	
	# 记录到日志
	logger.log_info(f"Epoch {epoch} - Training Loss: {epoch_loss:.4f}")
	logger.log_metrics({'loss': epoch_loss}, epoch, prefix='train/')
	
	return epoch_loss


def validate(model, dataloader, criterion, device, logger, epoch, visualizer=None):
	"""验证模型"""
	model.eval()
	val_loss = 0
	all_metrics = {}
	
	# 进度条
	progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation")
	
	with torch.no_grad():
		for i, batch in progress_bar:
			# 获取数据
			images = batch['image'].to(device)
			labels = batch['label'].to(device)
			tiers = batch['tier']
			case_ids = batch['case_id']
			
			# 对每个样本设置对应的tier
			outputs = []
			for j, tier in enumerate(tiers):
				# 设置当前tier
				model.set_tier(tier.item())
				
				# 前向传播
				output = model(images[j:j + 1])
				outputs.append(output)
			
			# 合并输出
			outputs = torch.cat(outputs, dim=0)
			
			# 计算损失
			loss = criterion(outputs, labels)
			val_loss += loss.item()
			
			# 计算评估指标
			preds = (outputs > 0.5).float()
			
			for j, (pred, label, tier, case_id) in enumerate(zip(preds, labels, tiers, case_ids)):
				# 计算指标
				metrics = SegmentationMetrics.evaluate_all(pred.cpu(), label.cpu())
				
				# 记录指标
				tier_str = f"tier_{tier.item()}"
				if tier_str not in all_metrics:
					all_metrics[tier_str] = []
				
				all_metrics[tier_str].append(metrics)
				
				# 可视化 (仅前几个样本)
				if visualizer and i < 5:
					# 创建保存路径
					save_dir = Path(logger.experiment_dir) / "visualizations" / f"epoch_{epoch}"
					save_dir.mkdir(exist_ok=True, parents=True)
					
					# 可视化分割结果
					visualizer.visualize_segmentation(
						images[j].cpu(),
						pred.cpu(),
						label.cpu(),
						title=f"Case {case_id} (Tier {tier.item()})",
						save_path=save_dir / f"case_{case_id}_tier_{tier.item()}.png"
					)
	
	# 计算平均损失
	val_loss /= len(dataloader)
	
	# 计算每个tier的平均指标
	avg_metrics = {}
	for tier_str, metrics_list in all_metrics.items():
		avg_tier_metrics = {}
		for metric in metrics_list[0].keys():
			avg_tier_metrics[metric] = np.mean([m[metric] for m in metrics_list])
		
		avg_metrics[tier_str] = avg_tier_metrics
	
	# 计算总体平均指标
	overall_metrics = {}
	for metric in list(avg_metrics.values())[0].keys():
		overall_metrics[metric] = np.mean([tier_metrics[metric] for tier_metrics in avg_metrics.values()])
	
	# 记录到日志
	logger.log_info(f"Epoch {epoch} - Validation Loss: {val_loss:.4f}")
	logger.log_metrics({'loss': val_loss}, epoch, prefix='val/')
	
	# 记录整体指标
	for metric, value in overall_metrics.items():
		logger.log_metrics({metric: value}, epoch, prefix='val/')
	
	# 记录每个tier的指标
	for tier_str, tier_metrics in avg_metrics.items():
		for metric, value in tier_metrics.items():
			logger.log_metrics({f"{tier_str}/{metric}": value}, epoch, prefix='val/')
	
	return val_loss, overall_metrics['dice']


def main():
	"""主函数"""
	# 解析参数
	args = parse_args()
	
	# 加载配置
	config = load_config(args.config)
	
	# 设置随机种子
	set_seed(args.seed)
	
	# 创建输出目录
	output_dir = Path(args.output_dir)
	output_dir.mkdir(exist_ok=True, parents=True)
	
	# 设置设备
	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
	
	# 初始化日志记录器
	logger = Logger(output_dir / 'logs', experiment_name=f"tier_{args.tier}" if args.tier is not None else "all_tiers")
	logger.log_info(f"Starting training with config: {config}")
	logger.log_info(f"Using device: {device}")
	
	# 初始化可视化器
	visualizer = Visualizer(output_dir / 'visualizations')
	
	# 创建数据集
	train_dataset = LiverVesselDataset(
		args.image_dir,
		args.label_dir,
		tier=args.tier,
		transform=None,  # 可以添加数据增强
		preprocess=True,
		max_cases=config.get('max_cases', None),
		random_sampling=config.get('random_sampling', True)
	)
	
	val_dataset = LiverVesselDataset(
		args.image_dir,
		args.label_dir,
		tier=args.tier,
		transform=None,
		preprocess=True,
		max_cases=config.get('max_val_cases', None),
		random_sampling=False  # 验证集不随机采样
	)
	
	# 创建数据加载器
	train_loader = DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=True
	)
	
	val_loader = DataLoader(
		val_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=True
	)
	
	logger.log_info(f"Training samples: {len(train_dataset)}")
	logger.log_info(f"Validation samples: {len(val_dataset)}")
	
	# 创建模型
	model = VesselSegmenter(
		in_channels=1,
		out_channels=1,  # 二分类
		ch_params=config.get('ch_params', None),
		tier_params=config.get('tier_params', None)
	)
	model.to(device)
	
	# 创建损失函数
	criterion = VesselSegmentationLoss(
		num_classes=1,
		vessel_weight=config.get('vessel_weight', 10.0),
		use_boundary=config.get('use_boundary', True)
	)
	
	# 创建优化器
	optimizer = optim.Adam(
		model.parameters(),
		lr=args.lr,
		weight_decay=config.get('weight_decay', 1e-5)
	)
	
	# 创建学习率调度器
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode='max',
		factor=0.5,
		patience=10,
		verbose=True
	)
	
	# 恢复训练 (如果需要)
	start_epoch = 0
	best_dice = 0
	
	if args.resume:
		checkpoint = torch.load(args.resume)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		start_epoch = checkpoint['epoch'] + 1
		best_dice = checkpoint['best_dice']
		logger.log_info(f"Resuming from epoch {start_epoch}, best dice: {best_dice:.4f}")
	
	# 训练循环
	for epoch in range(start_epoch, args.epochs):
		# 训练一个epoch
		train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, logger, epoch)
		
		# 验证
		if (epoch + 1) % args.val_interval == 0:
			val_loss, val_dice = validate(model, val_loader, criterion, device, logger, epoch, visualizer)
			
			# 更新学习率
			scheduler.step(val_dice)
			
			# 保存最佳模型
			if val_dice > best_dice:
				best_dice = val_dice
				
				# 保存模型
				checkpoint = {
					'epoch': epoch,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'scheduler_state_dict': scheduler.state_dict(),
					'best_dice': best_dice
				}
				
				torch.save(checkpoint, output_dir / 'best_model.pt')
				logger.log_info(f"Saved best model with dice: {best_dice:.4f}")
		
		# 保存最新模型
		checkpoint = {
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'scheduler_state_dict': scheduler.state_dict(),
			'best_dice': best_dice
		}
		
		torch.save(checkpoint, output_dir / 'latest_model.pt')
	
	# 训练结束
	logger.log_info(f"Training completed. Best dice: {best_dice:.4f}")
	logger.close()


if __name__ == '__main__':
	main()