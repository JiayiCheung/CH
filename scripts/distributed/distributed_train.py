#distributed_train.py

"""
分布式训练脚本 - 重构版
支持两节点多GPU训练，删除重复组件，简化代码结构
"""
import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data import LiverVesselDataset
from data.sampling_manager import SamplingManager
from models import VesselSegmenter
from losses import VesselSegmentationLoss
from utils import Logger, SegmentationMetrics


def parse_args():
	"""解析命令行参数"""
	parser = argparse.ArgumentParser(description='Distributed Training for Liver Vessel Segmentation')
	
	# 数据参数
	parser.add_argument('--image_dir', type=str, required=True, help='Path to image directory')
	parser.add_argument('--label_dir', type=str, required=True, help='Path to label directory')
	parser.add_argument('--output_dir', type=str, default='./output', help='Path to output directory')
	parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
	
	# 训练参数
	parser.add_argument('--batch_size', type=int, default=1, help='Batch size per GPU')
	parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
	parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
	parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
	parser.add_argument('--resume', type=str, help='Path to checkpoint')
	
	# 分布式参数
	parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
	parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
	parser.add_argument('--val_interval', type=int, default=5, help='Validation interval')
	
	return parser.parse_args()


def setup_distributed():
	"""初始化分布式训练环境"""
	dist.init_process_group(backend='nccl')
	torch.cuda.set_device(int(os.environ['LOCAL_RANK']))


def cleanup_distributed():
	"""清理分布式训练环境"""
	dist.destroy_process_group()


def create_model_and_loss(config, device):
	"""创建模型和损失函数"""
	# 创建模型
	model = VesselSegmenter(
		in_channels=3,
		out_channels=1,
		ch_params=config.get('ch_params'),
		tier_params=config.get('tier_params')
	)
	
	# 创建损失函数
	loss_config = config.get('loss', {})
	criterion = VesselSegmentationLoss(
		num_classes=1,
		vessel_weight=loss_config.get('vessel_weight', 10.0),
		tumor_weight=loss_config.get('tumor_weight', 15.0),
		use_boundary=loss_config.get('use_boundary', True)
	)
	
	return model.to(device), criterion.to(device)


def create_optimizer(model, config, lr):
	"""创建优化器"""
	optimizer_config = config.get('optimizer', {})
	
	# 分组参数：边缘增强核使用不同学习率
	edge_kernels = []
	other_params = []
	
	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		if 'kernels' in name:
			edge_kernels.append(param)
		else:
			other_params.append(param)
	
	param_groups = [
		{
			'params': edge_kernels,
			'lr': lr * optimizer_config.get('kernel_lr_scale', 0.1),
			'weight_decay': 0.0
		},
		{
			'params': other_params,
			'lr': lr,
			'weight_decay': optimizer_config.get('weight_decay', 1e-5)
		}
	]
	
	return optim.AdamW(param_groups)


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, args, logger):
	"""训练一个epoch"""
	model.train()
	running_loss = 0.0
	num_batches = len(dataloader)
	
	# 只在主进程显示进度条
	if dist.get_rank() == 0:
		pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
	else:
		pbar = dataloader
	
	for batch_idx, batch in enumerate(pbar):
		# 获取数据
		images = batch['image'].to(device, memory_format=torch.channels_last)
		labels = batch['label'].to(device)
		tiers = batch['tier']
		
		optimizer.zero_grad(set_to_none=True)
		
		# 处理不同tier的样本
		total_loss = 0.0
		for i, tier in enumerate(tiers):
			model.set_tier(int(tier))
			
			with autocast(enabled=args.amp):
				output = model(images[i:i + 1])
				loss = criterion(output, labels[i:i + 1])
				total_loss += loss
		
		# 平均损失
		total_loss = total_loss / len(tiers)
		
		# 反向传播
		scaler.scale(total_loss).backward()
		scaler.step(optimizer)
		scaler.update()
		
		running_loss += total_loss.item()
		
		# 更新进度条
		if dist.get_rank() == 0 and isinstance(pbar, tqdm):
			pbar.set_postfix(loss=total_loss.item())
	
	avg_loss = running_loss / num_batches
	return avg_loss


def validate(model, dataloader, device, args):
	"""验证模型"""
	model.eval()
	metrics_calc = SegmentationMetrics()
	all_metrics = []
	
	with torch.no_grad():
		for batch in dataloader:
			images = batch['image'].to(device, memory_format=torch.channels_last)
			labels = batch['label'].to(device)
			tiers = batch['tier']
			
			for i, tier in enumerate(tiers):
				model.set_tier(int(tier))
				
				with autocast(enabled=args.amp):
					output = model(images[i:i + 1])
					pred = (output > 0.5).float()
				
				# 计算指标
				batch_metrics = metrics_calc.evaluate_all(
					pred.cpu().numpy(),
					labels[i:i + 1].cpu().numpy()
				)
				all_metrics.append(batch_metrics)
	
	# 计算平均指标
	if all_metrics:
		avg_metrics = {}
		for key in all_metrics[0].keys():
			values = [m[key] for m in all_metrics if not torch.isnan(torch.tensor(m[key]))]
			avg_metrics[key] = sum(values) / len(values) if values else 0.0
		return avg_metrics
	
	return {'dice': 0.0, 'iou': 0.0}


def main():
	"""主函数"""
	args = parse_args()
	
	# 设置分布式环境
	setup_distributed()
	
	local_rank = int(os.environ['LOCAL_RANK'])
	world_size = int(os.environ['WORLD_SIZE'])
	device = torch.device(f'cuda:{local_rank}')
	
	# 创建输出目录和日志
	if local_rank == 0:
		os.makedirs(args.output_dir, exist_ok=True)
		logger = Logger(Path(args.output_dir) / 'logs')
		logger.log_info(f"Starting distributed training on {world_size} GPUs")
	else:
		logger = None
	
	# 加载配置
	with open(args.config, 'r') as f:
		config = yaml.safe_load(f)
	
	# 设置随机种子
	torch.manual_seed(42)
	torch.cuda.manual_seed_all(42)
	
	# 创建采样管理器（仅主进程）
	sampling_manager = None
	if local_rank == 0 and config.get('smart_sampling', {}).get('enabled', False):
		sampling_manager = SamplingManager(
			config.get('smart_sampling', {}),
			logger=logger
		)
		logger.log_info("Smart sampling enabled")
	
	# 创建数据集
	train_dataset = LiverVesselDataset(
		args.image_dir,
		args.label_dir,
		preprocess=True,
		max_cases=config.get('max_cases'),
		enable_smart_sampling=config.get('smart_sampling', {}).get('enabled', False),
		logger=logger if local_rank == 0 else None
	)
	
	val_dataset = LiverVesselDataset(
		args.image_dir,
		args.label_dir,
		preprocess=True,
		max_cases=config.get('max_val_cases', 5),
		random_sampling=False,
		enable_smart_sampling=False,
		logger=logger if local_rank == 0 else None
	)
	
	# 创建分布式采样器
	train_sampler = DistributedSampler(train_dataset, shuffle=True)
	val_sampler = DistributedSampler(val_dataset, shuffle=False)
	
	# 创建数据加载器
	train_loader = DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		sampler=train_sampler,
		num_workers=args.num_workers,
		pin_memory=True,
		drop_last=True
	)
	
	val_loader = DataLoader(
		val_dataset,
		batch_size=1,
		sampler=val_sampler,
		num_workers=args.num_workers,
		pin_memory=True
	)
	
	# 创建模型和损失函数
	model, criterion = create_model_and_loss(config, device)
	
	# 包装为分布式模型
	model = nn.parallel.DistributedDataParallel(
		model,
		device_ids=[local_rank],
		output_device=local_rank,
		find_unused_parameters=True
	)
	
	# 创建优化器和调度器
	optimizer = create_optimizer(model, config, args.lr)
	scaler = GradScaler(enabled=args.amp)
	
	# 恢复检查点
	start_epoch = 0
	best_dice = 0.0
	
	if args.resume and os.path.isfile(args.resume):
		if local_rank == 0:
			logger.log_info(f"Loading checkpoint: {args.resume}")
		
		checkpoint = torch.load(args.resume, map_location=device)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		scaler.load_state_dict(checkpoint['scaler'])
		start_epoch = checkpoint['epoch'] + 1
		best_dice = checkpoint.get('best_dice', 0.0)
	
	# 训练循环
	for epoch in range(start_epoch, args.epochs):
		train_sampler.set_epoch(epoch)
		
		# 更新采样策略
		if sampling_manager and sampling_manager.should_update(epoch):
			try:
				sampling_manager.update_sampling_strategy(
					model.module, train_dataset, epoch, device
				)
			except Exception as e:
				if logger:
					logger.log_warning(f"Sampling update failed: {e}")
		
		# 训练
		train_loss = train_one_epoch(
			model, train_loader, criterion, optimizer, scaler, device, epoch, args, logger
		)
		
		# 记录训练损失
		if local_rank == 0:
			logger.log_info(f"Epoch {epoch}: Loss = {train_loss:.4f}")
		
		# 验证
		if epoch % args.val_interval == 0 or epoch == args.epochs - 1:
			val_metrics = validate(model, val_loader, device, args)
			
			if local_rank == 0:
				dice_score = val_metrics.get('dice', 0.0)
				iou_score = val_metrics.get('iou', 0.0)
				logger.log_info(f"Validation - Dice: {dice_score:.4f}, IoU: {iou_score:.4f}")
				
				# 保存最佳模型
				if dice_score > best_dice:
					best_dice = dice_score
					
					torch.save({
						'epoch': epoch,
						'model_state_dict': model.state_dict(),
						'optimizer': optimizer.state_dict(),
						'scaler': scaler.state_dict(),
						'best_dice': best_dice,
						'config': config
					}, Path(args.output_dir) / 'best_model.pt')
					
					logger.log_info(f"New best model saved: Dice = {best_dice:.4f}")
		
		# 保存定期检查点
		if local_rank == 0 and epoch % 10 == 0:
			torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'scaler': scaler.state_dict(),
				'best_dice': best_dice,
				'config': config
			}, Path(args.output_dir) / f'checkpoint_epoch_{epoch}.pt')
	
	# 训练完成
	if local_rank == 0:
		logger.log_info(f"Training completed. Best Dice: {best_dice:.4f}")
	
	cleanup_distributed()


if __name__ == '__main__':
	main()