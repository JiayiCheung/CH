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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils.sampling_scheduler import SamplingScheduler
from data.hard_sample_tracker import HardSampleTracker
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import LiverVesselDataset
from models import VesselSegmenter
from losses import VesselSegmentationLoss
from utils import SegmentationMetrics, Logger, Visualizer
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils.sampling_scheduler import SamplingScheduler
from data.hard_sample_tracker import HardSampleTracker

logger = logging.getLogger(__name__)


def parse_args():
	"""解析命令行参数"""
	parser = argparse.ArgumentParser(description='Train Liver Vessel Segmentation Model')
	
	# 数据参数 (保持不变)
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
	
	# 多GPU参数
	parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
	parser.add_argument('--world_size', type=int, default=4, help='Number of GPUs to use')
	parser.add_argument('--rank', type=int, default=0, help='Node rank for distributed training')
	parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:23456', help='URL for distributed training')
	parser.add_argument('--dist_backend', type=str, default='nccl', help='Distributed backend')
	
	# 智能采样参数
	parser.add_argument('--smart_sampling', action='store_true', help='Enable smart sampling')
	parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs for smart sampling')
	parser.add_argument('--difficulty_maps_dir', type=str, default='difficulty_maps',
	                    help='Directory for difficulty maps')
	
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



def init_distributed(args):
	"""初始化分布式训练环境"""
	if args.distributed:
		if args.dist_url == "env://" and args.rank == -1:
			args.rank = int(os.environ["RANK"])
		
		# 初始化进程组
		dist.init_process_group(
			backend=args.dist_backend,
			init_method=args.dist_url,
			world_size=args.world_size,
			rank=args.rank
		)
		
		# 设置本地rank
		args.local_rank = args.rank % torch.cuda.device_count()
		torch.cuda.set_device(args.local_rank)
		
		logger.info(f"Initialized distributed training: rank={args.rank}, world_size={args.world_size}")
	else:
		args.local_rank = 0


# 在train.py中
def init_distributed_training(args):
	# 设置分布式训练环境
	args.distributed = args.world_size > 1 or args.multiprocessing_distributed
	
	if args.distributed:
		if args.dist_url == "env://" and args.rank == -1:
			args.rank = int(os.environ["RANK"])
		if args.multiprocessing_distributed:
			args.rank = args.rank * args.ngpus_per_node + gpu
		dist.init_process_group(
			backend=args.dist_backend,
			init_method=args.dist_url,
			world_size=args.world_size,
			rank=args.rank
		)


def create_data_loader(dataset, args):
	if args.distributed:
		sampler = torch.utils.data.distributed.DistributedSampler(dataset)
	else:
		sampler = None
	
	loader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=(sampler is None),
		num_workers=args.workers,
		pin_memory=True,
		sampler=sampler
	)
	return loader, sampler


def train_one_epoch(model, dataloader, criterion, optimizer, device, logger, epoch, args):
	"""训练一个epoch"""
	model.train()
	epoch_loss = 0
	
	# 进度条 (只在主进程显示)
	if args.rank == 0:
		progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")
	else:
		progress_bar = enumerate(dataloader)
	
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
			if args.distributed:
				model.module.set_tier(tier.item())
			else:
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
		
		# 更新进度条 (只在主进程)
		if args.rank == 0:
			if isinstance(progress_bar, tqdm):
				progress_bar.set_postfix({'loss': loss.item()})
	
	# 计算平均损失
	epoch_loss /= len(dataloader)
	
	# 在分布式环境中，收集所有进程的损失
	if args.distributed:
		loss_tensor = torch.tensor(epoch_loss, device=device)
		dist.all_reduce(loss_tensor)
		epoch_loss = loss_tensor.item() / dist.get_world_size()
	
	# 记录到日志 (只在主进程)
	if args.rank == 0 and logger:
		logger.log_info(f"Epoch {epoch} - Training Loss: {epoch_loss:.4f}")
		logger.log_metrics({'loss': epoch_loss}, epoch, prefix='train/')
	
	return epoch_loss


def validate(model, dataloader, criterion, device, logger, epoch, visualizer=None, args=None):
	"""验证模型"""
	model.eval()
	val_loss = 0
	all_metrics = {}
	
	# 进度条 (只在主进程显示)
	if args.rank == 0:
		progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation")
	else:
		progress_bar = enumerate(dataloader)
	
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
				if args.distributed:
					model.module.set_tier(tier.item())
				else:
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
				
				# 可视化 (仅在主进程和前几个样本)
				if args.rank == 0 and visualizer and i < 5:
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
	
	# 在分布式环境中，收集所有进程的损失和指标
	if args.distributed:
		# 收集损失
		loss_tensor = torch.tensor(val_loss, device=device)
		dist.all_reduce(loss_tensor)
		val_loss = loss_tensor.item() / dist.get_world_size()
		
		# 收集指标
		for tier_str in list(all_metrics.keys()):
			for metric_name in all_metrics[tier_str][0].keys():
				# 收集每个tier每个指标的平均值
				metric_values = [m[metric_name] for m in all_metrics[tier_str]]
				metric_mean = np.mean(metric_values)
				
				# 转换为张量
				metric_tensor = torch.tensor(metric_mean, device=device)
				dist.all_reduce(metric_tensor)
				
				# 更新所有指标
				for m in all_metrics[tier_str]:
					m[metric_name] = metric_tensor.item() / dist.get_world_size()
	
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
	
	# 记录到日志 (只在主进程)
	if args.rank == 0 and logger:
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
	
	# 初始化分布式环境
	if args.distributed:
		init_distributed(args)
	
	# 设置设备
	device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
	
	# 初始化日志记录器
	if args.rank == 0:  # 只在主进程记录日志
		logger = Logger(output_dir / 'logs',
		                experiment_name=f"tier_{args.tier}" if args.tier is not None else "all_tiers")
		logger.log_info(f"Starting training with config: {config}")
		logger.log_info(f"Using device: {device}")
	else:
		logger = None
	
	# 初始化可视化器
	visualizer = Visualizer(output_dir / 'visualizations') if args.rank == 0 else None
	
	# 初始化采样调度器
	sampling_scheduler = None
	hard_sample_tracker = None
	
	if args.smart_sampling:
		# 创建难度图目录
		difficulty_dir = Path(args.difficulty_maps_dir)
		difficulty_dir.mkdir(exist_ok=True, parents=True)
		
		# 初始化采样调度器
		sampling_scheduler = SamplingScheduler(
			base_tier1=config.get('tier1_samples', 10),
			base_tier2=config.get('tier2_samples', 30),
			warmup_epochs=args.warmup_epochs
		)
		
		# 初始化硬样本跟踪器
		hard_sample_tracker = HardSampleTracker(
			base_dir=args.difficulty_maps_dir,
			device=device
		)
	
	# 创建数据集
	train_dataset = LiverVesselDataset(
		args.image_dir,
		args.label_dir,
		tier=args.tier,
		transform=None,  # 可以添加数据增强
		preprocess=True,
		max_cases=config.get('max_cases', None),
		random_sampling=config.get('random_sampling', True),
		enable_smart_sampling=args.smart_sampling,
		sampling_scheduler=sampling_scheduler,
		hard_sample_tracker=hard_sample_tracker
	)
	
	val_dataset = LiverVesselDataset(
		args.image_dir,
		args.label_dir,
		tier=args.tier,
		transform=None,
		preprocess=True,
		max_cases=config.get('max_val_cases', None),
		random_sampling=False,  # 验证集不随机采样
		enable_smart_sampling=False  # 验证集不使用智能采样
	)
	
	# 创建数据加载器
	if args.distributed:
		train_sampler = DistributedSampler(train_dataset)
		val_sampler = DistributedSampler(val_dataset, shuffle=False)
	else:
		train_sampler = None
		val_sampler = None
	
	train_loader = DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		shuffle=(train_sampler is None),
		num_workers=args.num_workers,
		pin_memory=True,
		sampler=train_sampler
	)
	
	val_loader = DataLoader(
		val_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=True,
		sampler=val_sampler
	)
	
	if args.rank == 0:
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
	
	# 分布式包装模型
	if args.distributed:
		model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
	
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
		verbose=True if args.rank == 0 else False
	)
	
	# 恢复训练 (如果需要)
	start_epoch = 0
	best_dice = 0
	
	if args.resume:
		checkpoint = torch.load(args.resume, map_location=device)
		
		if args.distributed:
			# 加载DDP模型参数
			model.module.load_state_dict(checkpoint['model_state_dict'])
		else:
			model.load_state_dict(checkpoint['model_state_dict'])
		
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		start_epoch = checkpoint['epoch'] + 1
		best_dice = checkpoint['best_dice']
		
		if args.rank == 0:
			logger.log_info(f"Resuming from epoch {start_epoch}, best dice: {best_dice:.4f}")
	
	# 训练循环
	for epoch in range(start_epoch, args.epochs):
		# 设置采样器epoch
		if args.distributed:
			train_sampler.set_epoch(epoch)
		
		# 更新采样调度
		if args.smart_sampling and sampling_scheduler is not None:
			sampling_scheduler.update(epoch)
			
			# 记录采样状态
			if args.rank == 0 and logger:
				stats = sampling_scheduler.get_progress_stats()
				logger.log_metrics(stats, epoch, prefix='sampling/')
		
		# 训练一个epoch
		train_loss = train_one_epoch(
			model, train_loader, criterion, optimizer,
			device, logger if args.rank == 0 else None, epoch,
			args=args
		)
		
		# 更新难度图
		if args.smart_sampling and epoch >= args.warmup_epochs and (epoch + 1) % 2 == 0:
			# 确保只在主进程更新
			if args.distributed:
				torch.distributed.barrier()
			
			# 更新难度图
			if args.rank == 0:  # 只在主进程更新
				train_dataset.update_difficulty_maps(
					model.module if args.distributed else model,
					device
				)
				
				# 广播更新完成信号
				if args.distributed:
					torch.distributed.barrier()
		
		# 验证
		if (epoch + 1) % args.val_interval == 0:
			val_loss, val_dice = validate(
				model, val_loader, criterion, device,
				logger if args.rank == 0 else None, epoch,
				visualizer if args.rank == 0 else None,
				args=args
			)
			
			# 更新学习率 (只在主进程)
			if args.rank == 0:
				scheduler.step(val_dice)
				
				# 保存最佳模型
				if val_dice > best_dice:
					best_dice = val_dice
					
					# 保存模型
					checkpoint = {
						'epoch': epoch,
						'model_state_dict': (model.module if args.distributed else model).state_dict(),
						'optimizer_state_dict': optimizer.state_dict(),
						'scheduler_state_dict': scheduler.state_dict(),
						'best_dice': best_dice
					}
					
					torch.save(checkpoint, output_dir / 'best_model.pt')
					logger.log_info(f"Saved best model with dice: {best_dice:.4f}")
		
		# 保存最新模型 (只在主进程)
		if args.rank == 0:
			checkpoint = {
				'epoch': epoch,
				'model_state_dict': (model.module if args.distributed else model).state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'scheduler_state_dict': scheduler.state_dict(),
				'best_dice': best_dice
			}
			
			torch.save(checkpoint, output_dir / 'latest_model.pt')
	
	# 训练结束，清理资源
	if args.smart_sampling and hard_sample_tracker is not None:
		hard_sample_tracker.close()
	
	# 训练结束
	if args.rank == 0:
		logger.log_info(f"Training completed. Best dice: {best_dice:.4f}")
		if logger:
			logger.close()
	
	# 清理分布式环境
	if args.distributed:
		dist.destroy_process_group()



def distributed_main(rank, args):
	"""分布式训练入口点"""
	args.rank = rank
	
	# 初始化日志
	logging.basicConfig(
		level=logging.INFO if args.rank == 0 else logging.WARNING,
		format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	)
	
	# 调用主函数
	main()




if __name__ == '__main__':
	# 设置多进程启动方法 (必须在主程序开始时设置)
	mp.set_start_method('spawn', force=True)
	
	# 解析参数
	args = parse_args()
	
	# 如果使用分布式训练
	if args.distributed:
		mp.spawn(distributed_main, args=(args,), nprocs=args.world_size)
	else:
		main()