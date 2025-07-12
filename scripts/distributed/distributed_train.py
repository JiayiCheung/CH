import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data import LiverVesselDataset, HardSampleTracker
from models import VesselSegmenter
from losses import VesselSegmentationLoss
from utils import Logger, SamplingScheduler
from data.sampling_manager import SamplingManager
from scripts.evaluation import EvaluationManager


def parse_args():
	"""Parse command line arguments"""
	parser = argparse.ArgumentParser(description='Train Liver Vessel Segmentation Model with Pipeline Parallelism')
	
	# Data parameters
	parser.add_argument('--image_dir', type=str, required=True, help='Path to input volume or directory')
	parser.add_argument('--label_dir', type=str, required=True, help='Path to label directory')
	parser.add_argument('--output_dir', type=str, default='./output', help='Path to output directory')
	parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
	
	# Training parameters
	parser.add_argument('--tier', type=int, help='Training tier (0, 1, or 2)')
	parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
	parser.add_argument('--micro_batch_size', type=int, default=1, help='Micro batch size for pipeline parallelism')
	parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
	parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
	parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
	parser.add_argument('--resume', type=str, help='Path to model checkpoint')
	
	# Pipeline-specific parameters
	parser.add_argument('--local_rank', type=int, default=0, help='Local process rank')
	parser.add_argument('--world_size', type=int, default=4, help='Total number of GPUs')
	
	# Other parameters
	parser.add_argument('--seed', type=int, default=42, help='Random seed')
	parser.add_argument('--val_interval', type=int, default=5, help='Validation interval')
	parser.add_argument('--amp', action='store_true', help='Use mixed precision training')
	parser.add_argument('--smart_sampling', action='store_true', help='Enable smart sampling')
	
	return parser.parse_args()


def split_model_for_pipeline(model, local_rank, world_size):
	"""Split model into pipeline stages"""
	# Import stages implementation
	from scripts.distributed.stages import FrontendStage, CHProcessingStage, SpatialFusionStage, BackendStage
	
	# Create stages based on available GPUs
	stages = []
	if world_size >= 4:  # Need at least 4 GPUs
		# Frontend processing stage (GPU 0)
		stages.append(FrontendStage(model, device=f'cuda:0'))
		
		# CH processing stage (GPU 1)
		stages.append(CHProcessingStage(model, device=f'cuda:1'))
		
		# Spatial processing and fusion stage (GPU 2)
		stages.append(SpatialFusionStage(model, device=f'cuda:2'))
		
		# Backend processing stage (GPU 3)
		stages.append(BackendStage(model, device=f'cuda:3'))
	else:
		# If fewer GPUs, degenerate to sequential execution on same GPUs
		for i in range(min(world_size, 4)):
			device = f'cuda:{i}'
			if i == 0:
				stages.append(FrontendStage(model, device=device))
			elif i == 1:
				stages.append(CHProcessingStage(model, device=device))
			elif i == 2:
				stages.append(SpatialFusionStage(model, device=device))
			elif i == 3:
				stages.append(BackendStage(model, device=device))
	
	return stages


def setup_pipeline_model(model, world_size, config):
	"""Set up pipeline parallel model"""
	# Import pipeline implementation
	from scripts.distributed.pipeline import StagePipeline
	
	# Split model into stages
	stages = split_model_for_pipeline(model, 0, world_size)
	
	# Create pipeline
	pipeline = StagePipeline(stages)
	
	return pipeline


def build_optimizer(model_stages, base_lr=1e-4, kernel_lr_scale=0.1, weight_decay=1e-5):
	"""Build optimizer for pipeline model stages"""
	# Collect parameters from all stages
	edge_kernels, others = [], []
	
	for stage in model_stages:
		for name, param in stage.named_parameters():
			if not param.requires_grad:
				continue
			if name.endswith(".kernels"):
				edge_kernels.append(param)
			else:
				others.append(param)
	
	param_groups = [
		{
			"params": edge_kernels,
			"lr": base_lr * kernel_lr_scale,
			"weight_decay": 0.0
		},
		{
			"params": others,
			"lr": base_lr,
			"weight_decay": weight_decay
		}
	]
	
	print(f"Optimizer: {len(edge_kernels)} edge kernels with lr={base_lr * kernel_lr_scale}, "
	      f"{len(others)} other params with lr={base_lr}")
	
	return optim.Adam(param_groups)


def train_one_epoch_pipeline(pipeline, dataloader, criterion, optimizer, scaler, device, epoch, args, logger=None):
	"""Train one epoch using pipeline parallelism"""
	# Set training mode
	for stage in pipeline.stages:
		stage.train()
	
	running_loss = 0.0
	
	# Create iterator
	iterator = tqdm(dataloader, desc=f"Epoch {epoch}")
	
	for batch in iterator:
		# Get data
		img = batch['image']  # Keep on CPU, will be moved by pipeline
		lab = batch['label'].to(device)  # Labels go to loss computation device
		tier = batch['tier']
		
		# Clear gradients
		optimizer.zero_grad(set_to_none=True)
		
		# Process each tier
		outputs = []
		for j, t in enumerate(tier):
			# Set current tier
			pipeline.set_tier(int(t))
			
			# Forward pass (pipeline processing)
			with autocast(enabled=args.amp):
				outputs.append(pipeline.forward(img[j:j + 1], is_training=True, amp_enabled=args.amp))
		
		# Combine outputs
		output = torch.cat(outputs)
		
		# Compute loss
		with autocast(enabled=args.amp):
			loss = criterion(output, lab)
		
		# Backward pass and optimization
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()
		
		# Update statistics
		running_loss += loss.item()
		
		# Update progress bar
		iterator.set_postfix(loss=loss.item())
	
	# Calculate average loss
	avg_loss = running_loss / len(dataloader)
	
	return avg_loss


def run_simple_evaluation(pipeline, val_loader, device, args, logger, max_samples=5):
	"""Simple evaluation for pipeline model"""
	# Set evaluation mode
	for stage in pipeline.stages:
		stage.eval()
	
	metrics_sum = {}
	total_samples = 0
	
	with torch.no_grad():
		for i, batch in enumerate(val_loader):
			if i >= max_samples:
				break
			
			images = batch['image']  # Keep on CPU, will be moved by pipeline
			labels = batch['label'].to(device)
			tiers = batch['tier']
			
			for j, tier in enumerate(tiers):
				# Set current tier
				pipeline.set_tier(int(tier))
				
				# Forward pass
				output = pipeline.forward(images[j:j + 1], is_training=False)
				
				# Calculate simple metrics
				pred = (output > 0.5).float()
				target = labels[j:j + 1]
				
				# Calculate Dice
				intersection = torch.sum(pred * target)
				union = torch.sum(pred) + torch.sum(target)
				dice = (2.0 * intersection) / (union + 1e-7)
				
				if 'dice' not in metrics_sum:
					metrics_sum['dice'] = 0
				metrics_sum['dice'] += dice.item()
				total_samples += 1
	
	# Set back to training mode
	for stage in pipeline.stages:
		stage.train()
	
	if total_samples > 0:
		return {k: v / total_samples for k, v in metrics_sum.items()}
	return {'dice': 0.0}


def setup_evaluation_config(config, output_dir):
	"""Set up evaluation configuration"""
	evaluation_config = config.get('evaluation', {})
	
	# Ensure all necessary configuration items exist
	defaults = {
		'eval_full_interval': 10,
		'eval_quick_interval': 2,
		'quick_samples': 5,
		'group_by_tier': True,
		'feature_mmap_enabled': False,
		'feature_mmap_dir': os.path.join(output_dir, 'eval_tier_features'),
		'clear_cache_interval': 3,
		'max_eval_samples': 50,
		'include_advanced_metrics': False,
		'save_predictions': False
	}
	
	for key, default_value in defaults.items():
		if key not in evaluation_config:
			evaluation_config[key] = default_value
	
	return evaluation_config


def main_worker(local_rank, args):
	"""Main worker function for pipeline parallel training"""
	# Set device
	torch.cuda.set_device(local_rank)
	device = torch.device(f'cuda:{local_rank}')
	
	# Create logger
	logger = Logger(Path(args.output_dir) / 'logs') if local_rank == 0 else None
	
	# Load configuration
	config = yaml.safe_load(open(args.config, 'r'))
	
	# Set random seed
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	
	# Create sampling manager
	sampling_manager = None
	if args.smart_sampling and local_rank == 0:
		sampling_manager = SamplingManager(
			config.get('smart_sampling', {}),
			logger=logger
		)
	
	# Create sampling scheduler
	sampling_scheduler = None
	if config.get('smart_sampling', {}).get('enabled', False) and local_rank == 0:
		sampling_scheduler = SamplingScheduler(
			base_tier1=config.get('smart_sampling', {}).get('base_tier1', 10),
			base_tier2=config.get('smart_sampling', {}).get('base_tier2', 30),
			max_tier1=config.get('smart_sampling', {}).get('max_tier1', 20),
			max_tier2=config.get('smart_sampling', {}).get('max_tier2', 60),
			warmup_epochs=config.get('smart_sampling', {}).get('warmup_epochs', 5),
			enable_hard_mining=config.get('smart_sampling', {}).get('enable_hard_mining', True),
			enable_adaptive_density=config.get('smart_sampling', {}).get('enable_adaptive_density', True),
			enable_importance_sampling=config.get('smart_sampling', {}).get('enable_importance_sampling', True),
			logger=logger
		)
	
	# Create hard sample tracker
	hard_sample_tracker = None
	if local_rank == 0:
		hard_sample_tracker = HardSampleTracker(
			base_dir=Path(args.output_dir) / 'difficulty_maps',
			logger=logger
		)
	
	# Create datasets
	if local_rank == 0:
		train_dataset = LiverVesselDataset(
			args.image_dir,
			args.label_dir,
			tier=args.tier,
			transform=None,
			preprocess=True,
			max_cases=config.get('max_cases'),
			random_sampling=config.get('random_sampling', True),
			enable_smart_sampling=args.smart_sampling,
			sampling_params=sampling_scheduler.get_tier_sampling_params() if sampling_scheduler else None,
			hard_sample_tracker=hard_sample_tracker,
			logger=logger
		)
		
		val_dataset = LiverVesselDataset(
			args.image_dir,
			args.label_dir,
			tier=args.tier,
			transform=None,
			preprocess=True,
			max_cases=config.get('max_val_cases'),
			random_sampling=False,
			enable_smart_sampling=False,
			logger=logger
		)
		
		# Create data loaders
		train_loader = DataLoader(
			train_dataset,
			batch_size=args.batch_size,
			shuffle=True,
			num_workers=args.num_workers,
			pin_memory=True,
			drop_last=True
		)
		
		val_loader = DataLoader(
			val_dataset,
			batch_size=1,
			shuffle=False,
			num_workers=args.num_workers,
			pin_memory=True
		)
	else:
		train_loader = None
		val_loader = None
	
	# Create base model
	model = VesselSegmenter(
		in_channels=1,
		out_channels=1,
		ch_params=config.get('ch_params'),
		tier_params=config.get('tier_params')
	)
	
	# Only process rank 0 needs to setup the pipeline
	if local_rank == 0:
		# Setup pipeline model
		if logger:
			logger.log_info("Setting up pipeline parallelism for training")
		
		pipeline_model = setup_pipeline_model(model, args.world_size, config)
		
		# Create loss function on the last stage's device
		last_device = pipeline_model.stages[-1].device
		criterion = VesselSegmentationLoss(
			num_classes=1,
			vessel_weight=config.get('vessel_weight', 10.0),
			tumor_weight=config.get('tumor_weight', 15.0),
			use_boundary=config.get('use_boundary', True)
		).to(last_device)
		
		# Create optimizer
		optimizer = build_optimizer(
			pipeline_model.stages,
			base_lr=args.lr,
			kernel_lr_scale=config.get('optimizer', {}).get('kernel_lr_scale', 0.1),
			weight_decay=config.get('optimizer', {}).get('weight_decay', 1e-5)
		)
		
		# Create gradient scaler
		scaler = GradScaler(enabled=args.amp)
		
		# Create evaluation manager
		evaluation_manager = None
		evaluation_config = setup_evaluation_config(config, args.output_dir)
		try:
			evaluation_manager = EvaluationManager(evaluation_config, logger=logger)
		except Exception as e:
			if logger:
				logger.log_warning(f"Failed to create EvaluationManager: {e}. Using simple evaluation.")
		
		# Define evaluation function
		def run_evaluation(epoch):
			if evaluation_manager and evaluation_manager.should_evaluate(epoch):
				try:
					# Execute evaluation
					val_metrics = evaluation_manager.evaluate(pipeline_model, val_loader, epoch)
					
					if val_metrics:
						# Log validation metrics
						for key, value in val_metrics.items():
							logger.log_metrics({f'val/{key}': value}, epoch)
						return val_metrics
				except Exception as e:
					if logger:
						logger.log_warning(
							f"Advanced evaluation failed at epoch {epoch}: {e}. Using simple evaluation.")
			
			# Fallback to simple evaluation
			if (epoch % args.val_interval == 0 or epoch == args.epochs - 1):
				return run_simple_evaluation(pipeline_model, val_loader, last_device, args, logger)
			
			return None
		
		# Resume from checkpoint if provided
		start_epoch = 0
		best_dice = 0.0
		if args.resume:
			if os.path.isfile(args.resume):
				checkpoint = torch.load(args.resume, map_location=device)
				start_epoch = checkpoint['epoch'] + 1
				best_dice = checkpoint.get('best_dice', 0.0)
				
				# Load model weights
				if 'model_state_dict' in checkpoint:
					state_dict = checkpoint['model_state_dict']
					if isinstance(state_dict, dict) and 'stage_0' in state_dict:
						# Pipeline checkpoint
						for i, stage in enumerate(pipeline_model.stages):
							if f'stage_{i}' in state_dict:
								stage.load_state_dict(state_dict[f'stage_{i}'])
					else:
						# Regular checkpoint - distribute to stages
						model.load_state_dict(state_dict)
						# Re-setup pipeline to propagate loaded weights
						pipeline_model = setup_pipeline_model(model, args.world_size, config)
				
				# Load optimizer state
				if 'optimizer' in checkpoint:
					optimizer.load_state_dict(checkpoint['optimizer'])
				
				# Load mixed precision state
				if 'scaler' in checkpoint and args.amp:
					scaler.load_state_dict(checkpoint['scaler'])
				
				logger.log_info(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
			else:
				logger.log_warning(f"No checkpoint found at '{args.resume}'")
		
		# Training loop
		for epoch in range(start_epoch, args.epochs):
			# Update sampling strategy
			if sampling_manager and sampling_manager.should_update(epoch):
				try:
					sampling_manager.update_sampling_strategy(model, train_dataset, epoch, device)
				except Exception as e:
					if logger:
						logger.log_warning(f"Sampling strategy update failed: {e}")
			
			# Update sampling scheduler
			if sampling_scheduler:
				sampling_scheduler.update(epoch)
				if hasattr(train_dataset, 'sampler') and train_dataset.sampler:
					train_dataset.sampler.set_sampling_params(
						sampling_scheduler.get_tier_sampling_params()
					)
			
			# Train one epoch
			train_loss = train_one_epoch_pipeline(
				pipeline_model, train_loader, criterion, optimizer, scaler, last_device, epoch, args, logger
			)
			
			# Log training loss
			logger.log_info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")
			logger.log_metrics({'train/loss': train_loss}, epoch)
			
			# Validate
			val_metrics = run_evaluation(epoch)
			
			if val_metrics:
				current_dice = val_metrics.get('dice', 0)
				if current_dice > best_dice:
					best_dice = current_dice
					
					# Save best model
					stages_dict = {}
					for i, stage in enumerate(pipeline_model.stages):
						stages_dict[f'stage_{i}'] = stage.get_state_dict_prefix()
					
					torch.save({
						'epoch': epoch,
						'model_state_dict': stages_dict,
						'best_dice': best_dice,
						'config': config
					}, Path(args.output_dir) / 'best_model.pt')
					
					logger.log_info(f"New best model saved with Dice: {best_dice:.4f}")
			
			# Save periodic checkpoint
			if epoch % 10 == 0 or epoch == args.epochs - 1:
				stages_dict = {}
				for i, stage in enumerate(pipeline_model.stages):
					stages_dict[f'stage_{i}'] = stage.get_state_dict_prefix()
				
				torch.save({
					'epoch': epoch,
					'model_state_dict': stages_dict,
					'optimizer': optimizer.state_dict(),
					'scaler': scaler.state_dict(),
					'best_dice': best_dice,
					'config': config
				}, Path(args.output_dir) / f'checkpoint_epoch_{epoch}.pt')
				
				logger.log_info(f"Checkpoint saved at epoch {epoch}")
		
		# Training finished
		# Cleanup evaluation resources
		if evaluation_manager:
			try:
				if hasattr(evaluation_manager, '_cleanup_evaluation_resources'):
					evaluation_manager._cleanup_evaluation_resources()
				
				# Cleanup MMap resources
				if (hasattr(evaluation_manager, 'mmap_manager') and
						evaluation_manager.mmap_manager and
						evaluation_manager.config.get('feature_mmap_enabled', False)):
					
					mmap_dir = Path(evaluation_manager.config['feature_mmap_dir'])
					if mmap_dir.exists():
						try:
							shutil.rmtree(mmap_dir)
							logger.log_info("Cleaned up evaluation temporary files")
						except Exception as e:
							logger.log_warning(f"Error cleaning up temp files: {e}")
			
			except Exception as e:
				logger.log_warning(f"Error during evaluation cleanup: {e}")
		
		logger.log_info(f"Training completed. Best Dice: {best_dice:.4f}")
		
		# Save final model
		stages_dict = {}
		for i, stage in enumerate(pipeline_model.stages):
			stages_dict[f'stage_{i}'] = stage.get_state_dict_prefix()
		
		torch.save({
			'epoch': args.epochs - 1,
			'model_state_dict': stages_dict,
			'best_dice': best_dice,
			'config': config
		}, Path(args.output_dir) / 'final_model.pt')
		
		logger.log_info("Final model saved")
	
	# Other processes just wait
	else:
		# In a real implementation, these processes would handle specific pipeline stages
		# For now, they just wait for rank 0 to finish
		import time
		while True:
			time.sleep(3600)  # Sleep for an hour


def main():
	"""Main function"""
	args = parse_args()
	
	# Create output directory
	Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	
	# Launch workers
	main_worker(args.local_rank, args)


if __name__ == '__main__':
	main()