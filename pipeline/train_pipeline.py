# train_pipeline.py
"""
新的训练入口 - 8 GPU流水线
按部就班的rank->stage映射，业务逻辑完全交给Dispatcher
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.stages import *
from pipeline.comm import Channel
from pipeline.dispatcher import Dispatcher
from pipeline.health_monitor import HealthMonitor
from pipeline.ckpt_manager import CheckpointManager

# Rank到Stage的映射
RANK2STAGE = {
	0: Preprocess,
	1: PatchDispatch,
	2: CHBranch,
	3: SpatialBranch,
	4: FeatureFuse,
	5: Multiscale,
	6: SegHead
}


def setup_logging(rank: int, log_dir: str = "./logs"):
	"""设置日志"""
	log_dir = Path(log_dir)
	log_dir.mkdir(parents=True, exist_ok=True)
	
	# 为每个rank创建独立的日志文件
	log_file = log_dir / f"rank_{rank}.log"
	
	logging.basicConfig(
		level=logging.INFO,
		format=f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s',
		handlers=[
			logging.FileHandler(log_file),
			logging.StreamHandler()
		]
	)
	
	# 设置各模块日志级别
	logging.getLogger('pipeline').setLevel(logging.INFO)
	logging.getLogger('torch.distributed').setLevel(logging.WARNING)


def load_config(config_path: str) -> Dict:
	"""加载配置文件"""
	config_path = Path(config_path)
	
	if config_path.exists():
		with open(config_path, 'r') as f:
			if config_path.suffix.lower() in ['.yaml', '.yml']:
				return yaml.safe_load(f)
			else:
				# 假设是Python配置文件
				return {}
	else:
		# 返回默认配置
		return get_default_config()


def get_default_config() -> Dict:
	"""默认配置"""
	return {
		'stages': {
			'preprocess': {
				'norm_lower_percentile': 0.5,
				'norm_upper_percentile': 99.5,
				'roi_threshold_percentile': 99.8,
				'min_volume': 1000,
				'tier0_size': [64, 64, 64],  # 简化为较小尺寸便于测试
				'tier0_max': 1,
				'tier1_size': [32, 32, 32],
				'tier1_max': 3,
				'tier2_size': [24, 24, 24],
				'tier2_max': 5
			}
		},
		'data': {
			'batch_size': 1,
			'num_samples': 20,
			'image_shape': [1, 64, 64, 64]
		}
	}


def build_channels(rank: int) -> Tuple[Optional[Channel], List[Channel]]:
	"""
	根据拓扑构建通信通道

	拓扑结构:
	0(Preprocess) -> 1(Dispatch) -> 2(CH), 3(Spatial) -> 4(Fuse) -> 5(Multi) -> 6(SegHead)

	Returns:
		(input_channel, output_channels)
	"""
	# 固定拓扑：输入rank列表，输出rank列表
	topo = {
		0: ([], [1]),  # Preprocess -> Dispatch
		1: ([0], [2, 3]),  # Dispatch -> CH, Spatial
		2: ([1], [4]),  # CH -> Fuse
		3: ([1], [4]),  # Spatial -> Fuse
		4: ([2, 3], [5]),  # Fuse -> Multi
		5: ([4], [6]),  # Multi -> SegHead
		6: ([5], [])  # SegHead -> 终点
	}
	
	if rank not in topo:
		raise ValueError(f"Invalid rank {rank}")
	
	ins, outs = topo[rank]
	
	# 创建输入通道
	in_chan = None
	if ins:
		# 对于有多个输入的rank，这里简化为只取第一个
		# 实际应用中FeatureFuse需要特殊处理多输入
		src_rank = ins[0]
		tag_base = _get_tag_base(src_rank, rank)
		in_chan = Channel(src_rank, rank, tag_base)
	
	# 创建输出通道
	out_chans = []
	for dst_rank in outs:
		tag_base = _get_tag_base(rank, dst_rank)
		out_chan = Channel(rank, dst_rank, tag_base)
		out_chans.append(out_chan)
	
	return in_chan, out_chans


def _get_tag_base(src_rank: int, dst_rank: int) -> int:
	"""生成唯一的tag基数"""
	return 1000 + src_rank * 10 + dst_rank


def create_dummy_dataloader(config: Dict) -> DataLoader:
	"""创建虚拟数据加载器用于测试"""
	data_config = config.get('data', {})
	
	num_samples = data_config.get('num_samples', 10)
	batch_size = data_config.get('batch_size', 1)
	image_shape = data_config.get('image_shape', [1, 64, 64, 64])
	
	# 创建虚拟数据
	images = torch.randn(num_samples, *image_shape)
	labels = torch.randint(0, 2, size=(num_samples, *image_shape))
	
	# 添加case_id
	case_ids = [f"test_case_{i:03d}" for i in range(num_samples)]
	
	# 自定义数据集
	class DummyDataset:
		def __init__(self):
			self.images = images
			self.labels = labels
			self.case_ids = case_ids
		
		def __len__(self):
			return len(self.images)
		
		def __getitem__(self, idx):
			return {
				'image': self.images[idx:idx + 1],  # [1, C, D, H, W]
				'label': self.labels[idx:idx + 1],
				'case_id': [self.case_ids[idx]]
			}
	
	dataset = DummyDataset()
	
	# 简单的DataLoader，不使用多进程避免复杂性
	dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
	
	return dataloader


def worker_main(rank: int, world_size: int, config: Dict, args):
	"""工作进程主函数"""
	try:
		# 设置日志
		setup_logging(rank, args.log_dir)
		logger = logging.getLogger(__name__)
		
		logger.info(f"Worker {rank} starting...")
		
		# 初始化分布式
		os.environ['MASTER_ADDR'] = args.master_addr
		os.environ['MASTER_PORT'] = args.master_port
		
		dist.init_process_group(
			backend='nccl' if torch.cuda.is_available() else 'gloo',
			rank=rank,
			world_size=world_size
		)
		
		# 设置CUDA设备
		if torch.cuda.is_available():
			torch.cuda.set_device(rank)
			device = torch.device(f'cuda:{rank}')
		else:
			device = torch.device('cpu')
		
		logger.info(f"Rank {rank} initialized with device {device}")
		
		# 创建Stage
		if rank in RANK2STAGE:
			stage_class = RANK2STAGE[rank]
			stage_config = config.get('stages', {}).get(stage_class.__name__.lower(), {})
			stage = stage_class(stage_config, device)
			logger.info(f"Created stage: {stage}")
		else:
			# Rank 7 或其他，使用dummy
			logger.info(f"Rank {rank} is dummy, sleeping...")
			import time
			while True:
				time.sleep(10)
		
		# 创建通信通道
		in_chan, out_chans = build_channels(rank)
		logger.info(f"Channels: in={in_chan}, out={len(out_chans)} channels")
		
		# 创建数据加载器 (仅rank 0)
		data_loader = None
		if rank == 0:
			data_loader = create_dummy_dataloader(config)
			logger.info(f"Created data loader with {len(data_loader)} batches")
		
		# 创建健康监控
		health_monitor = HealthMonitor(rank)
		health_monitor.start_monitoring()
		
		# 创建checkpoint管理器
		ckpt_manager = CheckpointManager(args.output_dir, rank)
		
		# 等待所有进程就绪
		dist.barrier()
		logger.info(f"Rank {rank} ready, starting dispatcher...")
		
		# 创建并运行Dispatcher
		dispatcher = Dispatcher(
			stage=stage,
			in_chan=in_chan,
			out_chans=out_chans,
			data_loader=data_loader
		)
		
		# 运行主循环
		dispatcher.run_forever()
		
		logger.info(f"Rank {rank} completed successfully")
	
	except Exception as e:
		logging.error(f"Worker {rank} failed: {e}")
		raise
	finally:
		# 清理
		if 'health_monitor' in locals():
			health_monitor.stop_monitoring()
		
		if dist.is_initialized():
			dist.destroy_process_group()


def main():
	"""主函数"""
	parser = argparse.ArgumentParser(description='8-GPU Pipeline Training')
	
	parser.add_argument('--config', type=str, default='pipeline_config.yaml',
	                    help='Configuration file path')
	parser.add_argument('--world-size', type=int, default=7,
	                    help='Number of processes (0-6 for 7 GPUs)')
	parser.add_argument('--master-addr', type=str, default='localhost',
	                    help='Master address')
	parser.add_argument('--master-port', type=str, default='12355',
	                    help='Master port')
	parser.add_argument('--log-dir', type=str, default='./logs',
	                    help='Log directory')
	parser.add_argument('--output-dir', type=str, default='./output',
	                    help='Output directory')
	
	args = parser.parse_args()
	
	# 创建输出目录
	Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	Path(args.log_dir).mkdir(parents=True, exist_ok=True)
	
	# 加载配置
	config = load_config(args.config)
	
	# 检查CUDA
	if not torch.cuda.is_available():
		print("Warning: CUDA not available, using CPU")
	else:
		available_gpus = torch.cuda.device_count()
		if available_gpus < args.world_size:
			print(f"Warning: Only {available_gpus} GPUs available, need {args.world_size}")
	
	print(f"Starting {args.world_size}-GPU Pipeline Training")
	print(f"Config: {args.config}")
	print(f"Output: {args.output_dir}")
	print(f"Log: {args.log_dir}")
	
	# 启动多进程训练
	mp.spawn(
		worker_main,
		args=(args.world_size, config, args),
		nprocs=args.world_size,
		join=True
	)
	
	print("Pipeline training completed!")


if __name__ == '__main__':
	main()

