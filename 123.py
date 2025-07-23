#123.py
"""
分布式训练环境测试脚本
用于验证NCCL通信是否正常工作
"""

import os
import torch
import torch.distributed as dist
import time
import logging


def setup_logging():
	logging.basicConfig(
		level=logging.INFO,
		format='[Rank %(rank)s] %(asctime)s - %(levelname)s - %(message)s',
		handlers=[
			logging.StreamHandler(),
			logging.FileHandler(f'dist_test_rank_{os.environ.get("RANK", "unknown")}.log')
		]
	)


def test_distributed():
	"""测试分布式环境"""
	
	# 获取环境变量
	rank = int(os.environ['RANK'])
	world_size = int(os.environ['WORLD_SIZE'])
	local_rank = int(os.environ['LOCAL_RANK'])
	master_addr = os.environ['MASTER_ADDR']
	master_port = os.environ['MASTER_PORT']
	
	logger = logging.getLogger()
	logger.addFilter(lambda record: setattr(record, 'rank', rank) or True)
	
	logger.info(f"开始分布式测试")
	logger.info(f"Rank: {rank}/{world_size}, Local Rank: {local_rank}")
	logger.info(f"Master: {master_addr}:{master_port}")
	
	# 设置CUDA设备
	torch.cuda.set_device(local_rank)
	device = torch.device(f'cuda:{local_rank}')
	logger.info(f"CUDA设备: {device}")
	
	try:
		# 初始化进程组
		logger.info("初始化分布式进程组...")
		dist.init_process_group(
			backend='nccl',
			init_method='env://',
			world_size=world_size,
			rank=rank,
			timeout=torch.distributed.default_pg_timeout  # 使用默认超时
		)
		logger.info("✅ 分布式进程组初始化成功")
		
		# 测试基本通信
		logger.info("测试基本通信...")
		tensor = torch.ones(2, device=device) * rank
		dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
		expected = sum(range(world_size)) * torch.ones(2, device=device)
		
		if torch.allclose(tensor, expected):
			logger.info("✅ All-reduce通信测试成功")
		else:
			logger.error(f"❌ All-reduce测试失败: 期望 {expected}, 得到 {tensor}")
		
		# 测试广播
		if rank == 0:
			broadcast_tensor = torch.randn(10, device=device)
		else:
			broadcast_tensor = torch.zeros(10, device=device)
		
		dist.broadcast(broadcast_tensor, src=0)
		logger.info("✅ 广播通信测试成功")
		
		# 同步测试
		logger.info("测试同步屏障...")
		dist.barrier()
		logger.info("✅ 同步屏障测试成功")
		
		# 延迟测试
		start_time = time.time()
		for i in range(10):
			test_tensor = torch.randn(1000, device=device)
			dist.all_reduce(test_tensor)
		elapsed = time.time() - start_time
		logger.info(f"✅ 通信延迟测试完成: {elapsed:.3f}s for 10 iterations")
		
		logger.info("🎉 所有测试通过！分布式环境正常")
	
	except Exception as e:
		logger.error(f"❌ 分布式测试失败: {e}")
		import traceback
		traceback.print_exc()
		return False
	
	finally:
		if dist.is_initialized():
			dist.destroy_process_group()
	
	return True


if __name__ == "__main__":
	setup_logging()
	
	# 检查必要的环境变量
	required_env = ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']
	missing = [var for var in required_env if var not in os.environ]
	
	if missing:
		print(f"❌ 缺少环境变量: {missing}")
		exit(1)
	
	success = test_distributed()
	exit(0 if success else 1)