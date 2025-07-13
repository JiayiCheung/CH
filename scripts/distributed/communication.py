#communication.py

import torch
import torch.distributed as dist
import logging
from typing import Optional, Any

# 设置日志
logger = logging.getLogger("node_communication")


class NodeCommunicator:
	"""
	管理跨节点通信的工具类
	"""
	
	def __init__(self, rank: int, world_size: int, node_count: int, gpus_per_node: int):
		"""
		初始化节点通信器

		参数:
			rank: 全局进程排名
			world_size: 总进程数
			node_count: 节点数量
			gpus_per_node: 每个节点的GPU数量
		"""
		self.rank = rank
		self.world_size = world_size
		self.node_count = node_count
		self.gpus_per_node = gpus_per_node
		
		# 计算节点排名和本地排名
		self.node_rank = rank // gpus_per_node
		self.local_rank = rank % gpus_per_node
		
		# 计算每个节点的进程排名
		self.node_ranks = {}
		for node in range(node_count):
			start_rank = node * gpus_per_node
			node_ranks = [start_rank + i for i in range(gpus_per_node)]
			self.node_ranks[node] = node_ranks
		
		logger.info(f"NodeCommunicator initialized: rank={rank}, node_rank={self.node_rank}, "
		            f"local_rank={self.local_rank}, world_size={world_size}")
	
	def send_tensor(self, tensor: torch.Tensor, dst_rank: int):
		"""
		发送张量到目标进程

		参数:
			tensor: 要发送的张量
			dst_rank: 目标进程的排名
		"""
		if not dist.is_initialized():
			raise RuntimeError("分布式环境未初始化")
		
		# 确保张量在正确的设备上
		if tensor.device.type == 'cpu' and torch.cuda.is_available():
			tensor = tensor.to(f'cuda:{self.local_rank}')
		
		# 发送张量
		dist.send(tensor, dst_rank)
	
	def recv_tensor(self, src_rank: int, dtype: Optional[torch.dtype] = None,
	                device: Optional[str] = None) -> torch.Tensor:
		"""
		从源进程接收张量

		参数:
			src_rank: 源进程的排名
			dtype: 张量的数据类型 (如果已知)
			device: 接收张量的设备

		返回:
			接收到的张量
		"""
		if not dist.is_initialized():
			raise RuntimeError("分布式环境未初始化")
		
		# 确定接收设备
		if device is None:
			device = f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu'
		
		# 如果知道数据类型和形状，可以预先分配张量
		if dtype is not None:
			# 创建一个初始张量用于接收
			tensor = torch.zeros([1], dtype=dtype, device=device)
			dist.recv(tensor, src_rank)
			return tensor
		else:
			# 如果不知道类型，需要使用更复杂的机制
			# 这里简化为使用默认浮点类型
			tensor = torch.zeros([1], dtype=torch.float32, device=device)
			dist.recv(tensor, src_rank)
			return tensor
	
	def broadcast(self, tensor: torch.Tensor, src_rank: int) -> torch.Tensor:
		"""
		从源进程广播张量到所有进程

		参数:
			tensor: 要广播的张量
			src_rank: 源进程的排名

		返回:
			广播后的张量
		"""
		if not dist.is_initialized():
			raise RuntimeError("分布式环境未初始化")
		
		dist.broadcast(tensor, src_rank)
		return tensor
	
	def barrier(self):
		"""
		同步所有进程
		"""
		if not dist.is_initialized():
			raise RuntimeError("分布式环境未初始化")
		
		dist.barrier()
	
	def is_first_in_node(self) -> bool:
		"""
		检查当前进程是否是节点中的第一个进程

		返回:
			如果是节点中的第一个进程则为True
		"""
		return self.local_rank == 0
	
	def is_master(self) -> bool:
		"""
		检查当前进程是否是主进程(全局排名0)

		返回:
			如果是主进程则为True
		"""
		return self.rank == 0