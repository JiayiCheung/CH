# pipeline/channel.py
"""
通信通道定义
"""

import torch
from dataclasses import dataclass


@dataclass
class Channel:
	"""
	Stage间通信通道

	定义了源Stage和目标Stage之间的连接关系，
	包括rank信息和设备信息用于优化数据传输
	"""
	src_rank: int  # 源rank
	dst_rank: int  # 目标rank
	src_device: torch.device  # 源设备
	dst_device: torch.device  # 目标设备
	buffer_size: int = 10  # 缓冲区大小
	
	def __post_init__(self):
		"""验证通道配置"""
		if self.src_rank == self.dst_rank:
			# 同rank内通信，设备应该一致或明确指定
			pass
		
		if self.buffer_size <= 0:
			raise ValueError("buffer_size必须大于0")
	
	def is_cross_node(self) -> bool:
		"""判断是否跨节点通信"""
		# 简化判断：不同rank认为是跨节点
		return self.src_rank != self.dst_rank
	
	def is_cross_gpu(self) -> bool:
		"""判断是否跨GPU通信"""
		return (self.src_device.type == 'cuda' and
		        self.dst_device.type == 'cuda' and
		        self.src_device.index != self.dst_device.index)
	
	def get_transfer_method(self) -> str:
		"""获取推荐的传输方法"""
		if self.src_device.type == 'cuda' and self.dst_device.type == 'cuda':
			return 'nccl_p2p'  # GPU到GPU，使用NCCL P2P
		elif self.src_device.type == 'cpu' or self.dst_device.type == 'cpu':
			return 'gloo_object'  # 涉及CPU，使用Gloo对象传输
		else:
			return 'fallback'
	
	def __repr__(self):
		return (f"Channel(src_rank={self.src_rank}, dst_rank={self.dst_rank}, "
		        f"src_device={self.src_device}, dst_device={self.dst_device}, "
		        f"method={self.get_transfer_method()})")


