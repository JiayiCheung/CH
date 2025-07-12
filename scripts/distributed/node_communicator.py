import torch
import torch.distributed as dist
import torch.nn.functional as F
import zlib
import pickle
import io


class NodeCommunicator:
	"""管理节点间通信和数据传输优化"""
	
	def __init__(self, world_size, rank, local_rank, node_rank, node_count=2):
		"""
		初始化节点通信管理器

		参数:
			world_size: 总进程数
			rank: 全局进程编号
			local_rank: 节点内GPU编号
			node_rank: 节点编号 (0或1)
			node_count: 总节点数
		"""
		self.world_size = world_size
		self.rank = rank
		self.local_rank = local_rank
		self.node_rank = node_rank
		self.node_count = node_count
		
		# 计算每个节点的进程数
		self.procs_per_node = world_size // node_count
		
		# 创建节点内进程组
		self.node_ranks = list(range(
			node_rank * self.procs_per_node,
			(node_rank + 1) * self.procs_per_node
		))
		self.node_group = dist.new_group(ranks=self.node_ranks)
		
		# 通信优化参数
		self.compression_enabled = True
		self.compression_level = 3  # 1-9
		self.use_fp16_transfer = True
		self.max_chunk_size = 1024 * 1024 * 50  # 50MB
		
		print(f"Node communicator initialized: rank={rank}, node={node_rank}, group={self.node_ranks}")
	
	def is_master_node(self):
		"""检查是否是主节点"""
		return self.node_rank == 0
	
	def is_master_process(self):
		"""检查是否是主进程"""
		return self.rank == 0
	
	def is_local_master(self):
		"""检查是否是节点内主进程"""
		return self.local_rank == 0
	
	def is_cross_node(self, dest_rank):
		"""检查目标进程是否在不同节点"""
		dest_node = dest_rank // self.procs_per_node
		return dest_node != self.node_rank
	
	def _compress_tensor(self, tensor):
		"""压缩张量以优化网络传输"""
		# 降低精度 (如果需要)
		if self.use_fp16_transfer and tensor.dtype == torch.float32:
			tensor = tensor.half()
		
		# 序列化张量
		buffer = io.BytesIO()
		torch.save(tensor, buffer)
		data = buffer.getvalue()
		
		# 压缩数据 (如果启用)
		if self.compression_enabled:
			data = zlib.compress(data, level=self.compression_level)
		
		return data
	
	def _decompress_tensor(self, data, dtype=None, device=None):
		"""解压缩并恢复张量"""
		# 解压缩数据 (如果启用)
		if self.compression_enabled:
			data = zlib.decompress(data)
		
		# 反序列化张量
		buffer = io.BytesIO(data)
		tensor = torch.load(buffer, map_location=device)
		
		# 恢复精度 (如果需要)
		if dtype is not None and tensor.dtype != dtype:
			tensor = tensor.to(dtype=dtype)
		
		return tensor
	
	def _split_tensor(self, tensor, max_size=None):
		"""将张量分割为适合传输的块"""
		if max_size is None:
			max_size = self.max_chunk_size
		
		# 计算总大小和分块数
		total_size = tensor.numel() * tensor.element_size()
		num_chunks = (total_size + max_size - 1) // max_size
		
		if num_chunks <= 1:
			return [tensor]
		
		# 计算每个维度的分割方式
		chunks = []
		tensor_flat = tensor.view(-1)
		chunk_size = (tensor_flat.numel() + num_chunks - 1) // num_chunks
		
		for i in range(0, tensor_flat.numel(), chunk_size):
			end = min(i + chunk_size, tensor_flat.numel())
			chunks.append(tensor_flat[i:end].clone())
		
		return chunks
	
	def send_tensor(self, tensor, dst_rank, tag=0):
		"""发送张量到目标进程"""
		if self.is_cross_node(dst_rank):
			# 跨节点传输需要优化
			data = self._compress_tensor(tensor)
			size_tensor = torch.tensor([len(data)], dtype=torch.long, device=tensor.device)
			
			# 发送数据大小
			dist.send(size_tensor, dst=dst_rank, tag=tag)
			
			# 发送实际数据 (使用bytes对象)
			dist.send(
				torch.ByteTensor(list(data)).to(tensor.device),
				dst=dst_rank,
				tag=tag + 1
			)
		else:
			# 节点内直接传输
			dist.send(tensor, dst=dst_rank, tag=tag)
	
	def recv_tensor(self, src_rank, tag=0, dtype=None, device=None):
		"""从源进程接收张量"""
		if device is None:
			device = torch.device(f'cuda:{self.local_rank}')
		
		if self.is_cross_node(src_rank):
			# 跨节点接收需要解压缩
			size_tensor = torch.zeros(1, dtype=torch.long, device=device)
			dist.recv(size_tensor, src=src_rank, tag=tag)
			
			data_size = size_tensor.item()
			data_tensor = torch.zeros(data_size, dtype=torch.uint8, device=device)
			dist.recv(data_tensor, src=src_rank, tag=tag + 1)
			
			# 转换回bytes并解压缩
			data = bytes(data_tensor.cpu().numpy())
			tensor = self._decompress_tensor(data, dtype=dtype, device=device)
		else:
			# 节点内直接接收
			tensor = torch.empty([], dtype=dtype, device=device)
			dist.recv(tensor, src=src_rank, tag=tag)
		
		return tensor
	
	def broadcast_tensor(self, tensor, src_rank, group=None):
		"""广播张量"""
		if group is None:
			group = dist.group.WORLD
		
		dist.broadcast(tensor, src=src_rank, group=group)
		return tensor
	
	def node_barrier(self):
		"""节点内同步屏障"""
		dist.barrier(group=self.node_group)
	
	def global_barrier(self):
		"""全局同步屏障"""
		dist.barrier()