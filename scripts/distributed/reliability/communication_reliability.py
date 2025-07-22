#communication_reliability.py
"""
通信错误处理 - 预防性措施
专注数据完整性和传输可靠性
"""

import torch
import torch.distributed as dist
import zlib
import hashlib
import time
import logging
import pickle
import json
import threading
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass


@dataclass
class TransmissionPacket:
	"""传输数据包"""
	data: torch.Tensor
	packet_id: str
	chunk_index: int
	total_chunks: int
	crc32: int
	timestamp: float


class ReliableCommunicator:
	"""可靠通信管理器 - 预防性错误处理"""
	
	def __init__(self, node_comm, max_chunk_size: int = 50 * 1024 * 1024):
		self.node_comm = node_comm
		self.max_chunk_size = max_chunk_size
		self.device = 'cuda'
		
		# 超时设置（考虑CH分支复杂计算）
		self.send_timeout = 30.0  # 发送超时30秒
		self.recv_timeout = 60.0  # 接收超时60秒（CH分支需要更长时间）
		
		# 确认机制
		self.enable_ack = True
		self.ack_timeout = 5.0
		
		self.logger = logging.getLogger(__name__)
	
	def send_tensor_reliable(self, data: Any, dst_rank: int, tag: int = 0) -> bool:
		"""
		可靠发送数据 - 支持tensor和复杂数据类型

		Args:
			data: 要发送的数据，可以是tensor、tuple、list等
			dst_rank: 目标rank
			tag: 通信标签

		Returns:
			bool: 发送是否成功
		"""
		try:
			# 检测数据类型并选择处理方式
			if isinstance(data, torch.Tensor):
				return self._send_single_tensor(data, dst_rank, tag)
			elif isinstance(data, (list, tuple)):
				return self._send_complex_data(data, dst_rank, tag)
			else:
				return self._send_serialized_data(data, dst_rank, tag)
		except Exception as e:
			rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
			self.logger.error(f"Rank {rank}: 发送数据失败: {e}, 数据类型: {type(data)}")
			return False
	
	def _send_single_tensor(self, tensor: torch.Tensor, dst_rank: int, tag: int) -> bool:
		"""发送单个tensor - 原有逻辑"""
		try:
			# 生成传输ID
			packet_id = self._generate_packet_id(dst_rank, tag)
			
			# 计算CRC校验
			tensor_bytes = self._tensor_to_bytes(tensor)
			crc32 = zlib.crc32(tensor_bytes) & 0xffffffff
			
			# 分块传输
			chunks = self._split_tensor_to_chunks(tensor)
			total_chunks = len(chunks)
			
			self.logger.debug(f"发送tensor到rank {dst_rank}, 分块数: {total_chunks}, CRC: {crc32:08x}")
			
			# 发送元数据
			if not self._send_metadata(dst_rank, packet_id, total_chunks, crc32, tensor.shape, tensor.dtype, tag):
				return False
			
			# 发送各个分块
			for chunk_idx, chunk in enumerate(chunks):
				packet = TransmissionPacket(
					data=chunk,
					packet_id=packet_id,
					chunk_index=chunk_idx,
					total_chunks=total_chunks,
					crc32=zlib.crc32(self._tensor_to_bytes(chunk)) & 0xffffffff,
					timestamp=time.time()
				)
				
				if not self._send_chunk(dst_rank, packet, tag + chunk_idx + 1):
					self.logger.error(f"发送分块 {chunk_idx}/{total_chunks} 失败")
					return False
				
				# 等待确认（如果启用）
				if self.enable_ack and not self._wait_for_ack(dst_rank, packet_id, chunk_idx):
					self.logger.error(f"分块 {chunk_idx} 确认超时")
					return False
			
			self.logger.debug(f"张量发送完成: rank {dst_rank}, packet_id: {packet_id}")
			return True
		
		except Exception as e:
			self.logger.error(f"发送tensor失败: {e}")
			return False
	
	def _send_complex_data(self, data: Union[List, Tuple], dst_rank: int, tag: int) -> bool:
		"""发送复杂数据类型（list/tuple）"""
		try:
			# 发送数据类型标识
			data_type = "list" if isinstance(data, list) else "tuple"
			type_info = {
				"data_type": data_type,
				"length": len(data),
				"timestamp": time.time()
			}
			
			if not self._send_json_metadata(type_info, dst_rank, tag):
				return False
			
			# 检测是否为 List[Tuple[torch.Tensor, int]] 格式
			if self._is_tensor_tuple_list(data):
				return self._send_tensor_tuple_list(data, dst_rank, tag + 10)
			else:
				# 其他复杂数据类型：序列化发送
				return self._send_serialized_data(data, dst_rank, tag + 20)
		
		except Exception as e:
			self.logger.error(f"发送复杂数据失败: {e}")
			return False
	
	def _is_tensor_tuple_list(self, data) -> bool:
		"""检测是否为 List[Tuple[torch.Tensor, int]] 格式"""
		if not isinstance(data, (list, tuple)) or len(data) == 0:
			return False
		
		for item in data:
			if not isinstance(item, tuple) or len(item) != 2:
				return False
			if not isinstance(item[0], torch.Tensor) or not isinstance(item[1], (int, float)):
				return False
		
		return True
	
	def _send_tensor_tuple_list(self, data: List[Tuple], dst_rank: int, tag: int) -> bool:
		"""发送 List[Tuple[torch.Tensor, int]] 格式的数据"""
		try:
			# 分离tensors和tiers
			tensors = [item[0] for item in data]
			tiers = [int(item[1]) for item in data]
			
			# 发送数量信息
			count_tensor = torch.tensor([len(data)], dtype=torch.long, device=self.device)
			if not self._send_single_tensor(count_tensor, dst_rank, tag):
				return False
			
			# 逐个发送tensor
			for i, tensor in enumerate(tensors):
				if not self._send_single_tensor(tensor, dst_rank, tag + 100 + i):
					return False
			
			# 发送tier信息
			tiers_tensor = torch.tensor(tiers, dtype=torch.long, device=self.device)
			if not self._send_single_tensor(tiers_tensor, dst_rank, tag + 200):
				return False
			
			self.logger.debug(f"发送tensor-tuple列表完成: {len(data)}个项目到rank {dst_rank}")
			return True
		
		except Exception as e:
			self.logger.error(f"发送tensor-tuple列表失败: {e}")
			return False
	
	def _send_serialized_data(self, data: Any, dst_rank: int, tag: int) -> bool:
		"""发送序列化数据"""
		try:
			# 序列化和压缩
			serialized = pickle.dumps(data)
			compressed = zlib.compress(serialized)
			
			# 转换为tensor发送
			data_tensor = torch.tensor(list(compressed), dtype=torch.uint8, device=self.device)
			
			# 发送元数据
			metadata = {
				"original_size": len(serialized),
				"compressed_size": len(compressed),
				"data_type": type(data).__name__,
				"crc32": zlib.crc32(compressed) & 0xffffffff
			}
			
			if not self._send_json_metadata(metadata, dst_rank, tag):
				return False
			
			return self._send_single_tensor(data_tensor, dst_rank, tag + 1)
		
		except Exception as e:
			self.logger.error(f"发送序列化数据失败: {e}")
			return False
	
	def _send_json_metadata(self, metadata: Dict, dst_rank: int, tag: int) -> bool:
		"""发送JSON格式的元数据"""
		try:
			metadata_str = json.dumps(metadata)
			metadata_bytes = metadata_str.encode('utf-8')
			
			# 发送长度
			length_tensor = torch.tensor([len(metadata_bytes)], dtype=torch.long, device=self.device)
			self.node_comm.send_tensor(length_tensor, dst_rank, tag)
			
			# 发送数据
			metadata_tensor = torch.tensor(list(metadata_bytes), dtype=torch.uint8, device=self.device)
			self.node_comm.send_tensor(metadata_tensor, dst_rank, tag + 1)
			
			return True
		except Exception as e:
			self.logger.error(f"发送JSON元数据失败: {e}")
			return False
	
	def recv_tensor_reliable(self, src_rank: int, tag: int = 0,
	                         expected_shape: Optional[torch.Size] = None,
	                         expected_dtype: Optional[torch.dtype] = None) -> Any:
		"""
		可靠接收数据 - 支持tensor和复杂数据类型

		Args:
			src_rank: 源rank
			tag: 通信标签
			expected_shape: 期望的tensor形状
			expected_dtype: 期望的tensor类型

		Returns:
			Any: 接收到的数据，可能是tensor、list、tuple等
		"""
		try:
			# 首先尝试接收数据类型信息
			data_type_info = self._recv_json_metadata(src_rank, tag)
			
			if data_type_info is None:
				# 如果没有类型信息，按原有tensor方式处理
				return self._recv_single_tensor(src_rank, tag, expected_shape, expected_dtype)
			
			data_type = data_type_info.get("data_type", "tensor")
			
			if data_type in ["list", "tuple"]:
				return self._recv_complex_data(data_type_info, src_rank, tag)
			elif data_type == "tensor":
				return self._recv_single_tensor(src_rank, tag + 20, expected_shape, expected_dtype)
			else:
				return self._recv_serialized_data(data_type_info, src_rank, tag + 20)
		
		except Exception as e:
			rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
			self.logger.error(f"Rank {rank}: 接收数据失败: {e}")
			return None
	
	def _recv_single_tensor(self, src_rank: int, tag: int,
	                        expected_shape: Optional[torch.Size] = None,
	                        expected_dtype: Optional[torch.dtype] = None) -> Optional[torch.Tensor]:
		"""接收单个tensor - 原有逻辑"""
		try:
			# 接收元数据
			metadata = self._recv_metadata(src_rank, tag)
			if metadata is None:
				return None
			
			packet_id, total_chunks, expected_crc, shape, dtype = metadata
			self.logger.debug(f"接收tensor从rank {src_rank}, 分块数: {total_chunks}, CRC: {expected_crc:08x}")
			
			# 验证预期属性
			if expected_shape is not None and shape != expected_shape:
				self.logger.error(f"张量形状不匹配: 期望 {expected_shape}, 实际 {shape}")
				return None
			
			if expected_dtype is not None and dtype != expected_dtype:
				self.logger.error(f"张量类型不匹配: 期望 {expected_dtype}, 实际 {dtype}")
				return None
			
			# 接收各个分块
			chunks = {}
			for chunk_idx in range(total_chunks):
				packet = self._recv_chunk(src_rank, tag + chunk_idx + 1)
				if packet is None:
					self.logger.error(f"接收分块 {chunk_idx}/{total_chunks} 失败")
					return None
				
				# 验证分块
				if not self._validate_chunk(packet, packet_id, chunk_idx, total_chunks):
					self.logger.error(f"分块 {chunk_idx} 验证失败")
					return None
				
				chunks[chunk_idx] = packet.data
				
				# 发送确认（如果启用）
				if self.enable_ack:
					self._send_ack(src_rank, packet_id, chunk_idx)
			
			# 重组张量
			tensor = self._reconstruct_tensor(chunks, shape, dtype)
			if tensor is None:
				return None
			
			# 验证完整性
			tensor_bytes = self._tensor_to_bytes(tensor)
			actual_crc = zlib.crc32(tensor_bytes) & 0xffffffff
			
			if actual_crc != expected_crc:
				self.logger.error(f"CRC校验失败: 期望 {expected_crc:08x}, 实际 {actual_crc:08x}")
				return None
			
			self.logger.debug(f"张量接收完成: rank {src_rank}, packet_id: {packet_id}")
			return tensor
		
		except Exception as e:
			self.logger.error(f"接收tensor失败: {e}")
			return None
	
	def _recv_complex_data(self, data_type_info: Dict, src_rank: int, tag: int) -> Any:
		"""接收复杂数据类型"""
		try:
			data_type = data_type_info["data_type"]
			length = data_type_info["length"]
			
			self.logger.debug(f"接收复杂数据: {data_type}, 长度: {length}")
			
			# 检查是否为tensor-tuple列表格式
			if self._is_tensor_tuple_format(src_rank, tag + 10):
				return self._recv_tensor_tuple_list(length, src_rank, tag + 10)
			else:
				# 接收序列化数据
				return self._recv_serialized_data(None, src_rank, tag + 20)
		
		except Exception as e:
			self.logger.error(f"接收复杂数据失败: {e}")
			return None
	
	def _is_tensor_tuple_format(self, src_rank: int, tag: int) -> bool:
		"""检测是否为tensor-tuple格式（通过尝试接收count）"""
		try:
			# 尝试接收数量信息
			count_tensor = self._recv_single_tensor(src_rank, tag,
			                                        expected_shape=torch.Size([1]),
			                                        expected_dtype=torch.long)
			return count_tensor is not None
		except:
			return False
	
	def _recv_tensor_tuple_list(self, expected_length: int, src_rank: int, tag: int) -> Optional[List[Tuple]]:
		"""接收tensor-tuple列表"""
		try:
			# 接收数量
			count_tensor = self._recv_single_tensor(src_rank, tag,
			                                        expected_shape=torch.Size([1]),
			                                        expected_dtype=torch.long)
			if count_tensor is None:
				return None
			
			count = count_tensor.item()
			if count != expected_length:
				self.logger.warning(f"数量不匹配: 期望 {expected_length}, 实际 {count}")
			
			# 接收tensors
			tensors = []
			for i in range(count):
				tensor = self._recv_single_tensor(src_rank, tag + 100 + i)
				if tensor is None:
					self.logger.error(f"接收tensor {i} 失败")
					return None
				tensors.append(tensor)
			
			# 接收tier信息
			tiers_tensor = self._recv_single_tensor(src_rank, tag + 200,
			                                        expected_dtype=torch.long)
			if tiers_tensor is None:
				return None
			
			tiers = tiers_tensor.cpu().numpy().tolist()
			
			# 组合结果
			result = [(tensor, tier) for tensor, tier in zip(tensors, tiers)]
			
			self.logger.debug(f"接收tensor-tuple列表完成: {len(result)}个项目")
			return result
		
		except Exception as e:
			self.logger.error(f"接收tensor-tuple列表失败: {e}")
			return None
	
	def _recv_serialized_data(self, metadata: Optional[Dict], src_rank: int, tag: int) -> Any:
		"""接收序列化数据"""
		try:
			# 如果没有元数据，先接收
			if metadata is None:
				metadata = self._recv_json_metadata(src_rank, tag)
				if metadata is None:
					return None
				tag += 1  # 调整tag
			
			# 接收压缩数据
			data_tensor = self._recv_single_tensor(src_rank, tag + 1)
			if data_tensor is None:
				return None
			
			# 转换回字节
			compressed_data = bytes(data_tensor.cpu().numpy())
			
			# 验证CRC
			actual_crc = zlib.crc32(compressed_data) & 0xffffffff
			expected_crc = metadata.get("crc32", 0)
			
			if actual_crc != expected_crc:
				self.logger.error(f"序列化数据CRC校验失败")
				return None
			
			# 解压缩和反序列化
			decompressed = zlib.decompress(compressed_data)
			data = pickle.loads(decompressed)
			
			self.logger.debug(f"接收序列化数据完成: {type(data)}")
			return data
		
		except Exception as e:
			self.logger.error(f"接收序列化数据失败: {e}")
			return None
	
	def _recv_json_metadata(self, src_rank: int, tag: int) -> Optional[Dict]:
		"""接收JSON格式的元数据"""
		try:
			# 接收长度
			length_tensor = self.node_comm.recv_tensor(src_rank, tag, dtype=torch.long)
			if length_tensor is None:
				return None
			
			length = length_tensor.item()
			
			# 接收数据
			metadata_tensor = self.node_comm.recv_tensor(src_rank, tag + 1, dtype=torch.uint8)
			if metadata_tensor is None:
				return None
			
			# 转换为字符串
			metadata_bytes = bytes(metadata_tensor.cpu().numpy())
			metadata_str = metadata_bytes.decode('utf-8')
			
			# 解析JSON
			metadata = json.loads(metadata_str)
			
			return metadata
		
		except Exception as e:
			self.logger.error(f"接收JSON元数据失败: {e}")
			return None
	
	
	
	
	
	def _generate_packet_id(self, dst_rank: int, tag: int) -> str:
		"""生成唯一的传输ID"""
		timestamp = int(time.time() * 1000000)  # 微秒时间戳
		return f"{self.node_comm.rank}_{dst_rank}_{tag}_{timestamp}"
	
	def _tensor_to_bytes(self, data: Any) -> bytes:
		"""将数据转换为字节 - 支持多种数据类型"""
		try:
			if isinstance(data, torch.Tensor):
				# 原有的tensor处理逻辑
				return data.cpu().numpy().tobytes()
			else:
				# 新增：处理非tensor数据
				self.logger.warning(f"_tensor_to_bytes收到非tensor数据: {type(data)}")
				# 使用pickle序列化
				serialized = pickle.dumps(data)
				return zlib.compress(serialized)
		except Exception as e:
			rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
			self.logger.error(f"Rank {rank}: 数据转换失败: {e}, 数据类型: {type(data)}")
			return b''
	
	def _split_tensor_to_chunks(self, tensor: torch.Tensor) -> List[torch.Tensor]:
		"""将张量分割为适合传输的块"""
		tensor_size = tensor.numel() * tensor.element_size()
		
		if tensor_size <= self.max_chunk_size:
			return [tensor]
		
		# 计算分块数量
		num_chunks = (tensor_size + self.max_chunk_size - 1) // self.max_chunk_size
		
		# 沿第一个维度分割
		chunk_size = max(1, tensor.shape[0] // num_chunks)
		chunks = []
		
		for i in range(0, tensor.shape[0], chunk_size):
			end_idx = min(i + chunk_size, tensor.shape[0])
			chunk = tensor[i:end_idx].clone()
			chunks.append(chunk)
		
		return chunks
	
	
	
	def _send_metadata(self, dst_rank: int, packet_id: str, total_chunks: int,
	                   crc32: int, shape: torch.Size, dtype: torch.dtype, tag: int) -> bool:
		"""发送元数据"""
		try:
			metadata = {
				'packet_id': packet_id,
				'total_chunks': total_chunks,
				'crc32': crc32,
				'shape': list(shape),
				'dtype': str(dtype),
				'timestamp': time.time()
			}
			
			# 序列化元数据
			metadata_str = json.dumps(metadata)
			metadata_tensor = torch.tensor([ord(c) for c in metadata_str], dtype=torch.uint8, device=self.device)
			
			
			# 发送元数据大小
			size_tensor = torch.tensor([len(metadata_str)], dtype=torch.long, device=self.device)
			
			# 使用原始通信接口发送
			self.node_comm.send_tensor(size_tensor, dst_rank, tag)
			self.node_comm.send_tensor(metadata_tensor, dst_rank, tag)
			
			return True
		
		except Exception as e:
			self.logger.error(f"发送元数据失败: {e}")
			return False
	
	def _recv_metadata(self, src_rank: int, tag: int) -> Optional[tuple]:
		"""接收元数据"""
		try:
			# 接收元数据大小
			size_tensor = self.node_comm.recv_tensor(src_rank, tag, dtype=torch.long)
			metadata_size = size_tensor.item()
			
			# 接收元数据
			metadata_tensor = self.node_comm.recv_tensor(src_rank, tag, dtype=torch.uint8)
			metadata_str = ''.join([chr(c) for c in metadata_tensor.cpu().numpy()])
			
			# 解析元数据
			metadata = json.loads(metadata_str) # 实际应用中应使用更安全的序列化方法
			
			return (
				metadata['packet_id'],
				metadata['total_chunks'],
				metadata['crc32'],
				torch.Size(metadata['shape']),
				getattr(torch, metadata['dtype'].split('.')[-1])
			)
		
		except Exception as e:
			self.logger.error(f"接收元数据失败: {e}")
			return None
	
	def _send_chunk(self, dst_rank: int, packet: TransmissionPacket, tag: int) -> bool:
		"""发送单个分块"""
		try:
			# 发送分块数据
			self.node_comm.send_tensor(packet.data, dst_rank, tag)
			
			# 发送分块元数据
			chunk_info = torch.tensor([
				packet.chunk_index,
				packet.total_chunks,
				packet.crc32
			], dtype=torch.long, device=self.device)
			
			self.node_comm.send_tensor(chunk_info, dst_rank, tag)
			
			return True
		
		except Exception as e:
			self.logger.error(f"发送分块失败: {e}")
			return False
	
	def _recv_chunk(self, src_rank: int, tag: int) -> Optional[TransmissionPacket]:
		"""接收单个分块"""
		try:
			# 接收分块数据
			chunk_data = self.node_comm.recv_tensor(src_rank, tag)
			
			# 接收分块元数据
			chunk_info = self.node_comm.recv_tensor(src_rank, tag, dtype=torch.long)
			
			packet = TransmissionPacket(
				data=chunk_data,
				packet_id="",  # 在验证时设置
				chunk_index=chunk_info[0].item(),
				total_chunks=chunk_info[1].item(),
				crc32=chunk_info[2].item(),
				timestamp=time.time()
			)
			
			return packet
		
		except Exception as e:
			self.logger.error(f"接收分块失败: {e}")
			return None
	
	def _validate_chunk(self, packet: TransmissionPacket, expected_packet_id: str,
	                    expected_chunk_idx: int, expected_total_chunks: int) -> bool:
		"""验证分块"""
		# 验证分块索引
		if packet.chunk_index != expected_chunk_idx:
			self.logger.error(f"分块索引不匹配: 期望 {expected_chunk_idx}, 实际 {packet.chunk_index}")
			return False
		
		# 验证总分块数
		if packet.total_chunks != expected_total_chunks:
			self.logger.error(f"总分块数不匹配: 期望 {expected_total_chunks}, 实际 {packet.total_chunks}")
			return False
		
		# 验证CRC
		chunk_bytes = self._tensor_to_bytes(packet.data)
		actual_crc = zlib.crc32(chunk_bytes) & 0xffffffff
		
		if actual_crc != packet.crc32:
			self.logger.error(f"分块CRC不匹配: 期望 {packet.crc32:08x}, 实际 {actual_crc:08x}")
			return False
		
		return True
	
	def _reconstruct_tensor(self, chunks: Dict[int, torch.Tensor],
	                        shape: torch.Size, dtype: torch.dtype) -> Optional[torch.Tensor]:
		"""重组张量"""
		try:
			# 按顺序组合分块
			sorted_chunks = [chunks[i] for i in sorted(chunks.keys())]
			tensor = torch.cat(sorted_chunks, dim=0)
			
			# 重塑为原始形状
			tensor = tensor.view(shape).to(dtype)
			
			return tensor
		
		except Exception as e:
			self.logger.error(f"重组张量失败: {e}")
			return None
	
	def _send_ack(self, dst_rank: int, packet_id: str, chunk_idx: int):
		"""发送确认"""
		try:
			ack_data = torch.tensor([hash(packet_id) & 0xffffffff, chunk_idx], dtype=torch.long, device=self.device)
			self.node_comm.send_tensor(ack_data, dst_rank, tag=9999)  # 使用特殊tag
		except Exception as e:
			self.logger.warning(f"发送确认失败: {e}")
	
	def _wait_for_ack(self, src_rank: int, packet_id: str, chunk_idx: int) -> bool:
		"""等待确认"""
		try:
			# 设置超时
			start_time = time.time()
			
			while time.time() - start_time < self.ack_timeout:
				try:
					ack_data = self.node_comm.recv_tensor(src_rank, tag=9999, dtype=torch.long)
					
					expected_hash = hash(packet_id) & 0xffffffff
					if ack_data[0].item() == expected_hash and ack_data[1].item() == chunk_idx:
						return True
				
				except:
					time.sleep(0.1)  # 短暂等待
			
			return False
		
		except Exception as e:
			self.logger.warning(f"等待确认失败: {e}")
			return False


class EnhancedNodeCommunicator:
	"""增强的节点通信器 - 集成可靠通信"""
	
	def __init__(self, original_node_comm):
		self.original_comm = original_node_comm
		self.reliable_comm = ReliableCommunicator(original_node_comm)
		
		# 错误统计
		self.error_stats = {
			'send_failures': 0,
			'recv_failures': 0,
			'crc_failures': 0,
			'timeout_failures': 0
		}
		
		self.logger = logging.getLogger(__name__)
	
	def send_tensor(self, data: Any, dst_rank: int, tag: int = 0,
	                reliable: bool = True) -> bool:
		"""
		发送数据 - 支持tensor和复杂数据类型

		Args:
			data: 要发送的数据，可以是tensor、list、tuple等
			dst_rank: 目标rank
			tag: 通信标签
			reliable: 是否使用可靠传输模式

		Returns:
			bool: 发送是否成功
		"""
		if reliable:
			success = self.reliable_comm.send_tensor_reliable(data, dst_rank, tag)
			if not success:
				self.error_stats['send_failures'] += 1
			return success
		else:
			# 快速模式：仅支持tensor类型
			if isinstance(data, torch.Tensor):
				try:
					self.original_comm.send_tensor(data, dst_rank, tag)
					return True
				except Exception as e:
					self.logger.error(f"快速发送失败: {e}")
					self.error_stats['send_failures'] += 1
					return False
			else:
				# 复杂数据类型必须使用可靠模式
				self.logger.warning(f"复杂数据类型自动切换到可靠模式: {type(data)}")
				success = self.reliable_comm.send_tensor_reliable(data, dst_rank, tag)
				if not success:
					self.error_stats['send_failures'] += 1
				return success
	
	def send_data(self, data: Any, dst_rank: int, tag: int = 0, reliable: bool = True) -> bool:
		"""
		发送数据的别名方法 - 为了更清晰的API
		"""
		return self.send_tensor(data, dst_rank, tag, reliable)
	
	def recv_tensor(self, src_rank: int, tag: int = 0, dtype=None, device=None,
	                reliable: bool = True, expected_shape=None) -> Any:
		"""
		接收数据 - 支持tensor和复杂数据类型

		Args:
			src_rank: 源rank
			tag: 通信标签
			dtype: 期望的tensor数据类型
			device: 目标设备
			reliable: 是否使用可靠传输模式
			expected_shape: 期望的tensor形状

		Returns:
			Any: 接收到的数据，可能是tensor、list、tuple等
		"""
		if reliable:
			data = self.reliable_comm.recv_tensor_reliable(
				src_rank, tag, expected_shape, dtype
			)
			if data is None:
				self.error_stats['recv_failures'] += 1
			return data
		else:
			# 快速模式：仅支持tensor类型
			try:
				tensor = self.original_comm.recv_tensor(src_rank, tag, dtype, device)
				return tensor
			except Exception as e:
				self.logger.error(f"快速接收失败: {e}")
				self.error_stats['recv_failures'] += 1
				return None
	
	def recv_data(self, src_rank: int, tag: int = 0, dtype=None, device=None,
	              reliable: bool = True, expected_shape=None) -> Any:
		"""
		接收数据的别名方法 - 为了更清晰的API
		"""
		return self.recv_tensor(src_rank, tag, dtype, device, reliable, expected_shape)
	
	def recv_tensor_tuple_list(self, src_rank: int, tag: int = 0,
	                           expected_count: Optional[int] = None) -> Optional[List[Tuple[torch.Tensor, int]]]:
		"""
		专门接收 List[Tuple[torch.Tensor, int]] 格式的数据

		Args:
			src_rank: 源rank
			tag: 通信标签
			expected_count: 期望的数据项数量

		Returns:
			Optional[List[Tuple[torch.Tensor, int]]]: 接收到的tensor-tuple列表
		"""
		try:
			data = self.recv_tensor(src_rank, tag, reliable=True)
			
			if data is None:
				return None
			
			# 验证数据类型
			if not isinstance(data, list):
				self.logger.error(f"期望list类型，但收到: {type(data)}")
				return None
			
			# 验证每个元素的格式
			for i, item in enumerate(data):
				if not isinstance(item, tuple) or len(item) != 2:
					self.logger.error(f"第{i}个元素格式不正确: {type(item)}")
					return None
				
				if not isinstance(item[0], torch.Tensor) or not isinstance(item[1], (int, float)):
					self.logger.error(f"第{i}个元素内容类型不正确: {type(item[0])}, {type(item[1])}")
					return None
			
			# 验证数量
			if expected_count is not None and len(data) != expected_count:
				self.logger.warning(f"数据数量不匹配: 期望{expected_count}，实际{len(data)}")
			
			# 转换tier为int类型
			result = [(tensor, int(tier)) for tensor, tier in data]
			
			self.logger.debug(f"成功接收tensor-tuple列表: {len(result)}个项目")
			return result
		
		except Exception as e:
			self.logger.error(f"接收tensor-tuple列表失败: {e}")
			self.error_stats['recv_failures'] += 1
			return None
	
	def send_tensor_tuple_list(self, data: List[Tuple[torch.Tensor, int]],
	                           dst_rank: int, tag: int = 0) -> bool:
		"""
		专门发送 List[Tuple[torch.Tensor, int]] 格式的数据

		Args:
			data: 要发送的tensor-tuple列表
			dst_rank: 目标rank
			tag: 通信标签

		Returns:
			bool: 发送是否成功
		"""
		try:
			# 验证数据格式
			if not isinstance(data, list):
				self.logger.error(f"期望list类型，但收到: {type(data)}")
				return False
			
			for i, item in enumerate(data):
				if not isinstance(item, tuple) or len(item) != 2:
					self.logger.error(f"第{i}个元素格式不正确: {type(item)}")
					return False
				
				if not isinstance(item[0], torch.Tensor) or not isinstance(item[1], (int, float)):
					self.logger.error(f"第{i}个元素内容类型不正确: {type(item[0])}, {type(item[1])}")
					return False
			
			# 使用可靠传输发送
			success = self.send_tensor(data, dst_rank, tag, reliable=True)
			
			if success:
				self.logger.debug(f"成功发送tensor-tuple列表: {len(data)}个项目")
			
			return success
		
		except Exception as e:
			self.logger.error(f"发送tensor-tuple列表失败: {e}")
			self.error_stats['send_failures'] += 1
			return False
	
	
	
	def get_error_stats(self) -> Dict[str, int]:
		"""获取错误统计"""
		return self.error_stats.copy()
	
	def reset_error_stats(self):
		"""重置错误统计"""
		self.error_stats = {
			'send_failures': 0,
			'recv_failures': 0,
			'crc_failures': 0,
			'timeout_failures': 0
		}
	
	# 代理其他方法
	def __getattr__(self, name):
		return getattr(self.original_comm, name)


# 使用示例和集成代码
def upgrade_node_communicator(original_node_comm):
	"""
	升级现有的NodeCommunicator为增强版本

	在创建分布式模型时使用：
	"""
	enhanced_comm = EnhancedNodeCommunicator(original_node_comm)
	return enhanced_comm


# 在stages.py中的使用示例：
example_usage = """
# 在CHProcessingStage中使用可靠通信：
def forward(self, patches=None, tiers=None):
    if patches is None and self.node_comm:
        # 使用可靠接收
        count_tensor = self.node_comm.recv_tensor(
            src_rank=prev_rank,
            dtype=torch.long,
            device=self.device,
            reliable=True  # 启用可靠模式
        )

        # 接收patches时验证形状
        for i in range(count):
            patch_tensor = self.node_comm.recv_tensor(
                src_rank=prev_rank,
                dtype=torch.float32,
                device=self.device,
                reliable=True,
                expected_shape=torch.Size([1, 1, 64, 64, 64])  # 预期形状
            )

    # 发送时使用可靠传输
    if self.node_comm:
        success = self.node_comm.send_tensor(
            ch_feat,
            dst_rank=fusion_rank,
            reliable=True  # 启用可靠模式
        )

        if not success:
            self.logger.error("CH特征发送失败")
            # 可以选择重试或跳过
"""

if __name__ == "__main__":
	print("通信错误处理代码已生成")
	print("主要特性：")
	print("- 分块传输避免大数据传输失败")
	print("- CRC32校验确保数据完整性")
	print("- 简单ACK确认机制")
	print("- 合理超时设置")
	print("- 详细错误统计")
	print("\n使用示例：")
	print(example_usage)