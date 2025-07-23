#communication_reliability.py

"""
Frame-Based Reliable Communication for Multi-Node Pipeline Training
基于帧协议的可靠通信实现
"""

import torch
import torch.distributed as dist
import pickle
import zlib
import struct
import logging
import time
import threading
from typing import Any, List, Tuple, Optional, Union
import os
import json

# Frame protocol constants
FRAME_MAGIC = b'\xAB\xCD'  # 2 bytes magic number
FRAME_VERSION = 1  # 1 byte version
FRAME_HEADER_SIZE = 15  # MAGIC(2) + VER(1) + LEN(4) + TYPE(1) + RANK_SRC(2) + RANK_DST(2) + TAG(2) + CRC_HEADER(1)

# Frame types
FRAME_TYPE_TENSOR = 0
FRAME_TYPE_COMPLEX = 1
FRAME_TYPE_ACK = 2

# Debug logging
DEBUG_ENABLED = os.environ.get('COMM_DEBUG', '0') == '1'


class FrameLogger:
	"""Frame protocol debug logger"""
	
	def __init__(self, name: str):
		self.logger = logging.getLogger(f"frame_comm.{name}")
		self.enabled = DEBUG_ENABLED
	
	def debug_send(self, src_rank: int, dst_rank: int, tag: int, frame_type: int, data_len: int, crc: int):
		if self.enabled:
			self.logger.debug(
				f"[FRAME] SEND rank{src_rank}→{dst_rank} tag{tag} TYPE{frame_type} LEN{data_len} CRC8=0x{crc:02x}")
	
	def debug_recv(self, src_rank: int, dst_rank: int, tag: int, frame_type: int, data_len: int, success: bool):
		if self.enabled:
			status = "OK" if success else "FAIL"
			self.logger.debug(
				f"[FRAME] RECV rank{src_rank}→{dst_rank} tag{tag} TYPE{frame_type} LEN{data_len} {status}")
	
	def error(self, msg: str):
		self.logger.error(f"[FRAME] {msg}")
	
	def warning(self, msg: str):
		self.logger.warning(f"[FRAME] {msg}")


class FrameProtocol:
	"""Frame-based communication protocol implementation"""
	
	def __init__(self, max_chunk_size: int = 128 * 1024):
		self.max_chunk_size = max_chunk_size
		self.logger = FrameLogger("protocol")
	
	def _calculate_crc8(self, data: bytes) -> int:
		"""Simple CRC8 calculation"""
		crc = 0
		for byte in data:
			crc ^= byte
			for _ in range(8):
				if crc & 0x80:
					crc = (crc << 1) ^ 0x07
				else:
					crc <<= 1
				crc &= 0xFF
		return crc
	
	def _create_frame_header(self, frame_type: int, data_len: int, src_rank: int, dst_rank: int, tag: int) -> bytes:
		"""Create frame header: MAGIC(2) + VER(1) + LEN(4) + TYPE(1) + SRC(2) + DST(2) + TAG(2) + CRC_HEADER(1)"""
		header_data = struct.pack('>2sBIBHHH',
		                          FRAME_MAGIC,
		                          FRAME_VERSION,
		                          data_len,
		                          frame_type,
		                          src_rank,
		                          dst_rank,
		                          tag)
		header_crc = self._calculate_crc8(header_data)
		return header_data + struct.pack('B', header_crc)
	
	def _parse_frame_header(self, header_bytes: bytes) -> Tuple[int, int, int, int, int, bool]:
		"""Parse frame header, returns (frame_type, data_len, src_rank, dst_rank, tag, valid)"""
		if len(header_bytes) != FRAME_HEADER_SIZE:
			return 0, 0, 0, 0, 0, False
		
		try:
			magic, version, data_len, frame_type, src_rank, dst_rank, tag, header_crc = struct.unpack('>2sBIBHHHB',
			                                                                                          header_bytes)
			
			# Verify magic and version
			if magic != FRAME_MAGIC or version != FRAME_VERSION:
				return 0, 0, 0, 0, 0, False
			
			# Verify header CRC
			header_data = header_bytes[:-1]
			calculated_crc = self._calculate_crc8(header_data)
			if calculated_crc != header_crc:
				return 0, 0, 0, 0, 0, False
			
			return frame_type, data_len, src_rank, dst_rank, tag, True
		
		except struct.error:
			return 0, 0, 0, 0, 0, False
	
	def _serialize_tensor(self, tensor: torch.Tensor) -> bytes:
		"""Serialize tensor to bytes"""
		# Save tensor metadata and data
		buffer = {
			'shape': list(tensor.shape),
			'dtype': str(tensor.dtype),
			'device': str(tensor.device),
			'data': tensor.cpu().numpy().tobytes() if tensor.numel() > 0 else b''
		}
		return pickle.dumps(buffer)
	
	def _deserialize_tensor(self, data: bytes, target_device: str) -> torch.Tensor:
		"""Deserialize bytes to tensor"""
		buffer = pickle.loads(data)
		
		# Recreate tensor
		import numpy as np
		dtype_map = {
			'torch.float32': np.float32,
			'torch.float64': np.float64,
			'torch.int32': np.int32,
			'torch.int64': np.int64,
			'torch.uint8': np.uint8,
			'torch.bool': np.bool_,
		}
		
		np_dtype = dtype_map.get(buffer['dtype'], np.float32)
		if buffer['data']:
			np_array = np.frombuffer(buffer['data'], dtype=np_dtype).reshape(buffer['shape'])
			tensor = torch.from_numpy(np_array.copy())
		else:
			tensor = torch.empty(buffer['shape'])
		
		return tensor.to(target_device)
	
	def send_frame(self, obj: Any, dst_rank: int, tag: int, reliable: bool = True,
	               frame_type: Optional[int] = None) -> bool:
		"""Send object using frame protocol"""
		try:
			src_rank = dist.get_rank()
			
			# Determine frame type and serialize data
			if frame_type is None:
				if isinstance(obj, torch.Tensor):
					frame_type = FRAME_TYPE_TENSOR
					data = self._serialize_tensor(obj)
				else:
					frame_type = FRAME_TYPE_COMPLEX
					data = zlib.compress(pickle.dumps(obj))
			else:
				if frame_type == FRAME_TYPE_TENSOR:
					data = self._serialize_tensor(obj)
				else:
					data = zlib.compress(pickle.dumps(obj))
			
			data_len = len(data)
			data_crc = self._calculate_crc8(data)
			
			# Create frame header
			header = self._create_frame_header(frame_type, data_len, src_rank, dst_rank, tag)
			
			# Send in chunks if data is large
			if data_len <= self.max_chunk_size:
				# Send header + data as single message
				frame_data = header + data + struct.pack('I', data_crc)
				frame_tensor = torch.frombuffer(frame_data, dtype=torch.uint8)
				dist.send(frame_tensor, dst=dst_rank, tag=tag)
				
				self.logger.debug_send(src_rank, dst_rank, tag, frame_type, data_len, data_crc & 0xFF)
			else:
				# Send header first
				header_tensor = torch.frombuffer(header, dtype=torch.uint8)
				dist.send(header_tensor, dst=dst_rank, tag=tag)
				
				# Send data in chunks
				chunks_sent = 0
				for i in range(0, data_len, self.max_chunk_size):
					chunk = data[i:i + self.max_chunk_size]
					chunk_tensor = torch.frombuffer(chunk, dtype=torch.uint8)
					dist.send(chunk_tensor, dst=dst_rank, tag=tag + 1 + chunks_sent)
					chunks_sent += 1
				
				# Send CRC
				crc_tensor = torch.tensor([data_crc], dtype=torch.uint32)
				dist.send(crc_tensor, dst=dst_rank, tag=tag + 1 + chunks_sent)
				
				self.logger.debug_send(src_rank, dst_rank, tag, frame_type, data_len, data_crc & 0xFF)
			
			# Handle ACK if reliable mode
			if reliable:
				try:
					ack_tensor = torch.zeros(1, dtype=torch.uint8)
					dist.recv(ack_tensor, src=dst_rank, tag=tag + 1000)  # Use high tag for ACK
					return ack_tensor.item() == 1
				except:
					return False
			
			return True
		
		except Exception as e:
			self.logger.error(f"Send frame failed: {e}")
			return False
	
	def recv_frame(self, src_rank: int, tag: int, device: str = "cuda", reliable: bool = True) -> Tuple[Any, bool]:
		"""Receive object using frame protocol"""
		try:
			current_rank = dist.get_rank()
			
			# Receive header first
			header_tensor = torch.zeros(FRAME_HEADER_SIZE, dtype=torch.uint8)
			dist.recv(header_tensor, src=src_rank, tag=tag)
			header_bytes = header_tensor.numpy().tobytes()
			
			# Parse header
			frame_type, data_len, sender_rank, receiver_rank, frame_tag, header_valid = self._parse_frame_header(
				header_bytes)
			
			if not header_valid:
				self.logger.error(f"Invalid frame header from rank {src_rank}")
				if reliable:
					# Send NACK
					nack_tensor = torch.tensor([0], dtype=torch.uint8)
					dist.send(nack_tensor, dst=src_rank, tag=tag + 1000)
				return None, False
			
			# Receive data
			if data_len == 0:
				data = b''
				data_crc = 0
			elif data_len <= self.max_chunk_size - FRAME_HEADER_SIZE - 4:  # Single message
				# Receive remaining data + CRC
				remaining_len = data_len + 4  # data + CRC
				remaining_tensor = torch.zeros(remaining_len, dtype=torch.uint8)
				dist.recv(remaining_tensor, src=src_rank, tag=tag)
				remaining_bytes = remaining_tensor.numpy().tobytes()
				data = remaining_bytes[:-4]
				data_crc = struct.unpack('I', remaining_bytes[-4:])[0]
			else:
				# Receive data in chunks
				data = b''
				chunks_received = 0
				while len(data) < data_len:
					chunk_size = min(self.max_chunk_size, data_len - len(data))
					chunk_tensor = torch.zeros(chunk_size, dtype=torch.uint8)
					dist.recv(chunk_tensor, src=src_rank, tag=tag + 1 + chunks_received)
					data += chunk_tensor.numpy().tobytes()
					chunks_received += 1
				
				# Receive CRC
				crc_tensor = torch.zeros(1, dtype=torch.uint32)
				dist.recv(crc_tensor, src=src_rank, tag=tag + 1 + chunks_received)
				data_crc = crc_tensor.item()
			
			# Verify data CRC
			if data_len > 0:
				calculated_crc = self._calculate_crc8(data)
				if calculated_crc != (data_crc & 0xFF):
					self.logger.error(f"CRC mismatch: expected {data_crc & 0xFF:02x}, got {calculated_crc:02x}")
					if reliable:
						nack_tensor = torch.tensor([0], dtype=torch.uint8)
						dist.send(nack_tensor, dst=src_rank, tag=tag + 1000)
					return None, False
			
			# Deserialize object
			if frame_type == FRAME_TYPE_TENSOR:
				obj = self._deserialize_tensor(data, device)
			else:
				obj = pickle.loads(zlib.decompress(data))
			
			# Send ACK if reliable mode
			if reliable:
				ack_tensor = torch.tensor([1], dtype=torch.uint8)
				dist.send(ack_tensor, dst=src_rank, tag=tag + 1000)
			
			self.logger.debug_recv(src_rank, current_rank, tag, frame_type, data_len, True)
			return obj, True
		
		except Exception as e:
			self.logger.error(f"Recv frame failed: {e}")
			if reliable:
				try:
					nack_tensor = torch.tensor([0], dtype=torch.uint8)
					dist.send(nack_tensor, dst=src_rank, tag=tag + 1000)
				except:
					pass
			return None, False


class ReliableCommunicator:
	"""Frame-based reliable communicator - maintains same interface as original"""
	
	def __init__(self, node_comm, max_retries: int = 3, timeout_ms: int = 30000):
		self.node_comm = node_comm  # Keep reference but don't use for internal calls
		self.protocol = FrameProtocol()
		self.max_retries = max_retries
		self.timeout_ms = timeout_ms
		self.logger = FrameLogger("reliable")
		
		# Statistics
		self.send_attempts = 0
		self.send_failures = 0
		self.recv_attempts = 0
		self.recv_failures = 0
		self.crc_failures = 0
	
	def send_tensor_reliable(self, tensor: torch.Tensor, dst_rank: int, tag: int, dtype: torch.dtype = None) -> bool:
		"""Send tensor reliably using frame protocol"""
		self.send_attempts += 1
		
		# Convert dtype if needed
		if dtype is not None and tensor.dtype != dtype:
			tensor = tensor.to(dtype)
		
		for attempt in range(self.max_retries):
			if self.protocol.send_frame(tensor, dst_rank, tag, reliable=True, frame_type=FRAME_TYPE_TENSOR):
				return True
			
			if attempt < self.max_retries - 1:
				time.sleep(0.1 * (attempt + 1))  # Exponential backoff
		
		self.send_failures += 1
		return False
	
	def recv_tensor_reliable(self, src_rank: int, tag: int, dtype: torch.dtype = None, device: str = "cuda") -> \
	Optional[torch.Tensor]:
		"""Receive tensor reliably using frame protocol"""
		self.recv_attempts += 1
		
		obj, success = self.protocol.recv_frame(src_rank, tag, device, reliable=True)
		
		if not success:
			self.recv_failures += 1
			return None
		
		if not isinstance(obj, torch.Tensor):
			self.logger.error(f"Expected tensor, got {type(obj)}")
			self.recv_failures += 1
			return None
		
		# Convert dtype if needed
		if dtype is not None and obj.dtype != dtype:
			obj = obj.to(dtype)
		
		return obj
	
	def send_data_reliable(self, data: Any, dst_rank: int, tag: int) -> bool:
		"""Send complex data reliably using frame protocol"""
		self.send_attempts += 1
		
		for attempt in range(self.max_retries):
			if self.protocol.send_frame(data, dst_rank, tag, reliable=True, frame_type=FRAME_TYPE_COMPLEX):
				return True
			
			if attempt < self.max_retries - 1:
				time.sleep(0.1 * (attempt + 1))
		
		self.send_failures += 1
		return False
	
	def recv_data_reliable(self, src_rank: int, tag: int) -> Tuple[Any, bool]:
		"""Receive complex data reliably using frame protocol"""
		self.recv_attempts += 1
		
		obj, success = self.protocol.recv_frame(src_rank, tag, reliable=True)
		
		if not success:
			self.recv_failures += 1
			return None, False
		
		return obj, True
	
	def get_stats(self) -> dict:
		"""Get communication statistics"""
		return {
			'send_attempts': self.send_attempts,
			'send_failures': self.send_failures,
			'recv_attempts': self.recv_attempts,
			'recv_failures': self.recv_failures,
			'crc_failures': self.crc_failures,
			'success_rate': (self.send_attempts - self.send_failures) / max(1, self.send_attempts)
		}


class EnhancedNodeCommunicator:
	"""Enhanced node communicator with frame-based reliability - maintains exact same interface"""
	
	def __init__(self, base_comm, enable_reliability: bool = True):
		self.base_comm = base_comm
		self.enable_reliability = enable_reliability
		
		# Initialize frame protocol and reliable communicator
		self.protocol = FrameProtocol()
		self.reliable_comm = ReliableCommunicator(self) if enable_reliability else None
		
		self.logger = FrameLogger("enhanced")
		
		# Pass through base_comm attributes
		if hasattr(base_comm, 'rank'):
			self.rank = base_comm.rank
		if hasattr(base_comm, 'size'):
			self.size = base_comm.size
		if hasattr(base_comm, 'node_ranks'):
			self.node_ranks = base_comm.node_ranks
	
	def send_tensor(self, tensor: torch.Tensor, dst_rank: int, tag: int = 0, reliable: bool = True,
	                dtype: torch.dtype = None) -> bool:
		"""Send tensor - unified interface"""
		try:
			if reliable and self.reliable_comm:
				return self.reliable_comm.send_tensor_reliable(tensor, dst_rank, tag, dtype)
			else:
				# Fast path - direct frame protocol without ACK
				if dtype is not None and tensor.dtype != dtype:
					tensor = tensor.to(dtype)
				return self.protocol.send_frame(tensor, dst_rank, tag, reliable=False, frame_type=FRAME_TYPE_TENSOR)
		except Exception as e:
			self.logger.error(f"send_tensor failed: {e}")
			return False
	
	def recv_tensor(self, src_rank: int, tag: int = 0, dtype: torch.dtype = None, device: str = "cuda",
	                reliable: bool = True) -> Optional[torch.Tensor]:
		"""Receive tensor - unified interface"""
		try:
			if reliable and self.reliable_comm:
				return self.reliable_comm.recv_tensor_reliable(src_rank, tag, dtype, device)
			else:
				# Fast path - direct frame protocol
				obj, success = self.protocol.recv_frame(src_rank, tag, device, reliable=False)
				if success and isinstance(obj, torch.Tensor):
					if dtype is not None and obj.dtype != dtype:
						obj = obj.to(dtype)
					return obj
				return None
		except Exception as e:
			self.logger.error(f"recv_tensor failed: {e}")
			return None
	
	def send_data(self, data: Any, dst_rank: int, tag: int = 0, reliable: bool = True) -> bool:
		"""Send complex data - unified interface"""
		try:
			if reliable and self.reliable_comm:
				return self.reliable_comm.send_data_reliable(data, dst_rank, tag)
			else:
				return self.protocol.send_frame(data, dst_rank, tag, reliable=False, frame_type=FRAME_TYPE_COMPLEX)
		except Exception as e:
			self.logger.error(f"send_data failed: {e}")
			return False
	
	def recv_data(self, src_rank: int, tag: int = 0, reliable: bool = True) -> Tuple[Any, bool]:
		"""Receive complex data - unified interface"""
		try:
			if reliable and self.reliable_comm:
				return self.reliable_comm.recv_data_reliable(src_rank, tag)
			else:
				return self.protocol.recv_frame(src_rank, tag, reliable=False)
		except Exception as e:
			self.logger.error(f"recv_data failed: {e}")
			return None, False
	
	def send_tensor_tuple_list(self, tensor_tuple_list: List[Tuple[torch.Tensor, int]], dst_rank: int, tag: int = 0,
	                           reliable: bool = True) -> bool:
		"""Send list of tensor-tuple pairs - maintains exact same interface"""
		return self.send_data(tensor_tuple_list, dst_rank, tag, reliable)
	
	def recv_tensor_tuple_list(self, src_rank: int, tag: int = 0, reliable: bool = True) -> Optional[
		List[Tuple[torch.Tensor, int]]]:
		"""Receive list of tensor-tuple pairs - maintains exact same interface"""
		data, success = self.recv_data(src_rank, tag, reliable)
		if success and isinstance(data, list):
			return data
		return None
	
	def get_detailed_stats(self) -> dict:
		"""Get detailed communication statistics"""
		stats = {
			'protocol_type': 'frame_based',
			'reliability_enabled': self.enable_reliability
		}
		
		if self.reliable_comm:
			stats.update(self.reliable_comm.get_stats())
		
		return stats


def create_enhanced_communicator(base_comm, enable_reliability: bool = True) -> EnhancedNodeCommunicator:
	"""Factory function to create enhanced communicator - maintains same interface"""
	return EnhancedNodeCommunicator(base_comm, enable_reliability)


# Backwards compatibility aliases
class CommunicationReliability:
	"""Backwards compatibility wrapper"""
	
	def __init__(self, node_comm):
		self.enhanced_comm = EnhancedNodeCommunicator(node_comm)
	
	def __getattr__(self, name):
		return getattr(self.enhanced_comm, name)


def upgrade_node_communicator(base_comm, enable_reliability: bool = True) -> EnhancedNodeCommunicator:
	"""
	Upgrade basic node communicator to enhanced frame-based version

	This is the main entry point used by distributed_train.py
	Maintains exact same interface as the original function
	"""
	return EnhancedNodeCommunicator(base_comm, enable_reliability)


# Alternative factory function alias
def create_reliable_communicator(base_comm, **kwargs) -> EnhancedNodeCommunicator:
	"""Alternative factory function for creating reliable communicator"""
	return upgrade_node_communicator(base_comm, **kwargs)


# For debugging and testing
if __name__ == "__main__":
	# Simple test
	print("Frame-based communication reliability module loaded")
	print(f"Debug enabled: {DEBUG_ENABLED}")
	print(f"Frame header size: {FRAME_HEADER_SIZE} bytes")
	print(f"Supported frame types: TENSOR={FRAME_TYPE_TENSOR}, COMPLEX={FRAME_TYPE_COMPLEX}, ACK={FRAME_TYPE_ACK}")
	print("Available functions:")
	print("  - upgrade_node_communicator(base_comm, enable_reliability=True)")
	print("  - create_enhanced_communicator(base_comm, enable_reliability=True)")
	print("  - create_reliable_communicator(base_comm, **kwargs)")