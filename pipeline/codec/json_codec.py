# pipeline/codec/json_codec.py
"""
JSON + Tensor 编解码器
满足长度-头-体协议：flag=0 → JSON meta；flag=1 → Tensor 二进制
"""

import json
import zlib
import struct
import torch
import numpy as np
from typing import List, Iterator
from ..message import Message


class JSONCodec:
	"""JSON + Tensor 编解码器"""
	
	HEADER_FMT = '!BI'  # 1字节flag + 4字节长度
	FLAG_JSON = 0
	FLAG_TENSOR = 1
	
	@staticmethod
	def encode(msg: Message) -> List[torch.Tensor]:
		"""
		编码消息为tensor列表

		Args:
			msg: 要编码的消息

		Returns:
			List[torch.Tensor]: 编码后的tensor块列表
		"""
		chunks = []
		
		# 1. 分离 tensor 和非 tensor 数据
		meta_data = {
			'kind': msg.kind,
			'tensor_keys': [],
			'non_tensor': {}
		}
		
		tensor_data = {}
		
		for key, value in msg.payload.items():
			if isinstance(value, torch.Tensor):
				meta_data['tensor_keys'].append(key)
				tensor_data[key] = value
			else:
				meta_data['non_tensor'][key] = value
		
		# 2. 编码 JSON 元数据
		json_bytes = json.dumps(meta_data).encode('utf-8')
		
		# 可选压缩
		if len(json_bytes) > 1024:  # 大于1KB才压缩
			json_bytes = zlib.compress(json_bytes, level=1)
			meta_data['compressed'] = True
		
		# 打包 JSON 头和体
		json_header = struct.pack(JSONCodec.HEADER_FMT, JSONCodec.FLAG_JSON, len(json_bytes))
		json_chunk = torch.tensor(list(json_header + json_bytes), dtype=torch.uint8)
		chunks.append(json_chunk)
		
		# 3. 编码每个 tensor
		for key in meta_data['tensor_keys']:
			tensor = tensor_data[key]
			
			# 转换到 CPU 并序列化
			if tensor.is_cuda:
				tensor = tensor.cpu()
			
			# tensor 元信息
			tensor_meta = {
				'key': key,
				'shape': list(tensor.shape),
				'dtype': str(tensor.dtype),
			}
			
			# numpy 二进制数据
			tensor_bytes = tensor.numpy().tobytes()
			
			# 可选压缩
			if len(tensor_bytes) > 4096:  # 大于4KB才压缩
				tensor_bytes = zlib.compress(tensor_bytes, level=1)
				tensor_meta['compressed'] = True
			
			# 组合 tensor 元信息 + 数据
			tensor_meta_bytes = json.dumps(tensor_meta).encode('utf-8')
			total_length = len(tensor_meta_bytes) + 4 + len(tensor_bytes)  # meta长度(4字节) + meta + data
			
			# 打包
			tensor_header = struct.pack(JSONCodec.HEADER_FMT, JSONCodec.FLAG_TENSOR, total_length)
			meta_length = struct.pack('!I', len(tensor_meta_bytes))
			
			tensor_chunk_bytes = tensor_header + meta_length + tensor_meta_bytes + tensor_bytes
			tensor_chunk = torch.tensor(list(tensor_chunk_bytes), dtype=torch.uint8)
			chunks.append(tensor_chunk)
		
		return chunks
	
	@staticmethod
	def decode(chunk_iter: Iterator[torch.Tensor]) -> Message:
		"""
		从tensor块列表解码消息

		Args:
			chunk_iter: tensor块迭代器

		Returns:
			Message: 解码后的消息
		"""
		# 1. 读取第一个chunk (JSON 元数据)
		first_chunk = next(chunk_iter)
		chunk_bytes = bytes(first_chunk.cpu().numpy())
		
		# 解析头部
		flag, length = struct.unpack(JSONCodec.HEADER_FMT, chunk_bytes[:5])
		if flag != JSONCodec.FLAG_JSON:
			raise ValueError(f"Expected JSON flag, got {flag}")
		
		# 解析 JSON 数据
		json_bytes = chunk_bytes[5:5 + length]
		
		# 检查是否压缩
		try:
			# 先尝试解压
			json_bytes = zlib.decompress(json_bytes)
		except zlib.error:
			# 如果解压失败，说明没有压缩
			pass
		
		meta_data = json.loads(json_bytes.decode('utf-8'))
		
		# 2. 重建消息
		message = Message(
			kind=meta_data['kind'],
			payload=meta_data['non_tensor'].copy()
		)
		
		# 3. 解码每个 tensor
		for tensor_key in meta_data['tensor_keys']:
			tensor_chunk = next(chunk_iter)
			tensor_bytes = bytes(tensor_chunk.cpu().numpy())
			
			# 解析 tensor 头部
			flag, total_length = struct.unpack(JSONCodec.HEADER_FMT, tensor_bytes[:5])
			if flag != JSONCodec.FLAG_TENSOR:
				raise ValueError(f"Expected tensor flag, got {flag}")
			
			# 读取 tensor 元信息长度
			meta_length = struct.unpack('!I', tensor_bytes[5:9])[0]
			
			# 读取 tensor 元信息
			tensor_meta_bytes = tensor_bytes[9:9 + meta_length]
			tensor_meta = json.loads(tensor_meta_bytes.decode('utf-8'))
			
			# 读取 tensor 数据
			data_start = 9 + meta_length
			tensor_data_bytes = tensor_bytes[data_start:data_start + (total_length - 4 - meta_length)]
			
			# 检查是否压缩
			if tensor_meta.get('compressed', False):
				tensor_data_bytes = zlib.decompress(tensor_data_bytes)
			
			# 重建 tensor
			dtype_name = tensor_meta['dtype'].split('.')[-1]  # 'torch.float32' -> 'float32'
			numpy_dtype = getattr(np, dtype_name) if hasattr(np, dtype_name) else np.float32
			
			tensor_array = np.frombuffer(tensor_data_bytes, dtype=numpy_dtype)
			tensor = torch.from_numpy(tensor_array).reshape(tensor_meta['shape'])
			
			# 恢复原始 dtype
			if 'float32' in tensor_meta['dtype']:
				tensor = tensor.float()
			elif 'int64' in tensor_meta['dtype']:
				tensor = tensor.long()
			elif 'int32' in tensor_meta['dtype']:
				tensor = tensor.int()
			
			message.payload[tensor_key] = tensor
		
		return message


