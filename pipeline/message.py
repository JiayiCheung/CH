# pipeline/message.py
"""
流水线消息类 - 支持设备管理和NCCL直传
"""

import torch
from dataclasses import dataclass
from typing import Dict, Any, Optional


def _map_tensor(obj, fn):
	"""递归搬运工具函数"""
	if isinstance(obj, torch.Tensor):
		return fn(obj)
	elif isinstance(obj, dict):
		return {k: _map_tensor(v, fn) for k, v in obj.items()}
	elif isinstance(obj, list):
		return [_map_tensor(x, fn) for x in obj]
	else:
		return obj


@dataclass
class Message:
	"""流水线统一消息格式 + 设备标记"""
	kind: str  # e.g. 'patch', 'ch_feature', ...
	payload: Dict[str, Any]
	device: Optional[torch.device] = None  # 新增：整体数据所在设备
	
	# ---------- 设备搬运 ----------
	def cuda_(self, non_blocking=True):
		"""将消息内所有tensor搬到CUDA"""
		if self.device and self.device.type == 'cuda':
			return self
		self.payload = _map_tensor(self.payload, lambda t: t.cuda(non_blocking=non_blocking))
		self.device = torch.device('cuda')
		return self
	
	def cpu_(self):
		"""将消息内所有tensor搬到CPU"""
		if self.device is None or self.device.type == 'cpu':
			return self
		self.payload = _map_tensor(self.payload, lambda t: t.cpu())
		self.device = torch.device('cpu')
		return self
	
	def to(self, device: torch.device):
		"""简化成包装调用"""
		return self.cuda_() if device.type == 'cuda' else self.cpu_()
	
	# ---------- Pickle 辅助 ----------
	def to_dict(self):
		"""序列化到纯 CPU 基础类型，便于 send_object"""
		return {
			'kind': self.kind,
			'payload': _map_tensor(self.cpu_().payload, lambda t: t.cpu()),
		}
	
	@staticmethod
	def from_dict(d):
		return Message(kind=d['kind'], payload=d['payload'], device=torch.device('cpu'))
	
	def __repr__(self):
		tensor_info = []
		
		def collect_tensor_info(obj, path=""):
			if isinstance(obj, torch.Tensor):
				tensor_info.append(f"{path}:{list(obj.shape)}")
			elif isinstance(obj, dict):
				for k, v in obj.items():
					collect_tensor_info(v, f"{path}.{k}" if path else k)
			elif isinstance(obj, list):
				for i, v in enumerate(obj):
					collect_tensor_info(v, f"{path}[{i}]" if path else f"[{i}]")
		
		collect_tensor_info(self.payload)
		return f"Message(kind={self.kind}, device={self.device}, tensors=[{', '.join(tensor_info)}])"
