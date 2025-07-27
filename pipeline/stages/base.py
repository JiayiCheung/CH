# pipeline/stages/base.py
"""
Stage基类 - 解决循环导入并提供统一接口
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Iterable
from abc import ABC, abstractmethod


class StageBase(nn.Module, ABC):
	"""
	Pipeline Stage基类

	作用:
	1. 提供统一的Stage接口规范
	2. 解决循环导入问题 (各Stage都继承此基类)
	3. 封装公共的设备管理逻辑
	4. 标准化配置传递方式
	"""
	
	def __init__(self, cfg: Dict[str, Any], device: torch.device):
		super().__init__()
		self.cfg = cfg
		self.device = device
		
		# 公共状态跟踪
		self.processed_count = 0
		self.error_count = 0
	
	@abstractmethod
	def process(self, msg) -> Iterable:
		"""
		抽象方法：每个具体Stage必须实现

		Args:
			msg: Message对象

		Returns:
			Iterable[Message]: 输出消息迭代器
		"""
		raise NotImplementedError("子类必须实现process方法")
	
	def forward(self, *args, **kwargs):
		"""
		nn.Module兼容接口
		默认调用process方法
		"""
		if len(args) == 1:
			return self.process(args[0])
		return self.process(*args, **kwargs)
	
	def get_stats(self) -> Dict[str, int]:
		"""获取Stage统计信息"""
		return {
			'processed_count': self.processed_count,
			'error_count': self.error_count
		}
	
	def reset_stats(self):
		"""重置统计信息"""
		self.processed_count = 0
		self.error_count = 0


