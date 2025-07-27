# pipeline/stages/__init__.py
"""
Stage 体系基类
所有 Stage 只做张量或 numpy 运算，不调用 dist.*
"""

from typing import Dict, Any, Iterable, Union
import torch
import logging
from ..message import Message

logger = logging.getLogger(__name__)


class StageBase:
	"""Stage基类"""
	
	def __init__(self, cfg: Dict[str, Any], device: torch.device):
		"""
		初始化Stage

		Args:
			cfg: 配置字典
			device: 计算设备
		"""
		self.cfg = cfg
		self.device = device
		self.stage_name = self.__class__.__name__
		
		logger.info(f"Initialized {self.stage_name} on {device}")
	
	def process(self, obj: Union[Dict, Message]) -> Iterable[Message]:
		"""
		Stage核心处理逻辑

		Args:
			obj: 输入对象，可以是 batch(Dict) 或 Message

		Yields:
			Message: 输出消息
		"""
		raise NotImplementedError(f"{self.stage_name}.process() not implemented")
	
	def __repr__(self):
		return f"{self.stage_name}(device={self.device})"


# 导入所有具体Stage实现
from .preprocess import Preprocess
from .patch_dispatch import PatchDispatch
from .ch_branch import CHBranch
from .spatial_branch import SpatialBranch
from .feature_fuse import FeatureFuse
from .multiscale import Multiscale
from .seg_head import SegHead

__all__ = [
	'StageBase',
	'Preprocess',
	'PatchDispatch',
	'CHBranch',
	'SpatialBranch',
	'FeatureFuse',
	'Multiscale',
	'SegHead'
]