# pipeline/stages/multiscale.py
"""
多尺度融合Stage (GPU-5) - GPU直传版本
"""

import torch
import torch.nn as nn
from typing import Iterable, Dict, Any
import logging
import time

from .base import StageBase
from ..message import Message

logger = logging.getLogger(__name__)


class Multiscale(StageBase):
	"""多尺度融合Stage：GPU直传版本"""
	
	def __init__(self, cfg: Dict[str, Any], device: torch.device):
		super().__init__(cfg, device)
		
		from models.multiscale import MultiscaleHead
		
		multiscale_config = cfg.get('multiscale_params', {})
		self.multiscale_head = MultiscaleHead(
			tier_dims=multiscale_config.get('tier_dims', {0: 64, 1: 32, 2: 16}),
			output_dim=multiscale_config.get('output_dim', 32),
			fusion_method=multiscale_config.get('fusion_method', 'attention'),
			attention_heads=multiscale_config.get('attention_heads', 4)
		).to(device)
		
		# 触发条件
		self.required_tiers = multiscale_config.get('required_tiers', None)
		if self.required_tiers is not None:
			self.required_tiers = set(self.required_tiers)
		
		self.min_tiers_for_fusion = multiscale_config.get('min_tiers_for_fusion', 2)
		self.timeout_seconds = multiscale_config.get('timeout_seconds', 5.0)
		
		# case缓存
		self.case_cache = {}
		self.case_timestamps = {}
		
		logger.info(f"Multiscale初始化: required_tiers={self.required_tiers}")
	
	def process(self, msg: Message) -> Iterable[Message]:
		"""处理：GPU直传版本"""
		if msg.kind != 'fused':
			return
		
		try:
			case_id = msg.payload['case_id']
			tier = msg.payload['tier']
			# Already CUDA
			features = msg.payload['features']
			
			logger.debug(f"Multiscale接收: case={case_id}, tier={tier}")
			
			# 初始化case缓存
			if case_id not in self.case_cache:
				self.case_cache[case_id] = {}
				self.case_timestamps[case_id] = time.time()
			
			# 存储tier特征
			self.case_cache[case_id][tier] = {
				'features': features,
				'patch_id': msg.payload['patch_id'],
				'bbox': msg.payload.get('bbox', [])
			}
			
			# 检查触发条件
			case_entry = self.case_cache[case_id]
			should_fuse = False
			
			if self.required_tiers is not None:
				if self.required_tiers.issubset(set(case_entry.keys())):
					should_fuse = True
					logger.debug(f"required_tiers完成: {self.required_tiers}")
			else:
				elapsed = time.time() - self.case_timestamps[case_id]
				if len(case_entry) >= self.min_tiers_for_fusion:
					should_fuse = True
					logger.debug(f"{len(case_entry)} tiers ≥ {self.min_tiers_for_fusion}")
				elif elapsed > self.timeout_seconds:
					should_fuse = True
					logger.debug(f"超时 {elapsed:.1f}s")
			
			if should_fuse:
				# 多尺度融合
				tier_features = {}
				for tier_id, tier_data in case_entry.items():
					tier_features[tier_id] = tier_data['features']
				
				context = torch.no_grad() if not self.training else torch.enable_grad()
				with context:
					multiscale_features = self.multiscale_head(tier_features)
				
				# 保留CUDA
				multiscale_features_gpu = multiscale_features
				
				logger.debug(f"Multiscale输出: {multiscale_features_gpu.shape}")
				
				yield Message(
					kind='multi',
					payload={
						'features': multiscale_features_gpu,
						'case_id': case_id,
						'tier_count': len(case_entry),
						'processed_tiers': list(case_entry.keys())
					}
				)
				
				# 清理缓存
				del self.case_cache[case_id]
				del self.case_timestamps[case_id]
		
		except Exception as e:
			logger.error(f"多尺度融合失败: {e}")
