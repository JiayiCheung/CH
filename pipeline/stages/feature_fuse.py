# pipeline/stages/feature_fuse.py
"""
特征融合Stage (GPU-4) - GPU直传版本
"""

import torch
import torch.nn as nn
from typing import Iterable, Dict, Any
import logging
import time

from .base import StageBase
from ..message import Message

logger = logging.getLogger(__name__)


class Fusion(StageBase):
	"""特征融合Stage：GPU直传版本"""
	
	def __init__(self, cfg: Dict[str, Any], device: torch.device):
		super().__init__(cfg, device)
		
		from models.fusion import AttentionFusion
		
		fusion_config = cfg.get('fusion_params', {})
		self.attention_fusion = AttentionFusion(
			ch_dim=fusion_config.get('ch_dim', 64),
			spatial_dim=fusion_config.get('spatial_dim', 32),
			output_dim=fusion_config.get('output_dim', 64),
			num_heads=fusion_config.get('num_heads', 8),
			dropout=fusion_config.get('dropout', 0.1)
		).to(device)
		
		# 缓存特征等待配对
		self.feature_cache = {}
		self.timeout_seconds = fusion_config.get('timeout_seconds', 3.0)
		self.timestamps = {}
		
		logger.info(f"Fusion初始化: fusion_config={fusion_config}")
	
	def process(self, msg: Message) -> Iterable[Message]:
		"""处理：GPU直传版本"""
		if msg.kind not in ['ch_feature', 'spatial_feature']:
			return
		
		try:
			case_id = msg.payload['case_id']
			patch_id = msg.payload['patch_id']
			tier = msg.payload['tier']
			key = f"{case_id}_{patch_id}_{tier}"
			
			# Already CUDA
			features = msg.payload['features']
			
			logger.debug(f"Fusion接收: {msg.kind}, key={key}")
			
			# 初始化缓存
			if key not in self.feature_cache:
				self.feature_cache[key] = {}
				self.timestamps[key] = time.time()
			
			# 存储特征
			self.feature_cache[key][msg.kind] = {
				'features': features,
				'payload': msg.payload
			}
			
			# 检查是否可以融合
			cache_entry = self.feature_cache[key]
			should_fuse = False
			
			if 'ch_feature' in cache_entry and 'spatial_feature' in cache_entry:
				should_fuse = True
				logger.debug(f"特征配对完成: {key}")
			elif time.time() - self.timestamps[key] > self.timeout_seconds:
				should_fuse = True
				logger.debug(f"超时融合: {key}")
			
			if should_fuse:
				# 融合特征
				ch_feat = cache_entry.get('ch_feature', {}).get('features')
				spatial_feat = cache_entry.get('spatial_feature', {}).get('features')
				
				if ch_feat is not None and spatial_feat is not None:
					# 双特征融合
					context = torch.no_grad() if not self.training else torch.enable_grad()
					with context:
						fused_features = self.attention_fusion(ch_feat, spatial_feat)
				elif ch_feat is not None:
					# 只有CH特征
					fused_features = ch_feat
				elif spatial_feat is not None:
					# 只有空间特征
					fused_features = spatial_feat
				else:
					logger.warning(f"没有可用特征: {key}")
					return
				
				# 保留CUDA
				fused_features_gpu = fused_features
				
				logger.debug(f"Fusion输出: {fused_features_gpu.shape}")
				
				yield Message(
					kind='fused',
					payload={
						'features': fused_features_gpu,
						'tier': tier,
						'case_id': case_id,
						'patch_id': patch_id,
						'bbox': msg.payload.get('bbox', [])
					}
				)
				
				# 清理缓存
				del self.feature_cache[key]
				del self.timestamps[key]
		
		except Exception as e:
			logger.error(f"特征融合失败: {e}")