# pipeline/stages/ch_branch.py
"""
CH分支Stage (GPU-2) - GPU直传版本
"""

import torch
import torch.nn as nn
from typing import Iterable, Dict, Any
import logging

from .base import StageBase
from ..message import Message

logger = logging.getLogger(__name__)


class CHBranch(StageBase):
	"""CH分支Stage：GPU直传版本"""
	
	def __init__(self, cfg: Dict[str, Any], device: torch.device):
		super().__init__(cfg, device)
		
		from models.ch import CHBackbone
		
		ch_config = cfg.get('ch_params', {})
		self.ch_backbone = CHBackbone(
			in_channels=ch_config.get('in_channels', 1),
			max_order=ch_config.get('max_order', 3),
			radial_orders=ch_config.get('radial_orders', [2, 4, 6]),
			angular_orders=ch_config.get('angular_orders', [4, 8, 12]),
			attention_dim=ch_config.get('attention_dim', 128),
			num_heads=ch_config.get('num_heads', 8)
		).to(device)
		
		self.norm_params = cfg.get('preprocessing', {
			'lower_percentile': 0.5,
			'upper_percentile': 99.5,
			'target_range': [0, 1]
		})
		
		tier_config = cfg.get('tier_params', {})
		self.tier_params = {
			0: tier_config.get('tier_0', {'r_scale': 1.0}),
			1: tier_config.get('tier_1', {'r_scale': 0.8}),
			2: tier_config.get('tier_2', {'r_scale': 0.6})
		}
		
		logger.info(f"CHBranch初始化: ch_config={ch_config}")
	
	def _torch_normalize(self, x):
		q_lo = torch.quantile(x, self.norm_params['lower_percentile'] / 100., dim=(-3, -2, -1), keepdim=True)
		q_hi = torch.quantile(x, self.norm_params['upper_percentile'] / 100., dim=(-3, -2, -1), keepdim=True)
		x = (x - q_lo) / (q_hi - q_lo + 1e-5)
		return x.clamp(*self.norm_params['target_range'])
	
	def process(self, msg: Message) -> Iterable[Message]:
		"""处理：GPU直传版本"""
		if msg.kind != 'patch':
			return
		
		tier = msg.payload.get('tier', 0)
		if tier not in [0, 1]:  # CH分支只处理tier 0,1
			return
		
		try:
			# 已在 GPU-2
			image = msg.payload['image']  # 已在 GPU-2
			logger.debug(f"CHBranch接收patch tier={tier}, shape={image.shape}")
			
			# 换 GPU 归一化
			normalized_tensor = self._torch_normalize(image).unsqueeze(0)  # [1,1,D,H,W] CUDA
			
			# 添加batch维度
			normalized_tensor = normalized_tensor.unsqueeze(0)  # [1, 1, D, H, W]
			
			# 获取tier参数
			tier_params = self.tier_params.get(tier, {})
			r_scale = tier_params.get('r_scale', 1.0)
			
			# CHBackbone处理
			context = torch.no_grad() if not self.training else torch.enable_grad()
			with context:
				import inspect
				sig = inspect.signature(self.ch_backbone.forward)
				if 'r_scale' in sig.parameters:
					ch_features = self.ch_backbone(normalized_tensor, r_scale=r_scale)
				else:
					ch_features = self.ch_backbone(normalized_tensor)
			
			# 保留 CUDA
			ch_features_gpu = ch_features.squeeze(0)
			
			logger.debug(f"CHBranch输出: {ch_features_gpu.shape}")
			
			yield Message(
				kind='ch_feature',
				payload={
					'features': ch_features_gpu,
					'tier': tier,
					'case_id': msg.payload['case_id'],
					'patch_id': msg.payload['patch_id'],
					'bbox': msg.payload.get('bbox', [])
				}
			)
		
		except Exception as e:
			logger.error(f"CH分支处理失败 tier={tier}: {e}")

