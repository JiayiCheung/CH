# pipeline/stages/spatial_branch.py
"""
空间分支Stage (GPU-3) - GPU直传版本
"""

import torch
import torch.nn as nn
from typing import Iterable, Dict, Any
import logging

from .base import StageBase
from ..message import Message

logger = logging.getLogger(__name__)


class SpatialBranch(StageBase):
	"""空间分支Stage：GPU直传版本"""
	
	def __init__(self, cfg: Dict[str, Any], device: torch.device):
		super().__init__(cfg, device)
		
		from models.spatial import SpatialBackbone
		
		spatial_config = cfg.get('spatial_params', {})
		self.spatial_backbone = SpatialBackbone(
			in_channels=spatial_config.get('in_channels', 1),
			conv_channels=spatial_config.get('conv_channels', [16, 32, 16]),
			kernel_size=spatial_config.get('kernel_size', 3),
			enable_edge_enhancement=spatial_config.get('enable_edge_enhancement', True),
			edge_kernel_size=spatial_config.get('edge_kernel_size', 3)
		).to(device)
		
		self.norm_params = cfg.get('preprocessing', {
			'lower_percentile': 0.5,
			'upper_percentile': 99.5,
			'target_range': [0, 1]
		})
		
		logger.info(f"SpatialBranch初始化: spatial_config={spatial_config}")
	
	def _torch_normalize(self, x):
		q_lo = torch.quantile(x, self.norm_params['lower_percentile'] / 100., dim=(-3, -2, -1), keepdim=True)
		q_hi = torch.quantile(x, self.norm_params['upper_percentile'] / 100., dim=(-3, -2, -1), keepdim=True)
		x = (x - q_lo) / (q_hi - q_lo + 1e-5)
		return x.clamp(*self.norm_params['target_range'])
	
	def process(self, msg: Message) -> Iterable[Message]:
		"""处理：GPU直传版本"""
		if msg.kind != 'patch':
			return
		
		try:
			# 已在 GPU-3
			image = msg.payload['image']  # 已在 GPU-3
			tier = msg.payload.get('tier', 0)
			
			logger.debug(f"SpatialBranch接收patch tier={tier}, shape={image.shape}")
			
			# 换 GPU 归一化
			normalized_tensor = self._torch_normalize(image).unsqueeze(0)  # [1,1,D,H,W] CUDA
			
			# 添加batch维度
			normalized_tensor = normalized_tensor.unsqueeze(0)  # [1, 1, D, H, W]
			
			# SpatialBackbone处理
			context = torch.no_grad() if not self.training else torch.enable_grad()
			with context:
				spatial_features = self.spatial_backbone(normalized_tensor)
			
			# 保留 CUDA
			spatial_features_gpu = spatial_features.squeeze(0)
			
			logger.debug(f"SpatialBranch输出: {spatial_features_gpu.shape}")
			
			yield Message(
				kind='spatial_feature',
				payload={
					'features': spatial_features_gpu,
					'tier': tier,
					'case_id': msg.payload['case_id'],
					'patch_id': msg.payload['patch_id'],
					'bbox': msg.payload.get('bbox', [])
				}
			)
		
		except Exception as e:
			logger.error(f"空间分支处理失败 tier={msg.payload.get('tier', 0)}: {e}")
