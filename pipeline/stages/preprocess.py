# pipeline/stages/patch_dispatch.py
"""
补丁采样和分发Stage (GPU-1) - GPU直传版本
"""

import torch
import numpy as np
from typing import Iterable, Dict, Any
import logging

from .base import StageBase
from ..message import Message

logger = logging.getLogger(__name__)


class PatchDispatch(StageBase):
	"""补丁采样和分发Stage：GPU直传版本"""
	
	def __init__(self, cfg: Dict[str, Any], device: torch.device):
		super().__init__(cfg, device)
		
		from data.tier_sampling import TierSampler
		
		sampling_config = cfg.get('smart_sampling', {})
		self.tier_sampler = TierSampler(
			tier0_size=sampling_config.get('tier0_size', 256),
			tier1_size=sampling_config.get('tier1_size', 96),
			tier2_size=sampling_config.get('tier2_size', 64),
			max_tier1=sampling_config.get('max_tier1', 10),
			max_tier2=sampling_config.get('max_tier2', 20),
			random_state=sampling_config.get('random_state', 42),
			seed=sampling_config.get('seed', None),
			mode=sampling_config.get('mode', 'adaptive')
		)
		
		logger.info(f"PatchDispatch初始化: {sampling_config}")
	
	def process(self, msg: Message) -> Iterable[Message]:
		"""处理：GPU直传，不经过CPU"""
		if msg.kind != 'preprocessed':
			return
		
		try:
			# 一次性搬到 GPU-1
			image_gpu = msg.payload['image'].cuda(non_blocking=True)
			roi_mask_gpu = msg.payload['roi_mask'].cuda(non_blocking=True)
			label_tensor = msg.payload.get('label')
			if label_tensor is not None:
				label_tensor = label_tensor.cuda(non_blocking=True)
			case_id = msg.payload['case_id']
			
			logger.debug(f"PatchDispatch接收: {image_gpu.shape}, device={image_gpu.device}")
			
			# 直接喂 CUDA 给 TierSampler
			patches = self.tier_sampler.sample(
				image=image_gpu.squeeze(0),  # [D,H,W] CUDA
				label=label_tensor.squeeze(0) if label_tensor is not None else None,
				liver_mask=roi_mask_gpu,  # CUDA
				case_id=case_id)
			
			# 输出CUDA tensor
			for patch in patches:
				patch_tensor = patch['image']  # 已是 CUDA
				patch_label = patch.get('label')
				
				logger.debug(f"生成patch tier={patch['tier']}, shape={patch_tensor.shape}")
				
				yield Message(
					kind='patch',
					payload={
						'image': patch_tensor,  # CUDA tensor
						'label': patch_label,  # CUDA tensor
						'tier': patch['tier'],
						'patch_id': patch.get('patch_id', 0),
						'case_id': case_id,
						'bbox': patch.get('bbox', []),
						'center': patch.get('center', [])
					}
				)
		
		except Exception as e:
			logger.error(f"补丁采样失败: {e}")
