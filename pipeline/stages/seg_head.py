# pipeline/stages/seg_head.py
"""
分割头Stage (GPU-6) - GPU直传版本
"""

import torch
import torch.nn as nn
from typing import Iterable, Dict, Any
import logging

from .base import StageBase
from ..message import Message

logger = logging.getLogger(__name__)


class SegHead(StageBase):
	"""分割头Stage：GPU直传版本"""
	
	def __init__(self, cfg: Dict[str, Any], device: torch.device):
		super().__init__(cfg, device)
		
		from models.seg_head import SegHead as SegHeadModel
		
		seg_config = cfg.get('seg_head_params', {})
		self.seg_head_model = SegHeadModel(
			in_channels=seg_config.get('in_channels', 32),
			out_channels=seg_config.get('out_channels', 1),
			num_classes=seg_config.get('num_classes', 1),
			intermediate_channels=seg_config.get('intermediate_channels', [16, 8]),
			dropout=seg_config.get('dropout', 0.1),
			use_deep_supervision=seg_config.get('use_deep_supervision', False)
		).to(device)
		
		# 损失计算
		self.compute_loss = seg_config.get('compute_loss', True)
		if self.compute_loss:
			from loss import CombinedLoss
			self.criterion = CombinedLoss().to(device)
		
		self.label_cache = {}
		
		logger.info(f"SegHead初始化: seg_config={seg_config}")
	
	def process(self, msg: Message) -> Iterable[Message]:
		"""处理：GPU直传版本"""
		if msg.kind != 'multi':
			return
		
		try:
			# 已 CUDA
			features = msg.payload['features']
			case_id = msg.payload['case_id']
			
			logger.debug(f"SegHead接收: {features.shape}")
			
			# 确保batch维度
			if features.dim() == 4:  # [C, D, H, W]
				features = features.unsqueeze(0)  # [1, C, D, H, W]
			
			# 分割预测
			context = torch.no_grad() if not self.training else torch.enable_grad()
			with context:
				seg_output = self.seg_head_model(features)
			
			logits = seg_output['logits']
			if logits.dim() == 5:
				logits = logits.squeeze(0)  # 移除batch维度
			
			# 端到端反传；不再发 loss Message
			if self.training and self.compute_loss:
				labels = self._get_labels_for_case(case_id, logits.shape).cuda()
				loss = self.criterion(logits, labels)
				loss.backward()  # DDP 会自动 AllReduce
			
			# 构建输出
			output_payload = {
				'logits': logits.detach(),
				'probabilities': torch.sigmoid(logits).detach(),
				'case_id': case_id,
				'tier_count': msg.payload.get('tier_count', 0),
				'processed_tiers': msg.payload.get('processed_tiers', [])
			}
			
			# 输出logits
			yield Message(
				kind='logit',
				payload=output_payload
			)
		
		except Exception as e:
			logger.error(f"分割头处理失败: {e}")
	
	def _get_labels_for_case(self, case_id: str, logits_shape: torch.Size) -> torch.Tensor:
		"""获取labels（简化版）"""
		if self.training:
			# 生成虚拟labels用于测试
			dummy_labels = torch.randint(0, 2, logits_shape, device=logits_shape.device, dtype=torch.float32)
			logger.debug(f"使用虚拟labels {dummy_labels.shape}")
			return dummy_labels
		
		return None
