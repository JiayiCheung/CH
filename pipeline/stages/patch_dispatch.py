# pipeline/stages/patch_dispatch.py
"""
补丁分发Stage (GPU-1)
输入: Message(kind='patch')
输出: Message(kind='patch') (路由到不同GPU)
"""

from typing import Iterable
from . import StageBase
from ..message import Message


class PatchDispatch(StageBase):
	"""补丁分发Stage：根据tier路由到不同分支"""
	
	def process(self, msg: Message) -> Iterable[Message]:
		"""
		根据patch的tier进行路由
		tier 0,1 -> CH分支(GPU-2) 和 Spatial分支(GPU-3)
		tier 2 -> 只到Spatial分支(GPU-3)
		"""
		if msg.kind != 'patch':
			return
		
		tier = msg.payload.get('tier', 0)
		
		if tier in [0, 1]:
			# 发送到CH分支和Spatial分支
			yield msg  # 会被dispatcher路由到两个输出通道
		elif tier == 2:
			# 只发送到Spatial分支
			yield msg