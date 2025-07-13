#!/usr/bin/env python3
"""
模型模块 - 重构版
删除废弃的分布式实现，保留核心组件
"""

# 核心模型
from .vessel_segmenter import VesselSegmenter

# 模型组件
from .ch_branch import CHBranch
from .spatial_branch import SpatialBranch

# 融合模块
from .fusion import AttentionFusion, MultiscaleFusion

# 导出列表 - 删除DistributedVesselSegmenter
__all__ = [
	# 核心模型
	'VesselSegmenter',
	
	# 模型组件
	'CHBranch',
	'SpatialBranch',
	
	# 融合模块
	'AttentionFusion',
	'MultiscaleFusion'
]

# 版本信息
__version__ = "2.0.0"
__author__ = "Liver Vessel Segmentation Team"


# 模型工厂函数
def create_vessel_segmenter(config=None, **kwargs):
	"""
    创建血管分割模型的工厂函数

    参数:
        config: 配置字典
        **kwargs: 额外参数

    返回:
        VesselSegmenter实例
    """
	if config is not None:
		model_config = config.get('model', {})
		
		# 从配置中提取参数
		in_channels = model_config.get('in_channels', 1)
		out_channels = model_config.get('out_channels', 1)
		ch_params = model_config.get('ch_params')
		tier_params = model_config.get('tier_params')
		
		# 用kwargs覆盖配置参数
		in_channels = kwargs.get('in_channels', in_channels)
		out_channels = kwargs.get('out_channels', out_channels)
		ch_params = kwargs.get('ch_params', ch_params)
		tier_params = kwargs.get('tier_params', tier_params)
		
		return VesselSegmenter(
			in_channels=in_channels,
			out_channels=out_channels,
			ch_params=ch_params,
			tier_params=tier_params
		)
	else:
		return VesselSegmenter(**kwargs)


# 向后兼容性警告
def __getattr__(name):
	"""处理废弃组件的访问"""
	if name == 'DistributedVesselSegmenter':
		import warnings
		warnings.warn(
			"DistributedVesselSegmenter has been removed. "
			"Use VesselSegmenter with torch.nn.parallel.DistributedDataParallel instead.",
			DeprecationWarning,
			stacklevel=2
		)
		raise AttributeError(f"'{name}' has been removed from models module")
	
	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")