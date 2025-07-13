#!/usr/bin/env python3
"""
工具模块 - 重构版
删除SamplingScheduler，添加ComponentFactory
"""

# 核心工具类
from .metrics import SegmentationMetrics
from .logger import Logger
from .component_factory import ComponentFactory

# 条件导入可视化工具（云服务器可能不需要）
try:
	from .visualization import Visualizer
	
	_has_visualization = True
except ImportError:
	_has_visualization = False
	Visualizer = None

# 导出列表 - 删除SamplingScheduler
__all__ = [
	'SegmentationMetrics',
	'Logger',
	'ComponentFactory'
]

# 有条件导出可视化工具
if _has_visualization:
	__all__.append('Visualizer')

# 版本信息
__version__ = "2.0.0"


# 工厂函数
def create_logger(output_dir, level='INFO'):
	"""创建日志记录器的便捷函数"""
	return ComponentFactory.create_logger(output_dir, level)


def create_metrics_calculator():
	"""创建评估指标计算器的便捷函数"""
	return SegmentationMetrics()


# 向后兼容性处理
def __getattr__(name):
	"""处理废弃组件的访问"""
	if name == 'SamplingScheduler':
		import warnings
		warnings.warn(
			"SamplingScheduler has been removed. "
			"Use SamplingManager from data.sampling_manager instead.",
			DeprecationWarning,
			stacklevel=2
		)
		raise AttributeError(f"'{name}' has been removed from utils module")
	
	# 处理可视化工具的访问
	if name == 'Visualizer' and not _has_visualization:
		import warnings
		warnings.warn(
			"Visualizer is not available. This may be due to missing visualization dependencies.",
			RuntimeWarning,
			stacklevel=2
		)
		return None
	
	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")