import numpy as np
from scipy import ndimage
from skimage import morphology


class CTPreprocessor:
	"""CT图像预处理类，提供归一化和ROI提取功能"""
	
	def __init__(self, clip_percentiles=(0.5, 99.5), logger=None):
		"""
		初始化预处理器

		参数:
			clip_percentiles: 用于归一化的百分位数裁剪范围
			logger: 日志记录器实例
		"""
		self.clip_percentiles = clip_percentiles
		self.logger = logger
	
	def normalize(self, volume):
		"""
		使用百分位数进行鲁棒归一化

		参数:
			volume: 输入体积数据

		返回:
			归一化后的体积，范围[0,1]
		"""
		# 计算裁剪阈值
		p_low, p_high = np.percentile(volume, self.clip_percentiles)
		
		# 线性变换到[0,1]
		normalized = (volume - p_low) / (p_high - p_low + 1e-8)
		
		# 裁剪到[0,1]范围
		normalized = np.clip(normalized, 0, 1)
		
		if self.logger:
			self.logger.log_info(f"Normalized volume with percentiles: {self.clip_percentiles}")
		
		return normalized
	
	def extract_liver_roi(self, volume, threshold=None, largest_cc=True):
		"""
		提取肝脏ROI

		参数:
			volume: 输入体积数据
			threshold: 阈值，如果为None则使用99.8%百分位数
			largest_cc: 是否只保留最大连通区域

		返回:
			肝脏掩码
		"""
		# 自动计算阈值
		if threshold is None:
			threshold = np.percentile(volume, 99.8)
		
		# 阈值分割
		binary = volume > threshold
		
		# 形态学操作填充空洞
		binary = ndimage.binary_closing(binary, structure=morphology.ball(3))
		
		# 只保留最大连通区域
		if largest_cc:
			labels, num_features = ndimage.label(binary)
			if num_features > 0:
				sizes = np.bincount(labels.flatten())
				sizes[0] = 0  # 忽略背景
				largest_label = np.argmax(sizes)
				binary = labels == largest_label
		
		if self.logger:
			self.logger.log_info(f"Extracted liver ROI with threshold: {threshold}")
			self.logger.log_info(f"ROI volume: {np.sum(binary)} voxels")
		
		return binary
	
	def apply_windowing(self, volume, window_center, window_width):
		"""
		应用窗位窗宽变换（可选）

		参数:
			volume: 输入体积数据
			window_center: 窗位
			window_width: 窗宽

		返回:
			窗位窗宽处理后的体积
		"""
		# 计算窗位窗宽范围
		min_value = window_center - window_width / 2
		max_value = window_center + window_width / 2
		
		# 线性变换到[0,1]
		windowed = (volume - min_value) / (max_value - min_value)
		
		# 裁剪到[0,1]范围
		windowed = np.clip(windowed, 0, 1)
		
		if self.logger:
			self.logger.log_info(f"Applied windowing: center={window_center}, width={window_width}")
		
		return windowed