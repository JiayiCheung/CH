import numpy as np
from scipy import ndimage
from skimage import morphology
import torch

from utils.logger import Logger  # 统一使用项目的Logger


class CTPreprocessor:
	"""CT图像预处理类，提供归一化和ROI提取功能，支持GPU加速"""
	
	def __init__(self, clip_percentiles=(0.5, 99.5), roi_threshold=None, roi_percentile=99.8,
	             use_largest_cc=True, device='cuda', logger=None):
		"""
		初始化预处理器

		参数:
			clip_percentiles: 用于归一化的百分位数裁剪范围
			device: 计算设备，默认使用CUDA加速
			logger: 日志记录器实例，应为utils.Logger的实例
		"""
		self.clip_percentiles = clip_percentiles
		self.clip_percentiles = clip_percentiles
		self.roi_threshold = roi_threshold
		self.roi_percentile = roi_percentile
		self.use_largest_cc = use_largest_cc
		self.device = 'cuda'
		self.logger = logger
		
		# 缓存上次计算的结果，避免重复计算
		self.last_volume_hash = None
		self.last_threshold = None
		self.last_roi = None
	
	def normalize(self, volume):
		"""
		使用百分位数进行鲁棒归一化，支持GPU加速

		参数:
			volume: 输入体积数据

		返回:
			归一化后的体积，范围[0,1]
		"""
		# 尝试使用GPU加速
		use_gpu = False
		if 'cuda' in self.device and volume.size > 1000000:  # 对于大体积使用GPU
			try:
				# 转移数据到GPU
				if not torch.is_tensor(volume):
					volume_tensor = torch.tensor(volume, device=self.device)
				else:
					volume_tensor = volume.to(self.device)
				
				# 使用PyTorch计算百分位数
				k_low = int(volume_tensor.numel() * self.clip_percentiles[0] / 100)
				k_high = int(volume_tensor.numel() * self.clip_percentiles[1] / 100)
				
				# 展平并排序
				flat_volume = volume_tensor.flatten()
				sorted_volume, _ = torch.sort(flat_volume)
				
				# 获取百分位数值
				p_low = sorted_volume[max(0, k_low)].item()
				p_high = sorted_volume[min(sorted_volume.numel() - 1, k_high)].item()
				
				# 在GPU上进行归一化
				normalized = (volume_tensor - p_low) / (p_high - p_low + 1e-8)
				normalized = torch.clamp(normalized, 0, 1)
				
				# 转回CPU并转为NumPy
				result = normalized.cpu().numpy()
				use_gpu = True
				
				# 清理GPU内存
				del volume_tensor, flat_volume, sorted_volume
				torch.cuda.empty_cache()
			except Exception as e:
				if self.logger:
					self.logger.log_warning(f"GPU归一化失败: {e}，回退到CPU")
				use_gpu = False
		
		# 如果GPU加速失败或不适用，使用CPU
		if not use_gpu:
			# 计算裁剪阈值
			p_low, p_high = np.percentile(volume, self.clip_percentiles)
			
			# 线性变换到[0,1]
			normalized = (volume - p_low) / (p_high - p_low + 1e-8)
			
			# 裁剪到[0,1]范围
			result = np.clip(normalized, 0, 1)
		
		if self.logger:
			self.logger.log_info(
				f"Normalized volume with percentiles: {self.clip_percentiles}, using {'GPU' if use_gpu else 'CPU'}")
		
		return result
	
	def extract_liver_roi(self, volume, threshold=None, largest_cc=None):
		"""
		提取肝脏ROI，带缓存以避免重复计算

		参数:
			volume: 输入体积数据
			threshold: 阈值，如果为None则使用实例变量
			largest_cc: 是否只保留最大连通区域，如果为None则使用实例变量

		返回:
			肝脏掩码
		"""
		# 使用实例变量作为默认值
		if threshold is None:
			threshold = self.roi_threshold
		
		if largest_cc is None:
			largest_cc = self.use_largest_cc
		
		# 检查是否可以使用缓存
		volume_hash = hash(volume.tobytes()) if hasattr(volume, 'tobytes') else None
		if volume_hash == self.last_volume_hash and threshold == self.last_threshold and self.last_roi is not None:
			if self.logger:
				self.logger.log_info(f"Using cached liver ROI")
			return self.last_roi
		
		# 自动计算阈值（如果未提供固定阈值）
		if threshold is None:
			threshold = np.percentile(volume, self.roi_percentile)
		
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
		
		# 更新缓存
		self.last_volume_hash = volume_hash
		self.last_threshold = threshold
		self.last_roi = binary
		
		if self.logger:
			self.logger.log_info(f"Extracted liver ROI with threshold: {threshold}")
			self.logger.log_info(f"ROI volume: {np.sum(binary)} voxels")
		
		return binary
	
	
	
	
	
	
	def apply_windowing(self, volume, window_center, window_width):
		"""
		应用窗位窗宽变换（可选），支持GPU加速

		参数:
			volume: 输入体积数据
			window_center: 窗位
			window_width: 窗宽

		返回:
			窗位窗宽处理后的体积
		"""
		# 尝试使用GPU加速
		use_gpu = False
		if 'cuda' in self.device and volume.size > 1000000:  # 对于大体积使用GPU
			try:
				# 转移数据到GPU
				if not torch.is_tensor(volume):
					volume_tensor = torch.tensor(volume, device=self.device)
				else:
					volume_tensor = volume.to(self.device)
				
				# 计算窗位窗宽范围
				min_value = window_center - window_width / 2
				max_value = window_center + window_width / 2
				
				# 在GPU上进行变换
				windowed = (volume_tensor - min_value) / (max_value - min_value)
				windowed = torch.clamp(windowed, 0, 1)
				
				# 转回CPU并转为NumPy
				result = windowed.cpu().numpy()
				use_gpu = True
				
				# 清理GPU内存
				del volume_tensor, windowed
				torch.cuda.empty_cache()
			except Exception as e:
				if self.logger:
					self.logger.log_warning(f"GPU窗位窗宽变换失败: {e}，回退到CPU")
				use_gpu = False
		
		# 如果GPU加速失败或不适用，使用CPU
		if not use_gpu:
			# 计算窗位窗宽范围
			min_value = window_center - window_width / 2
			max_value = window_center + window_width / 2
			
			# 线性变换到[0,1]
			windowed = (volume - min_value) / (max_value - min_value)
			
			# 裁剪到[0,1]范围
			result = np.clip(windowed, 0, 1)
		
		if self.logger:
			self.logger.log_info(
				f"Applied windowing: center={window_center}, width={window_width}, using {'GPU' if use_gpu else 'CPU'}")
		
		return result