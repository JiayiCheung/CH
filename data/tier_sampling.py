import numpy as np
from scipy import ndimage
from scipy.ndimage import zoom
from skimage import morphology
import random


class TierSampler:
	"""实现三级采样策略，支持智能采样"""
	
	def __init__(self, tier0_size=256, tier1_size=96, tier2_size=64,
	             max_tier1=10, max_tier2=20, logger=None):
		"""
		初始化三级采样器

		参数:
			tier0_size: Tier-0采样块大小
			tier1_size: Tier-1采样块大小
			tier2_size: Tier-2采样块大小
			max_tier1: Tier-1最大采样数量
			max_tier2: Tier-2最大采样数量
			logger: 日志记录器实例
		"""
		self.tier0_size = tier0_size
		self.tier1_size = tier1_size
		self.tier2_size = tier2_size
		self.max_tier1 = max_tier1
		self.max_tier2 = max_tier2
		self.logger = logger
		
		# 初始化智能采样组件 - 延迟导入避免循环引用
		from .importance_sampler import ImportanceSampler
		from .complexity_analyzer import ComplexityAnalyzer
		
		self.importance_sampler = ImportanceSampler(logger=logger)
		self.complexity_analyzer = ComplexityAnalyzer(logger=logger)
		
		# 初始化采样参数
		self.sampling_params = {
			'tier1_samples': max_tier1,
			'tier2_samples': max_tier2,
			'hard_mining_weight': 0.0,
			'importance_weight': 0.0
		}
	
	def set_sampling_params(self, params):
		"""
		设置采样参数

		参数:
			params: 采样参数字典
		"""
		self.sampling_params.update(params)
		if self.logger:
			self.logger.log_info(f"Updated sampling params: {self.sampling_params}")
	
	def sample(self, image_data, label_data, liver_mask=None,
	           difficulty_map=None, case_id=None):
		"""
		执行三级采样

		参数:
			image_data: 输入图像数据
			label_data: 输入标签数据
			liver_mask: 肝脏掩码（可选）
			difficulty_map: 难度图（可选）
			case_id: 案例ID（可选）

		返回:
			采样块列表，每个元素为包含tier、image和label的字典
		"""
		# 计算案例复杂度
		if label_data is not None:
			complexity = self.complexity_analyzer.compute_complexity(label_data)
		else:
			complexity = 1.0
		
		# 应用复杂度调整采样数量
		tier1_samples = self.sampling_params['tier1_samples']
		tier2_samples = self.sampling_params['tier2_samples']
		
		if self.sampling_params.get('enable_adaptive_density', True):
			adjusted = self.complexity_analyzer.adjust_sampling_density(
				complexity, tier1_samples, tier2_samples
			)
			tier1_samples, tier2_samples = adjusted
		
		# 应用采样参数
		max_tier1 = tier1_samples
		max_tier2 = tier2_samples
		
		if self.logger:
			self.logger.log_info(f"Sampling case {case_id} with complexity {complexity:.2f}")
			self.logger.log_info(f"Tier1 samples: {max_tier1}, Tier2 samples: {max_tier2}")
		
		# 调用增强版三级采样函数
		return self.enhanced_three_tier_sampling(
			image_data, label_data, liver_mask,
			self.tier0_size, self.tier1_size, self.tier2_size,
			max_tier1, max_tier2,
			difficulty_map=difficulty_map,
			importance_weight=self.sampling_params.get('importance_weight', 0.0),
			hard_mining_weight=self.sampling_params.get('hard_mining_weight', 0.0)
		)
	
	def enhanced_three_tier_sampling(self, image_data, label_data, liver_mask=None,
	                                 tier0_size=256, tier1_size=96, tier2_size=64,
	                                 max_tier1=10, max_tier2=30,
	                                 difficulty_map=None,
	                                 importance_weight=0.0,
	                                 hard_mining_weight=0.0):
		"""
		增强版三级采样: 支持重要性采样和硬样本挖掘

		返回包含图像和血管/肿瘤掩码的采样块列表
		"""
		patches = []
		
		# ---- TIER-0: 器官级别采样块 (保持不变) ----
		if liver_mask is None:
			# 如果没有提供肝脏掩码，使用所有标签>0的区域
			liver_mask = (label_data > 0).astype(np.uint8)
		# 寻找肝脏边界框
		bbox = np.array(np.where(liver_mask > 0))
		if bbox.shape[1] == 0:
			center = [s // 2 for s in image_data.shape]
		else:
			min_coord = np.min(bbox, axis=1)
			max_coord = np.max(bbox, axis=1)
			center = ((min_coord + max_coord) / 2).astype(int)
		slices = tuple(
			slice(max(0, c - tier0_size // 2), min(image_data.shape[i], c + tier0_size // 2))
			for i, c in enumerate(center)
		)
		img0 = image_data[slices]
		lbl0 = label_data[slices] if label_data is not None else None
		# 强制调整为目标大小
		img0 = zoom(img0, [tier0_size / s for s in img0.shape], order=1)
		if lbl0 is not None:
			lbl0 = zoom(lbl0, [tier0_size / s for s in lbl0.shape], order=0)
		patches.append({'tier': 0, 'image': img0, 'label': lbl0})
		
		if self.logger:
			self.logger.log_info(f"Created Tier-0 patch of size {img0.shape}")
		
		# 如果没有标签数据，只返回Tier-0采样块
		if label_data is None:
			return patches
		
		# ---- TIER-1: 结构级别 (智能采样增强) ----
		foreground_mask = (label_data > 0)
		
		# 处理前景稀疏情况
		if np.sum(foreground_mask) > 0 and np.sum(foreground_mask) < 1000:  # 前景太稀疏
			if self.logger:
				self.logger.log_info(f"前景过于稀疏({np.sum(foreground_mask)}个体素)，进行膨胀操作")
			foreground_mask = morphology.binary_dilation(foreground_mask, morphology.ball(2))
			if self.logger:
				self.logger.log_info(f"膨胀后前景体素数: {np.sum(foreground_mask)}")
		
		if self.logger:
			self.logger.log_info(f"前景掩码体素数: {np.sum(foreground_mask)}")
		
		# 提取骨架或使用前景点
		if np.sum(foreground_mask) < 500:  # 前景太稀疏
			points = np.array(np.where(foreground_mask)).T
			if self.logger:
				self.logger.log_info(f"前景过于稀疏，跳过骨架化，直接使用所有前景点: {len(points)}个")
		else:
			# 尝试骨架化
			skeleton = morphology.skeletonize(foreground_mask)
			points = np.array(np.where(skeleton)).T
			if self.logger:
				self.logger.log_info(f"骨架化后得到点数: {len(points)}")
			if len(points) == 0:
				points = np.array(np.where(foreground_mask)).T
				if self.logger:
					self.logger.log_info(f"骨架为空，使用前景点: {len(points)}个")
		
		if len(points) == 0:
			points = np.array(np.where(foreground_mask)).T
		
		# 应用智能采样 (如果启用)
		if importance_weight > 0 and len(points) > 0:
			# 计算重要性图
			importance_map = self.importance_sampler.compute_importance_map(
				label_data, difficulty_map
			)
			
			# 获取点的重要性值
			importance_values = np.array([
				importance_map[tuple(pt)] for pt in points
			])
			
			# 如果有难度图，结合难度信息
			if difficulty_map is not None and hard_mining_weight > 0:
				difficulty_values = np.array([
					difficulty_map[tuple(pt)] for pt in points
				])
				
				# 结合重要性和难度
				combined_weights = (
						(1 - hard_mining_weight - importance_weight) * 0.5 +  # 均匀采样
						importance_weight * importance_values +  # 重要性
						hard_mining_weight * difficulty_values  # 难度
				)
			else:
				# 只使用重要性
				combined_weights = (
						(1 - importance_weight) * 0.5 +  # 均匀采样
						importance_weight * importance_values  # 重要性
				)
			
			# 智能采样
			if len(points) > max_tier1:
				points = self.importance_sampler.importance_based_sampling(
					points, combined_weights, max_tier1
				)
				if self.logger:
					self.logger.log_info(f"通过智能采样选择了{len(points)}个Tier-1点")
		else:
			# 原始均匀采样
			if len(points) > max_tier1:
				sel_idx = np.linspace(0, len(points) - 1, max_tier1).astype(int)
				points = points[sel_idx]
				if self.logger:
					self.logger.log_info(f"通过均匀采样选择了{len(points)}个Tier-1点")
		
		# 提取Tier-1块
		for pt in points:
			c = pt
			slices = tuple(
				slice(max(0, c[i] - tier1_size // 2), min(image_data.shape[i], c[i] + tier1_size // 2))
				for i in range(3)
			)
			img1 = image_data[slices]
			lbl1 = label_data[slices]
			# 调整大小
			if img1.shape != (tier1_size,) * 3:
				img1 = zoom(img1, [tier1_size / s for s in img1.shape], order=1)
				lbl1 = zoom(lbl1, [tier1_size / s for s in lbl1.shape], order=0)
			patches.append({'tier': 1, 'image': img1, 'label': lbl1})
		
		if self.logger:
			self.logger.log_info(f"Created {len(points)} Tier-1 patches")
		
		# ---- TIER-2: 细节级别 (智能采样增强) ----
		# 提取细微结构
		vessel_mask = (label_data > 0.5) & (label_data < 1.5)
		tumor_mask = (label_data > 1.5) & (label_data < 2.5)
		detail_mask = vessel_mask | tumor_mask
		dist_map = ndimage.distance_transform_edt(detail_mask)
		thin_mask = (dist_map > 0) & (dist_map <= 5)
		skel_detail = morphology.skeletonize(thin_mask)
		detail_points = np.array(np.where(skel_detail)).T
		
		if self.logger:
			self.logger.log_info(f"细节骨架上得到点数: {len(detail_points)}")
		
		# 应用智能采样 (如果启用)
		if importance_weight > 0 and len(detail_points) > 0:
			# 计算重要性图 (如果Tier-1未计算)
			if 'importance_map' not in locals():
				importance_map = self.importance_sampler.compute_importance_map(
					label_data, difficulty_map
				)
			
			# 获取点的重要性值
			importance_values = np.array([
				importance_map[tuple(pt)] for pt in detail_points
			])
			
			# 如果有难度图，结合难度信息
			if difficulty_map is not None and hard_mining_weight > 0:
				difficulty_values = np.array([
					difficulty_map[tuple(pt)] for pt in detail_points
				])
				
				# 结合重要性和难度
				combined_weights = (
						(1 - hard_mining_weight - importance_weight) * 0.5 +  # 均匀采样
						importance_weight * importance_values +  # 重要性
						hard_mining_weight * difficulty_values  # 难度
				)
			else:
				# 只使用重要性
				combined_weights = (
						(1 - importance_weight) * 0.5 +  # 均匀采样
						importance_weight * importance_values  # 重要性
				)
			
			# 智能采样
			if len(detail_points) > max_tier2:
				detail_points = self.importance_sampler.importance_based_sampling(
					detail_points, combined_weights, max_tier2
				)
				if self.logger:
					self.logger.log_info(f"通过智能采样选择了{len(detail_points)}个Tier-2点")
		else:
			# 原始随机采样
			if len(detail_points) > max_tier2:
				idx = np.random.choice(len(detail_points), max_tier2, replace=False)
				detail_points = detail_points[idx]
				if self.logger:
					self.logger.log_info(f"通过随机采样选择了{len(detail_points)}个Tier-2点")
		
		# 提取Tier-2块
		for pt in detail_points:
			c = pt
			slices = tuple(
				slice(max(0, c[i] - tier2_size // 2), min(image_data.shape[i], c[i] + tier2_size // 2))
				for i in range(3)
			)
			img2 = image_data[slices]
			lbl2 = label_data[slices]
			# 调整大小
			if img2.shape != (tier2_size,) * 3:
				img2 = zoom(img2, [tier2_size / s for s in img2.shape], order=1)
				lbl2 = zoom(lbl2, [tier2_size / s for s in lbl2.shape], order=0)
			patches.append({'tier': 2, 'image': img2, 'label': lbl2})
		
		if self.logger:
			self.logger.log_info(f"Created {len(detail_points)} Tier-2 patches")
			self.logger.log_info(f"Total patches: {len(patches)}")
		
		return patches