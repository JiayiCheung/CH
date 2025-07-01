import numpy as np
from scipy import ndimage
from scipy.ndimage import zoom
from skimage import morphology
import random


class TierSampler:
	"""实现三级采样策略"""
	
	def __init__(self, tier0_size=256, tier1_size=96, tier2_size=64,
	             max_tier1=10, max_tier2=30):
		"""
		初始化三级采样器

		参数:
			tier0_size: Tier-0采样块大小
			tier1_size: Tier-1采样块大小
			tier2_size: Tier-2采样块大小
			max_tier1: Tier-1最大采样数量
			max_tier2: Tier-2最大采样数量
		"""
		self.tier0_size = tier0_size
		self.tier1_size = tier1_size
		self.tier2_size = tier2_size
		self.max_tier1 = max_tier1
		self.max_tier2 = max_tier2
	
	def sample(self, image_data, label_data, liver_mask=None):
		"""
		执行三级采样

		参数:
			image_data: 输入图像数据
			label_data: 输入标签数据
			liver_mask: 肝脏掩码（可选）

		返回:
			采样块列表，每个元素为包含tier、image和label的字典
		"""
		# 调用三级采样函数
		return self.three_tier_patch_sampling(
			image_data, label_data, liver_mask,
			self.tier0_size, self.tier1_size, self.tier2_size,
			self.max_tier1, self.max_tier2
		)
	
	def three_tier_patch_sampling(self, image_data, label_data, liver_mask=None,
	                              tier0_size=256, tier1_size=96, tier2_size=64,
	                              max_tier1=10, max_tier2=30):
		"""
		三级采样: Tier-0 (整个肝脏), Tier-1 (大型结构), Tier-2 (细微血管/肿瘤)

		返回包含图像和血管/肿瘤掩码的采样块列表
		"""
		patches = []
		
		# ---- TIER-0: 器官级别采样块 (仅1个，覆盖整个肝脏) ----
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
		lbl0 = label_data[slices]
		# 强制调整为目标大小
		img0 = zoom(img0, [tier0_size / s for s in img0.shape], order=1)
		lbl0 = zoom(lbl0, [tier0_size / s for s in lbl0.shape], order=0)
		patches.append({'tier': 0, 'image': img0, 'label': lbl0})
		
		# ---- TIER-1: 结构级别 (粗结构，大血管，小肿瘤) ----
		# 在骨架或大血管处设置点，均匀采样max_tier1个
		foreground_mask = (label_data > 0)
		if np.sum(foreground_mask) > 0 and np.sum(foreground_mask) < 1000:  # 前景太稀疏
			print(f"前景过于稀疏({np.sum(foreground_mask)}个体素)，进行膨胀操作")
			foreground_mask = morphology.binary_dilation(foreground_mask, morphology.ball(2))
			print(f"膨胀后前景体素数: {np.sum(foreground_mask)}")
		
		print(f"前景掩码体素数: {np.sum(foreground_mask)}")
		if np.sum(foreground_mask) < 500:  # 前景太稀疏
			# 直接使用前景点而不是骨架
			points = np.array(np.where(foreground_mask)).T
			print(f"前景过于稀疏，跳过骨架化，直接使用所有前景点: {len(points)}个")
		else:
			# 尝试骨架化
			skeleton = morphology.skeletonize(foreground_mask)
			points = np.array(np.where(skeleton)).T
			print(f"骨架化后得到点数: {len(points)}")
			if len(points) == 0:
				points = np.array(np.where(foreground_mask)).T
				print(f"骨架为空，使用前景点: {len(points)}个")
		if len(points) == 0:
			points = np.array(np.where(foreground_mask)).T
		if len(points) > max_tier1:
			sel_idx = np.linspace(0, len(points) - 1, max_tier1).astype(int)
			points = points[sel_idx]
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
		
		# ---- TIER-2: 细节级别 (最细微血管/肿瘤边界) ----
		# 腐蚀血管掩码提取直径2-4的细微血管
		vessel_mask = (label_data > 0.5) & (label_data < 1.5)
		tumor_mask = (label_data > 1.5) & (label_data < 2.5)
		detail_mask = vessel_mask | tumor_mask
		dist_map = ndimage.distance_transform_edt(detail_mask)
		thin_mask = (dist_map > 0) & (dist_map <= 5)
		skel_detail = morphology.skeletonize(thin_mask)
		detail_points = np.array(np.where(skel_detail)).T
		# 随机选择max_tier2个点
		if len(detail_points) > max_tier2:
			idx = np.random.choice(len(detail_points), max_tier2, replace=False)
			detail_points = detail_points[idx]
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
		
		return patches