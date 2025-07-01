import torch
import numpy as np
import scipy.ndimage as ndimage
from skimage import measure
from sklearn.metrics import confusion_matrix


class SegmentationMetrics:
	"""分割评估指标计算类"""
	
	@staticmethod
	def dice_coefficient(pred, target, smooth=1e-6):
		"""
		计算Dice系数

		参数:
			pred: 预测掩码 (二值化后)
			target: 目标掩码
			smooth: 平滑项，防止分母为0

		返回:
			Dice系数值 [0,1]
		"""
		# 将输入转换为numpy数组
		if torch.is_tensor(pred):
			pred = pred.cpu().numpy()
		if torch.is_tensor(target):
			target = target.cpu().numpy()
		
		# 计算Dice系数
		intersection = np.sum(pred * target)
		union = np.sum(pred) + np.sum(target)
		
		dice = (2. * intersection + smooth) / (union + smooth)
		
		return dice
	
	@staticmethod
	def iou(pred, target, smooth=1e-6):
		"""
		计算IoU (Intersection over Union)

		参数:
			pred: 预测掩码 (二值化后)
			target: 目标掩码
			smooth: 平滑项，防止分母为0

		返回:
			IoU值 [0,1]
		"""
		# 将输入转换为numpy数组
		if torch.is_tensor(pred):
			pred = pred.cpu().numpy()
		if torch.is_tensor(target):
			target = target.cpu().numpy()
		
		# 计算IoU
		intersection = np.sum(pred * target)
		union = np.sum(pred) + np.sum(target) - intersection
		
		iou = (intersection + smooth) / (union + smooth)
		
		return iou
	
	@staticmethod
	def precision_recall_f1(pred, target):
		"""
		计算精确率、召回率和F1分数

		参数:
			pred: 预测掩码 (二值化后)
			target: 目标掩码

		返回:
			精确率、召回率和F1分数
		"""
		# 将输入转换为numpy数组
		if torch.is_tensor(pred):
			pred = pred.cpu().numpy()
		if torch.is_tensor(target):
			target = target.cpu().numpy()
		
		# 展平数组
		pred_flat = pred.flatten()
		target_flat = target.flatten()
		
		# 计算混淆矩阵
		tn, fp, fn, tp = confusion_matrix(target_flat, pred_flat, labels=[0, 1]).ravel()
		
		# 计算指标
		precision = tp / (tp + fp) if (tp + fp) > 0 else 0
		recall = tp / (tp + fn) if (tp + fn) > 0 else 0
		f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
		
		return precision, recall, f1
	
	@staticmethod
	def hausdorff_distance(pred, target, percentile=95):
		"""
		计算Hausdorff距离 (使用percentile版本，更鲁棒)

		参数:
			pred: 预测掩码 (二值化后)
			target: 目标掩码
			percentile: 百分位数，用于计算部分Hausdorff距离

		返回:
			Hausdorff距离值
		"""
		# 将输入转换为numpy数组
		if torch.is_tensor(pred):
			pred = pred.cpu().numpy()
		if torch.is_tensor(target):
			target = target.cpu().numpy()
		
		# 提取边界
		pred_boundary = measure.find_contours(pred.squeeze())
		target_boundary = measure.find_contours(target.squeeze())
		
		# 如果任一掩码为空，返回最大距离
		if not pred_boundary or not target_boundary:
			return np.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2)
		
		# 将边界转换为点集
		pred_points = np.vstack(pred_boundary)
		target_points = np.vstack(target_boundary)
		
		# 计算预测到目标的距离
		d1 = []
		for p in pred_points:
			d1.append(np.min(np.sum((target_points - p) ** 2, axis=1)))
		
		# 计算目标到预测的距离
		d2 = []
		for p in target_points:
			d2.append(np.min(np.sum((pred_points - p) ** 2, axis=1)))
		
		# 计算percentile Hausdorff距离
		h1 = np.percentile(np.sqrt(d1), percentile)
		h2 = np.percentile(np.sqrt(d2), percentile)
		
		return max(h1, h2)
	
	@staticmethod
	def vessel_connectivity_score(pred, target):
		"""
		计算血管连通性分数 (基于拓扑结构)

		参数:
			pred: 预测掩码 (二值化后)
			target: 目标掩码

		返回:
			连通性分数 [0,1]
		"""
		# 将输入转换为numpy数组
		if torch.is_tensor(pred):
			pred = pred.cpu().numpy()
		if torch.is_tensor(target):
			target = target.cpu().numpy()
		
		# 获取连通分量
		pred_labeled, pred_num = ndimage.label(pred)
		target_labeled, target_num = ndimage.label(target)
		
		# 计算连通分量数量比率
		if target_num == 0:
			return 1.0 if pred_num == 0 else 0.0
		
		# 理想情况下，预测和目标的连通分量数量相同
		connectivity_ratio = min(pred_num, target_num) / max(pred_num, target_num)
		
		# 计算每个连通分量的重叠度
		component_overlap = 0
		for i in range(1, target_num + 1):
			target_comp = (target_labeled == i)
			best_overlap = 0
			
			for j in range(1, pred_num + 1):
				pred_comp = (pred_labeled == j)
				overlap = np.sum(target_comp & pred_comp) / np.sum(target_comp | pred_comp)
				best_overlap = max(best_overlap, overlap)
			
			component_overlap += best_overlap
		
		# 平均每个目标连通分量的最佳重叠度
		avg_overlap = component_overlap / target_num if target_num > 0 else 0
		
		# 结合连通性比率和重叠度
		connectivity_score = 0.5 * connectivity_ratio + 0.5 * avg_overlap
		
		return connectivity_score
	
	@staticmethod
	def vessel_branch_accuracy(pred, target):
		"""
		计算血管分支准确率

		参数:
			pred: 预测掩码 (二值化后)
			target: 目标掩码

		返回:
			分支准确率 [0,1]
		"""
		# 将输入转换为numpy数组
		if torch.is_tensor(pred):
			pred = pred.cpu().numpy()
		if torch.is_tensor(target):
			target = target.cpu().numpy()
		
		# 细化/骨架化血管
		from skimage.morphology import skeletonize
		pred_skel = skeletonize(pred)
		target_skel = skeletonize(target)
		
		# 检测分叉点 (有3个或更多邻居的点)
		def detect_branch_points(skel):
			branch_points = np.zeros_like(skel)
			
			# 对3D骨架
			if skel.ndim == 3:
				for z in range(1, skel.shape[0] - 1):
					for y in range(1, skel.shape[1] - 1):
						for x in range(1, skel.shape[2] - 1):
							if skel[z, y, x]:
								# 检查26邻域
								neighbors = skel[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2]
								if np.sum(neighbors) >= 4:  # 中心点 + 至少3个邻居
									branch_points[z, y, x] = 1
			# 对2D骨架
			else:
				for y in range(1, skel.shape[0] - 1):
					for x in range(1, skel.shape[1] - 1):
						if skel[y, x]:
							# 检查8邻域
							neighbors = skel[y - 1:y + 2, x - 1:x + 2]
							if np.sum(neighbors) >= 4:  # 中心点 + 至少3个邻居
								branch_points[y, x] = 1
			
			return branch_points
		
		# 检测分叉点
		pred_branches = detect_branch_points(pred_skel)
		target_branches = detect_branch_points(target_skel)
		
		# 计算分叉点的精确率和召回率
		# 考虑到位置可能有轻微偏移，使用扩张操作
		dilated_pred_branches = ndimage.binary_dilation(pred_branches, iterations=2)
		dilated_target_branches = ndimage.binary_dilation(target_branches, iterations=2)
		
		# 真阳性: 预测的分叉点在扩张的目标分叉点内
		tp = np.sum(pred_branches & dilated_target_branches)
		
		# 假阳性: 预测的分叉点不在扩张的目标分叉点内
		fp = np.sum(pred_branches) - tp
		
		# 假阴性: 目标分叉点不在扩张的预测分叉点内
		fn = np.sum(target_branches) - np.sum(target_branches & dilated_pred_branches)
		
		# 计算F1分数
		precision = tp / (tp + fp) if (tp + fp) > 0 else 0
		recall = tp / (tp + fn) if (tp + fn) > 0 else 0
		f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
		
		return f1
	
	@staticmethod
	def evaluate_all(pred, target, include_advanced=True):
		"""
		计算所有评估指标

		参数:
			pred: 预测掩码 (二值化后)
			target: 目标掩码
			include_advanced: 是否包括高级指标 (计算较慢)

		返回:
			包含所有指标的字典
		"""
		# 基本指标
		dice = SegmentationMetrics.dice_coefficient(pred, target)
		iou_score = SegmentationMetrics.iou(pred, target)
		precision, recall, f1 = SegmentationMetrics.precision_recall_f1(pred, target)
		
		# 创建结果字典
		metrics = {
			'dice': dice,
			'iou': iou_score,
			'precision': precision,
			'recall': recall,
			'f1': f1
		}
		
		# 高级指标 (计算较慢)
		if include_advanced:
			hausdorff = SegmentationMetrics.hausdorff_distance(pred, target)
			connectivity = SegmentationMetrics.vessel_connectivity_score(pred, target)
			branch_accuracy = SegmentationMetrics.vessel_branch_accuracy(pred, target)
			
			metrics.update({
				'hausdorff': hausdorff,
				'connectivity': connectivity,
				'branch_accuracy': branch_accuracy
			})
		
		return metrics