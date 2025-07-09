import torch
import numpy as np
import scipy.ndimage as ndimage
from skimage import morphology
from sklearn.metrics import confusion_matrix


class SegmentationMetrics:
	"""分割评估指标计算类，专注于3D数据并使用PyTorch加速"""
	
	@staticmethod
	def dice_coefficient(pred, target, smooth=1e-6):
		"""
		计算Dice系数，使用PyTorch加速

		参数:
			pred: 预测掩码 (二值化后)
			target: 目标掩码
			smooth: 平滑项，防止分母为0

		返回:
			Dice系数值 [0,1]
		"""
		# 确保输入是PyTorch张量，并且在相同设备上
		if not torch.is_tensor(pred):
			pred = torch.as_tensor(pred, dtype=torch.float32)
		if not torch.is_tensor(target):
			target = torch.as_tensor(target, dtype=torch.float32)
		
		# 确保设备一致
		if pred.device != target.device:
			target = target.to(pred.device)
		
		# 计算Dice系数
		intersection = torch.sum(pred * target)
		union = torch.sum(pred) + torch.sum(target)
		
		dice = (2. * intersection + smooth) / (union + smooth)
		
		return dice.item()
	
	@staticmethod
	def iou(pred, target, smooth=1e-6):
		"""
		计算IoU (Intersection over Union)，使用PyTorch加速

		参数:
			pred: 预测掩码 (二值化后)
			target: 目标掩码
			smooth: 平滑项，防止分母为0

		返回:
			IoU值 [0,1]
		"""
		# 确保输入是PyTorch张量，并且在相同设备上
		if not torch.is_tensor(pred):
			pred = torch.as_tensor(pred, dtype=torch.float32)
		if not torch.is_tensor(target):
			target = torch.as_tensor(target, dtype=torch.float32)
		
		# 确保设备一致
		if pred.device != target.device:
			target = target.to(pred.device)
		
		# 计算IoU
		intersection = torch.sum(pred * target)
		union = torch.sum(pred) + torch.sum(target) - intersection
		
		iou = (intersection + smooth) / (union + smooth)
		
		return iou.item()
	
	@staticmethod
	def precision_recall_f1(pred, target):
		"""
		计算精确率、召回率和F1分数，使用PyTorch加速

		参数:
			pred: 预测掩码 (二值化后)
			target: 目标掩码

		返回:
			精确率、召回率和F1分数
		"""
		# 确保输入是PyTorch张量，并且在相同设备上
		if not torch.is_tensor(pred):
			pred = torch.as_tensor(pred, dtype=torch.float32)
		if not torch.is_tensor(target):
			target = torch.as_tensor(target, dtype=torch.float32)
		
		# 确保设备一致
		if pred.device != target.device:
			target = target.to(pred.device)
		
		# 计算混淆矩阵元素
		tp = torch.sum(pred * target)
		fp = torch.sum(pred * (1 - target))
		fn = torch.sum((1 - pred) * target)
		
		# 计算指标
		precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0, device=pred.device)
		recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0, device=pred.device)
		f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0,
		                                                                                                 device=pred.device)
		
		return precision.item(), recall.item(), f1.item()
	
	@staticmethod
	def hausdorff_distance(pred, target, percentile=95):
		"""
		计算3D Hausdorff距离 (使用percentile版本，更鲁棒)
		使用PyTorch加速计算距离变换

		参数:
			pred: 预测掩码 (二值化后)
			target: 目标掩码
			percentile: 百分位数，用于计算部分Hausdorff距离

		返回:
			Hausdorff距离值
		"""
		# 尝试使用PyTorch加速
		try:
			device = 'cuda' if torch.cuda.is_available() else 'cpu'
			
			# 确保输入是PyTorch张量
			if not torch.is_tensor(pred):
				pred = torch.as_tensor(pred, dtype=torch.float32, device=device)
			if not torch.is_tensor(target):
				target = torch.as_tensor(target, dtype=torch.float32, device=device)
			
			# 确保设备一致
			pred = pred.to(device)
			target = target.to(device)
			
			# 移除单维度
			pred = pred.squeeze()
			target = target.squeeze()
			
			# 计算binary mask的边界 (使用PyTorch的卷积实现膨胀和腐蚀)
			# 创建3x3x3核
			kernel_size = 3
			kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=device)
			
			# 腐蚀操作
			pred_eroded = (torch.nn.functional.conv3d(
				pred.unsqueeze(0).unsqueeze(0),
				kernel,
				padding=kernel_size // 2
			) >= kernel.sum()).float().squeeze()
			
			target_eroded = (torch.nn.functional.conv3d(
				target.unsqueeze(0).unsqueeze(0),
				kernel,
				padding=kernel_size // 2
			) >= kernel.sum()).float().squeeze()
			
			# 边界 = 原始 - 腐蚀
			pred_border = pred - pred_eroded
			target_border = target - target_eroded
			
			# 如果边界为空，返回最大距离
			if not torch.any(pred_border) or not torch.any(target_border):
				max_dist = torch.sqrt(torch.tensor(sum(d * d for d in pred.shape), dtype=torch.float32, device=device))
				return max_dist.item()
			
			# 将PyTorch张量转为NumPy，因为PyTorch中没有直接的距离变换函数
			pred_border_np = pred_border.cpu().numpy()
			target_border_np = target_border.cpu().numpy()
			
			# 计算距离变换
			dt_pred = ndimage.distance_transform_edt(~target_border_np)
			dt_target = ndimage.distance_transform_edt(~pred_border_np)
			
			# 获取边界点上的距离
			pred_distances = dt_pred[pred_border_np.astype(bool)]
			target_distances = dt_target[target_border_np.astype(bool)]
			
			# 计算percentile Hausdorff距离
			h1 = np.percentile(pred_distances, percentile)
			h2 = np.percentile(target_distances, percentile)
			
			return max(h1, h2)
		
		except Exception as e:
			print(f"PyTorch加速Hausdorff计算失败: {e}，回退到CPU版本")
			
			# 回退到CPU实现
			if torch.is_tensor(pred):
				pred = pred.cpu().numpy()
			if torch.is_tensor(target):
				target = target.cpu().numpy()
			
			# 确保输入是numpy数组
			pred = np.asarray(pred)
			target = np.asarray(target)
			
			# 移除单维度
			pred = np.squeeze(pred)
			target = np.squeeze(target)
			
			# 计算binary mask的边界
			pred_border = pred ^ ndimage.binary_erosion(pred)
			target_border = target ^ ndimage.binary_erosion(target)
			
			# 如果边界为空，返回最大距离
			if not np.any(pred_border) or not np.any(target_border):
				return np.sqrt(np.sum(np.array(pred.shape) ** 2))
			
			# 计算距离变换
			dt_pred = ndimage.distance_transform_edt(~target_border)
			dt_target = ndimage.distance_transform_edt(~pred_border)
			
			# 获取边界点上的距离
			pred_distances = dt_pred[pred_border]
			target_distances = dt_target[target_border]
			
			# 计算percentile Hausdorff距离
			h1 = np.percentile(pred_distances, percentile)
			h2 = np.percentile(target_distances, percentile)
			
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
		# 将输入转换为numpy数组 (此操作难以在PyTorch中实现)
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
		计算血管分支准确率，仅支持3D数据，使用PyTorch加速

		参数:
			pred: 预测掩码 (二值化后)
			target: 目标掩码

		返回:
			分支准确率 [0,1]
		"""
		try:
			device = 'cuda' if torch.cuda.is_available() else 'cpu'
			
			# 首先在CPU上骨架化 (PyTorch没有直接的骨架化操作)
			# 转换到NumPy数组用于骨架化
			if torch.is_tensor(pred):
				pred_np = pred.cpu().numpy()
			else:
				pred_np = pred
			
			if torch.is_tensor(target):
				target_np = target.cpu().numpy()
			else:
				target_np = target
			
			# 执行骨架化
			from skimage.morphology import skeletonize
			pred_skel = skeletonize(pred_np)
			target_skel = skeletonize(target_np)
			
			# 转回PyTorch张量
			pred_skel = torch.as_tensor(pred_skel, dtype=torch.float32, device=device)
			target_skel = torch.as_tensor(target_skel, dtype=torch.float32, device=device)
			
			# 使用PyTorch的卷积操作检测分支点
			kernel = torch.ones((1, 1, 3, 3, 3), device=device)
			kernel[0, 0, 1, 1, 1] = 0  # 中心点不算邻居
			
			# 计算邻居
			pred_neighbors = torch.nn.functional.conv3d(
				pred_skel.unsqueeze(0).unsqueeze(0),
				kernel,
				padding=1
			).squeeze()
			
			target_neighbors = torch.nn.functional.conv3d(
				target_skel.unsqueeze(0).unsqueeze(0),
				kernel,
				padding=1
			).squeeze()
			
			# 分支点 = 有3个或更多邻居的点
			pred_branches = (pred_neighbors >= 3) & pred_skel.bool()
			target_branches = (target_neighbors >= 3) & target_skel.bool()
			
			# 使用PyTorch的膨胀操作
			dilate_kernel = torch.ones((1, 1, 5, 5, 5), device=device)
			
			dilated_pred_branches = (torch.nn.functional.conv3d(
				pred_branches.float().unsqueeze(0).unsqueeze(0),
				dilate_kernel,
				padding=2
			) > 0).squeeze()
			
			dilated_target_branches = (torch.nn.functional.conv3d(
				target_branches.float().unsqueeze(0).unsqueeze(0),
				dilate_kernel,
				padding=2
			) > 0).squeeze()
			
			# 计算TP, FP, FN
			tp = torch.sum(pred_branches & dilated_target_branches).float()
			fp = torch.sum(pred_branches).float() - tp
			fn = torch.sum(target_branches).float() - torch.sum(target_branches & dilated_pred_branches).float()
			
			# 计算F1
			precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0, device=device)
			recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0, device=device)
			f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0,
			                                                                                                 device=device)
			
			return f1.item()
		
		except Exception as e:
			print(f"PyTorch加速分支准确率计算失败: {e}，回退到CPU版本")
			
			# 回退到CPU版本
			if torch.is_tensor(pred):
				pred = pred.cpu().numpy()
			if torch.is_tensor(target):
				target = target.cpu().numpy()
			
			# 细化/骨架化血管
			from skimage.morphology import skeletonize
			pred_skel = skeletonize(pred)
			target_skel = skeletonize(target)
			
			# 使用卷积核检测分叉点
			kernel = np.ones((3, 3, 3), dtype=np.uint8)
			kernel[1, 1, 1] = 0  # 中心点不算邻居
			
			# 使用卷积计算每个点的邻居数
			pred_neighbors = ndimage.convolve(
				pred_skel.astype(np.uint8),
				kernel,
				mode='constant',
				cval=0
			)
			
			target_neighbors = ndimage.convolve(
				target_skel.astype(np.uint8),
				kernel,
				mode='constant',
				cval=0
			)
			
			# 选择有3个或更多邻居的点作为分支点
			pred_branches = (pred_neighbors >= 3) & pred_skel
			target_branches = (target_neighbors >= 3) & target_skel
			
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
		计算所有评估指标，容错处理高级指标计算

		参数:
			pred: 预测掩码 (二值化后)
			target: 目标掩码
			include_advanced: 是否包括高级指标 (计算较慢)

		返回:
			包含所有指标的字典
		"""
		# 检查设备 - 如果是CUDA张量，使用GPU，否则检查CUDA可用性
		device = None
		if torch.is_tensor(pred) and pred.device.type == 'cuda':
			device = pred.device
		elif torch.is_tensor(target) and target.device.type == 'cuda':
			device = target.device
		elif torch.cuda.is_available():
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')
		
		# 转换为PyTorch张量并移动到正确设备
		if not torch.is_tensor(pred):
			pred = torch.as_tensor(pred, dtype=torch.float32, device=device)
		else:
			pred = pred.to(device)
		
		if not torch.is_tensor(target):
			target = torch.as_tensor(target, dtype=torch.float32, device=device)
		else:
			target = target.to(device)
		
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
			try:
				hausdorff = SegmentationMetrics.hausdorff_distance(pred, target)
				metrics['hausdorff'] = hausdorff
			except Exception as e:
				print(f"Error calculating Hausdorff distance: {e}")
				metrics['hausdorff'] = float('nan')
			
			try:
				connectivity = SegmentationMetrics.vessel_connectivity_score(pred, target)
				metrics['connectivity'] = connectivity
			except Exception as e:
				print(f"Error calculating connectivity score: {e}")
				metrics['connectivity'] = float('nan')
			
			try:
				branch_accuracy = SegmentationMetrics.vessel_branch_accuracy(pred, target)
				metrics['branch_accuracy'] = branch_accuracy
			except Exception as e:
				print(f"Error calculating branch accuracy: {e}")
				metrics['branch_accuracy'] = float('nan')
		
		return metrics