#validation_engine.py

"""
验证逻辑完整实现
复用训练流水线进行验证 + 医学指标评估
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import cv2


class MedicalMetrics:
	"""医学分割指标计算"""
	
	@staticmethod
	def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
		"""计算Dice系数"""
		pred = (pred > 0.5).float()
		target = (target > 0.5).float()
		
		intersection = (pred * target).sum()
		union = pred.sum() + target.sum()
		
		dice = (2. * intersection + smooth) / (union + smooth)
		return dice.item()
	
	@staticmethod
	def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
		"""计算IoU分数"""
		pred = (pred > 0.5).float()
		target = (target > 0.5).float()
		
		intersection = (pred * target).sum()
		union = pred.sum() + target.sum() - intersection
		
		iou = (intersection + smooth) / (union + smooth)
		return iou.item()
	
	@staticmethod
	def hausdorff_distance(pred: torch.Tensor, target: torch.Tensor) -> float:
		"""计算Hausdorff距离（简化2D版本）"""
		try:
			pred_np = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
			target_np = (target.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
			
			# 找到边界点
			pred_contours, _ = cv2.findContours(pred_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			target_contours, _ = cv2.findContours(target_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			
			if not pred_contours or not target_contours:
				return float('inf')
			
			# 简化版Hausdorff距离计算
			pred_points = np.vstack(pred_contours[0])[:, 0, :]
			target_points = np.vstack(target_contours[0])[:, 0, :]
			
			# 计算双向最大最小距离
			d1 = max(min(np.linalg.norm(p - target_points, axis=1)) for p in pred_points)
			d2 = max(min(np.linalg.norm(t - pred_points, axis=1)) for t in target_points)
			
			return max(d1, d2)
		
		except Exception:
			return float('inf')
	
	@staticmethod
	def vessel_connectivity_score(pred: torch.Tensor, target: torch.Tensor) -> float:
		"""血管连通性评分"""
		try:
			pred_np = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
			target_np = (target.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
			
			# 计算连通组件
			pred_components = cv2.connectedComponents(pred_np)[0]
			target_components = cv2.connectedComponents(target_np)[0]
			
			# 连通性评分：组件数量相似度
			if target_components == 0:
				return 1.0 if pred_components == 0 else 0.0
			
			connectivity_score = 1.0 - abs(pred_components - target_components) / max(pred_components,
			                                                                          target_components)
			return max(0.0, connectivity_score)
		
		except Exception:
			return 0.0


class DistributedValidator:
	"""分布式验证器 - 复用训练流水线"""
	
	def __init__(self, model, val_loader, device, rank, world_size, output_dir: str = "./validation_results"):
		self.model = model
		self.val_loader = val_loader
		self.device = device
		self.rank = rank
		self.world_size = world_size
		self.output_dir = Path(output_dir)
		self.output_dir.mkdir(parents=True, exist_ok=True)
		
		# 验证指标记录
		self.validation_history = []
		self.best_metrics = {
			'dice': 0.0,
			'iou': 0.0,
			'hausdorff': float('inf'),
			'connectivity': 0.0
		}
		
		# 早停参数
		self.patience = 5
		self.no_improve_count = 0
		
		self.logger = logging.getLogger(__name__)
	
	def validate_epoch(self, loss_fn, epoch: int) -> Dict[str, float]:
		"""
		验证一个epoch - 复用训练流水线
		所有rank参与验证，rank 0负责最终结果收集
		"""
		# 设置验证模式
		self.model.eval()
		
		# 验证指标累积
		total_loss = 0.0
		total_dice = 0.0
		total_iou = 0.0
		total_hausdorff = 0.0
		total_connectivity = 0.0
		num_batches = 0
		
		predictions = []
		targets = []
		
		with torch.no_grad():
			for batch_idx, batch in enumerate(self.val_loader):
				try:
					# 数据预处理（与训练相同）
					if isinstance(batch, (list, tuple)) and len(batch) >= 2:
						images, labels = batch[0], batch[1]
					else:
						images, labels = batch, None
					
					images = images.to(self.device, non_blocking=True)
					if labels is not None:
						labels = labels.to(self.device, non_blocking=True)
					
					# 前向传播（走完整7卡流水线）
					outputs = self.model.forward(images)
					
					# 计算损失
					if labels is not None:
						loss = loss_fn(outputs, labels)
						total_loss += loss.item()
						
						# 计算医学指标
						dice = MedicalMetrics.dice_coefficient(outputs, labels)
						iou = MedicalMetrics.iou_score(outputs, labels)
						hausdorff = MedicalMetrics.hausdorff_distance(outputs, labels)
						connectivity = MedicalMetrics.vessel_connectivity_score(outputs, labels)
						
						total_dice += dice
						total_iou += iou
						total_hausdorff += hausdorff if hausdorff != float('inf') else 0
						total_connectivity += connectivity
						
						# 保存预测结果（用于可视化）
						if batch_idx < 5:  # 只保存前5个batch的结果
							predictions.append(outputs.cpu())
							targets.append(labels.cpu())
					
					num_batches += 1
					
					if batch_idx % 10 == 0:
						self.logger.info(f"Validation Epoch {epoch}, Batch {batch_idx}/{len(self.val_loader)}")
				
				except Exception as e:
					self.logger.error(f"Validation batch {batch_idx} failed: {e}")
					continue
		
		# 计算当前rank的平均指标
		if num_batches > 0:
			rank_metrics = {
				'loss': total_loss / num_batches,
				'dice': total_dice / num_batches,
				'iou': total_iou / num_batches,
				'hausdorff': total_hausdorff / num_batches,
				'connectivity': total_connectivity / num_batches,
				'num_batches': num_batches
			}
		else:
			rank_metrics = {
				'loss': 0.0,
				'dice': 0.0,
				'iou': 0.0,
				'hausdorff': 0.0,
				'connectivity': 0.0,
				'num_batches': 0
			}
		
		# 跨rank聚合指标
		global_metrics = self._aggregate_metrics_across_ranks(rank_metrics)
		
		# rank 0负责最终处理
		if self.rank == 0:
			# 记录验证历史
			self.validation_history.append(global_metrics)
			
			# 更新最佳指标
			self._update_best_metrics(global_metrics)
			
			# 保存可视化结果
			if predictions:
				self._save_validation_visualizations(predictions, targets, epoch)
			
			# 检查早停
			should_stop = self._check_early_stopping(global_metrics)
			if should_stop:
				self.logger.info(f"Early stopping triggered at epoch {epoch}")
			
			# 打印详细结果
			self._print_validation_results(global_metrics, epoch)
		
		return global_metrics
	
	def _aggregate_metrics_across_ranks(self, rank_metrics: Dict[str, float]) -> Dict[str, float]:
		"""跨rank聚合验证指标"""
		import torch.distributed as dist
		
		# 准备张量用于聚合
		metrics_tensor = torch.tensor([
			rank_metrics['loss'],
			rank_metrics['dice'],
			rank_metrics['iou'],
			rank_metrics['hausdorff'],
			rank_metrics['connectivity'],
			rank_metrics['num_batches']
		], device=self.device)
		
		# 收集所有rank的指标
		if dist.is_initialized():
			gathered_metrics = [torch.zeros_like(metrics_tensor) for _ in range(self.world_size)]
			dist.all_gather(gathered_metrics, metrics_tensor)
		else:
			gathered_metrics = [metrics_tensor]
		
		# 加权平均计算全局指标
		total_batches = sum(m[5].item() for m in gathered_metrics)
		if total_batches == 0:
			return rank_metrics
		
		global_metrics = {
			'loss': sum(m[0].item() * m[5].item() for m in gathered_metrics) / total_batches,
			'dice': sum(m[1].item() * m[5].item() for m in gathered_metrics) / total_batches,
			'iou': sum(m[2].item() * m[5].item() for m in gathered_metrics) / total_batches,
			'hausdorff': sum(m[3].item() * m[5].item() for m in gathered_metrics) / total_batches,
			'connectivity': sum(m[4].item() * m[5].item() for m in gathered_metrics) / total_batches,
		}
		
		return global_metrics
	
	def _update_best_metrics(self, metrics: Dict[str, float]):
		"""更新最佳指标"""
		improved = False
		
		if metrics['dice'] > self.best_metrics['dice']:
			self.best_metrics['dice'] = metrics['dice']
			improved = True
		
		if metrics['iou'] > self.best_metrics['iou']:
			self.best_metrics['iou'] = metrics['iou']
			improved = True
		
		if metrics['hausdorff'] < self.best_metrics['hausdorff']:
			self.best_metrics['hausdorff'] = metrics['hausdorff']
			improved = True
		
		if metrics['connectivity'] > self.best_metrics['connectivity']:
			self.best_metrics['connectivity'] = metrics['connectivity']
			improved = True
		
		if improved:
			self.no_improve_count = 0
		else:
			self.no_improve_count += 1
	
	def _check_early_stopping(self, metrics: Dict[str, float]) -> bool:
		"""检查是否需要早停"""
		return self.no_improve_count >= self.patience
	
	def _save_validation_visualizations(self, predictions: List[torch.Tensor],
	                                    targets: List[torch.Tensor], epoch: int):
		"""保存验证结果可视化"""
		try:
			fig, axes = plt.subplots(3, len(predictions), figsize=(len(predictions) * 4, 12))
			if len(predictions) == 1:
				axes = axes.reshape(-1, 1)
			
			for i, (pred, target) in enumerate(zip(predictions, targets)):
				# 取中间层进行可视化
				if pred.dim() == 5:  # [B, C, D, H, W]
					mid_slice = pred.shape[2] // 2
					pred_slice = pred[0, 0, mid_slice].numpy()
					target_slice = target[0, 0, mid_slice].numpy()
				else:
					pred_slice = pred[0, 0].numpy()
					target_slice = target[0, 0].numpy()
				
				# 原图
				axes[0, i].imshow(target_slice, cmap='gray')
				axes[0, i].set_title(f'Ground Truth {i + 1}')
				axes[0, i].axis('off')
				
				# 预测
				axes[1, i].imshow(pred_slice, cmap='gray')
				axes[1, i].set_title(f'Prediction {i + 1}')
				axes[1, i].axis('off')
				
				# 差异
				diff = np.abs(pred_slice - target_slice)
				axes[2, i].imshow(diff, cmap='hot')
				axes[2, i].set_title(f'Difference {i + 1}')
				axes[2, i].axis('off')
			
			plt.tight_layout()
			plt.savefig(self.output_dir / f'validation_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
			plt.close()
		
		except Exception as e:
			self.logger.warning(f"保存可视化失败: {e}")
	
	def _print_validation_results(self, metrics: Dict[str, float], epoch: int):
		"""打印验证结果"""
		self.logger.info("=" * 60)
		self.logger.info(f"Validation Results - Epoch {epoch}")
		self.logger.info("=" * 60)
		self.logger.info(f"Loss:         {metrics['loss']:.6f}")
		self.logger.info(f"Dice:         {metrics['dice']:.4f}")
		self.logger.info(f"IoU:          {metrics['iou']:.4f}")
		self.logger.info(f"Hausdorff:    {metrics['hausdorff']:.2f}")
		self.logger.info(f"Connectivity: {metrics['connectivity']:.4f}")
		self.logger.info("-" * 60)
		self.logger.info(f"Best Dice:         {self.best_metrics['dice']:.4f}")
		self.logger.info(f"Best IoU:          {self.best_metrics['iou']:.4f}")
		self.logger.info(f"Best Hausdorff:    {self.best_metrics['hausdorff']:.2f}")
		self.logger.info(f"Best Connectivity: {self.best_metrics['connectivity']:.4f}")
		self.logger.info(f"No improve count:  {self.no_improve_count}/{self.patience}")
		self.logger.info("=" * 60)


def integrate_validation_to_training():
	"""
	将验证逻辑集成到训练循环的示例代码

	在 distributed_train.py 的主训练循环中使用：
	"""
	example_code = """
def main():
    # ... 其他初始化代码 ...

    # 创建验证器
    validator = DistributedValidator(
        model=model,
        val_loader=val_loader,
        device=device,
        rank=rank,
        world_size=world_size,
        output_dir=args.output_dir
    )

    # 主训练循环
    for epoch in range(start_epoch, args.epochs):
        # 设置epoch（用于DistributedSampler）
        train_sampler.set_epoch(epoch)

        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn,
                                device, epoch, args, scaler)

        # 验证（每几个epoch一次）
        if epoch % args.val_interval == 0:
            val_metrics = validator.validate_epoch(loss_fn, epoch)

            if rank == 0:
                logger.info(f'Epoch {epoch}: Train Loss: {train_loss:.6f}')

                # 检查是否需要早停
                if validator._check_early_stopping(val_metrics):
                    logger.info("Early stopping triggered, saving final checkpoint...")
                    save_checkpoint(model, optimizer, epoch, args.output_dir, rank)
                    break

        # 更新学习率
        scheduler.step()

        # 保存检查点
        if epoch % args.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, args.output_dir, rank)

    # 清理
    model.stop_worker()
    cleanup_distributed()
    """
	return example_code


# 使用示例：
if __name__ == "__main__":
	print("验证逻辑完整实现代码已生成")
	print("请将 DistributedValidator 类集成到您的训练脚本中")
	print("集成示例：")
	print(integrate_validation_to_training())