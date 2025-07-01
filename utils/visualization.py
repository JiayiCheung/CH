import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap


class Visualizer:
	"""可视化工具"""
	
	def __init__(self, output_dir=None, dpi=150, cmap='viridis'):
		"""
		初始化可视化器

		参数:
			output_dir: 输出目录
			dpi: 图像DPI
			cmap: 默认颜色图
		"""
		self.output_dir = Path(output_dir) if output_dir else None
		if self.output_dir:
			self.output_dir.mkdir(exist_ok=True, parents=True)
		
		self.dpi = dpi
		self.cmap = cmap
		
		# 创建自定义颜色图
		self.vessel_cmap = LinearSegmentedColormap.from_list(
			"vessel_cmap", ["black", "red"], N=256
		)
		
		self.tumor_cmap = LinearSegmentedColormap.from_list(
			"tumor_cmap", ["black", "green"], N=256
		)
	
	def visualize_slice(self, volume, slice_idx=None, axis=0, title=None, cmap=None, save_path=None):
		"""
		可视化3D体积的切片

		参数:
			volume: 3D体积
			slice_idx: 切片索引，如果为None则使用中间切片
			axis: 切片轴向 (0=z, 1=y, 2=x)
			title: 标题
			cmap: 颜色图
			save_path: 保存路径

		返回:
			如果save_path为None，返回图像对象；否则保存图像
		"""
		# 转换为numpy数组
		if torch.is_tensor(volume):
			volume = volume.cpu().numpy()
		
		# 去除批次和通道维度 (如果存在)
		if volume.ndim > 3:
			volume = volume.squeeze()
		
		# 如果依然大于3D，可能是多通道图像
		if volume.ndim > 3:
			raise ValueError("Volume has too many dimensions after squeezing")
		
		# 确定切片索引
		if slice_idx is None:
			slice_idx = volume.shape[axis] // 2
		
		# 提取切片
		if axis == 0:
			slice_data = volume[slice_idx]
		elif axis == 1:
			slice_data = volume[:, slice_idx]
		else:
			slice_data = volume[:, :, slice_idx]
		
		# 创建图像
		fig, ax = plt.subplots(figsize=(10, 10))
		im = ax.imshow(slice_data, cmap=cmap or self.cmap)
		plt.colorbar(im, ax=ax)
		
		if title:
			ax.set_title(title)
		
		ax.set_axis_off()
		
		# 保存或返回
		if save_path:
			plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
			plt.close(fig)
		else:
			return fig
	
	def visualize_segmentation(self, image, pred, target=None, slice_idx=None, axis=0,
	                           title=None, alpha=0.5, save_path=None):
		"""
		可视化分割结果

		参数:
			image: 原始图像
			pred: 预测掩码
			target: 目标掩码 (可选)
			slice_idx: 切片索引
			axis: 切片轴向
			title: 标题
			alpha: 掩码透明度
			save_path: 保存路径

		返回:
			如果save_path为None，返回图像对象；否则保存图像
		"""
		# 转换为numpy数组
		if torch.is_tensor(image):
			image = image.cpu().numpy()
		if torch.is_tensor(pred):
			pred = pred.cpu().numpy()
		if target is not None and torch.is_tensor(target):
			target = target.cpu().numpy()
		
		# 去除批次和通道维度
		if image.ndim > 3:
			image = image.squeeze()
		if pred.ndim > 3:
			pred = pred.squeeze()
		if target is not None and target.ndim > 3:
			target = target.squeeze()
		
		# 确定切片索引
		if slice_idx is None:
			slice_idx = image.shape[axis] // 2
		
		# 提取切片
		if axis == 0:
			image_slice = image[slice_idx]
			pred_slice = pred[slice_idx]
			target_slice = target[slice_idx] if target is not None else None
		elif axis == 1:
			image_slice = image[:, slice_idx]
			pred_slice = pred[:, slice_idx]
			target_slice = target[:, slice_idx] if target is not None else None
		else:
			image_slice = image[:, :, slice_idx]
			pred_slice = pred[:, :, slice_idx]
			target_slice = target[:, :, slice_idx] if target is not None else None
		
		# 创建图像
		if target is not None:
			fig, axes = plt.subplots(1, 3, figsize=(15, 5))
			
			# 原始图像
			axes[0].imshow(image_slice, cmap='gray')
			axes[0].set_title("原始图像")
			axes[0].set_axis_off()
			
			# 预测掩码
			axes[1].imshow(image_slice, cmap='gray')
			axes[1].imshow(pred_slice, cmap=self.vessel_cmap, alpha=alpha)
			axes[1].set_title("预测掩码")
			axes[1].set_axis_off()
			
			# 目标掩码
			axes[2].imshow(image_slice, cmap='gray')
			axes[2].imshow(target_slice, cmap=self.vessel_cmap, alpha=alpha)
			axes[2].set_title("目标掩码")
			axes[2].set_axis_off()
			
			if title:
				fig.suptitle(title)
		else:
			fig, axes = plt.subplots(1, 2, figsize=(10, 5))
			
			# 原始图像
			axes[0].imshow(image_slice, cmap='gray')
			axes[0].set_title("原始图像")
			axes[0].set_axis_off()
			
			# 预测掩码
			axes[1].imshow(image_slice, cmap='gray')
			axes[1].imshow(pred_slice, cmap=self.vessel_cmap, alpha=alpha)
			axes[1].set_title("预测掩码")
			axes[1].set_axis_off()
			
			if title:
				fig.suptitle(title)
		
		# 保存或返回
		if save_path:
			plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
			plt.close(fig)
		else:
			return fig
	
	def visualize_3d(self, mask, threshold=0.5, title=None, save_path=None):
		"""
		创建3D体积渲染可视化 (使用matplotlib)

		参数:
			mask: 3D掩码
			threshold: 二值化阈值
			title: 标题
			save_path: 保存路径

		返回:
			如果save_path为None，返回图像对象；否则保存图像
		"""
		# 转换为numpy数组
		if torch.is_tensor(mask):
			mask = mask.cpu().numpy()
		
		# 去除批次和通道维度
		if mask.ndim > 3:
			mask = mask.squeeze()
		
		# 二值化
		binary_mask = mask > threshold
		
		# 使用matplotlib的3D功能
		from mpl_toolkits.mplot3d import Axes3D
		
		# 找到非零点
		z, y, x = np.nonzero(binary_mask)
		
		# 创建图像
		fig = plt.figure(figsize=(10, 10))
		ax = fig.add_subplot(111, projection='3d')
		
		# 绘制点云
		ax.scatter(x, y, z, c='r', marker='.', alpha=0.1)
		
		# 设置轴标签和标题
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		
		if title:
			ax.set_title(title)
		
		# 保存或返回
		if save_path:
			plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
			plt.close(fig)
		else:
			return fig
	
	def visualize_ch_coefficients(self, ch_coeffs, max_n, max_k, max_l, title=None, save_path=None):
		"""
		可视化CH系数

		参数:
			ch_coeffs: CH系数 [B, C, 2*max_n+1, max_k, 2*max_l+1]
			max_n: 最大角谐波阶数
			max_k: 最大径向阶数
			max_l: 最大轴向阶数
			title: 标题
			save_path: 保存路径

		返回:
			如果save_path为None，返回图像对象；否则保存图像
		"""
		# 转换为numpy数组
		if torch.is_tensor(ch_coeffs):
			ch_coeffs = ch_coeffs.cpu().numpy()
		
		# 计算系数幅度
		coeffs_mag = np.abs(ch_coeffs)
		
		# 选择第一个批次和通道
		if coeffs_mag.ndim > 3:
			coeffs_mag = coeffs_mag[0, 0]
		
		# 创建子图
		fig, axes = plt.subplots(max_k, 1, figsize=(15, 5 * max_k))
		if max_k == 1:
			axes = [axes]
		
		# 遍历每个k值
		for k_idx in range(max_k):
			# 创建网格表示n和l
			n_grid, l_grid = np.meshgrid(
				np.arange(-max_n, max_n + 1),
				np.arange(-max_l, max_l + 1)
			)
			
			# 绘制热图
			im = axes[k_idx].pcolormesh(
				n_grid, l_grid, coeffs_mag[:, k_idx, :].T,
				cmap='hot', shading='auto'
			)
			
			# 设置标题和标签
			axes[k_idx].set_title(f'k = {k_idx + 1}')
			axes[k_idx].set_xlabel('Angular Order (n)')
			axes[k_idx].set_ylabel('Axial Order (l)')
			
			# 添加颜色条
			plt.colorbar(im, ax=axes[k_idx])
		
		# 设置总标题
		if title:
			fig.suptitle(title)
		
		plt.tight_layout()
		
		# 保存或返回
		if save_path:
			plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
			plt.close(fig)
		else:
			return fig
	
	def visualize_attention_maps(self, attention_maps, image=None, title=None, save_path=None):
		"""
		可视化注意力图

		参数:
			attention_maps: 注意力图 [B, C, D, H, W]
			image: 原始图像 (可选)
			title: 标题
			save_path: 保存路径

		返回:
			如果save_path为None，返回图像对象；否则保存图像
		"""
		# 转换为numpy数组
		if torch.is_tensor(attention_maps):
			attention_maps = attention_maps.cpu().numpy()
		if image is not None and torch.is_tensor(image):
			image = image.cpu().numpy()
		
		# 去除批次维度
		if attention_maps.ndim > 4:
			attention_maps = attention_maps[0]
		if image is not None and image.ndim > 3:
			image = image[0]
		
		# 获取通道数
		num_channels = attention_maps.shape[0]
		
		# 选择中间切片
		slice_idx = attention_maps.shape[1] // 2
		
		# 创建子图
		fig, axes = plt.subplots(1, num_channels, figsize=(5 * num_channels, 5))
		if num_channels == 1:
			axes = [axes]
		
		# 绘制每个通道的注意力图
		for c in range(num_channels):
			# 获取当前通道的中间切片
			attention_slice = attention_maps[c, slice_idx]
			
			# 如果提供了原始图像，作为背景
			if image is not None:
				if image.ndim == 3:
					image_slice = image[slice_idx]
				else:
					image_slice = image[0, slice_idx]  # 假设第一个通道
				
				axes[c].imshow(image_slice, cmap='gray')
				im = axes[c].imshow(attention_slice, cmap='jet', alpha=0.7)
			else:
				im = axes[c].imshow(attention_slice, cmap='jet')
			
			axes[c].set_title(f'Channel {c + 1}')
			axes[c].set_axis_off()
			
			plt.colorbar(im, ax=axes[c])
		
		# 设置总标题
		if title:
			fig.suptitle(title)
		
		plt.tight_layout()
		
		# 保存或返回
		if save_path:
			plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
			plt.close(fig)
		else:
			return fig
	
	def create_multi_tier_visualization(self, tier_images, tier_preds, tier_targets=None,
	                                    title=None, save_path=None):
		"""
		创建多tier可视化

		参数:
			tier_images: 字典 {tier: image}
			tier_preds: 字典 {tier: pred}
			tier_targets: 字典 {tier: target} (可选)
			title: 标题
			save_path: 保存路径

		返回:
			如果save_path为None，返回图像对象；否则保存图像
		"""
		# 确定tier数量和是否有目标
		tiers = sorted(tier_images.keys())
		has_targets = tier_targets is not None
		
		# 创建图像
		cols = 2 if not has_targets else 3
		fig, axes = plt.subplots(len(tiers), cols, figsize=(5 * cols, 5 * len(tiers)))
		
		# 处理单tier情况
		if len(tiers) == 1:
			axes = axes.reshape(1, -1)
		
		# 绘制每个tier
		for i, tier in enumerate(tiers):
			# 获取中间切片
			image = tier_images[tier]
			pred = tier_preds[tier]
			target = tier_targets[tier] if has_targets else None
			
			# 转换为numpy数组
			if torch.is_tensor(image):
				image = image.cpu().numpy()
			if torch.is_tensor(pred):
				pred = pred.cpu().numpy()
			if target is not None and torch.is_tensor(target):
				target = target.cpu().numpy()
			
			# 去除批次和通道维度
			if image.ndim > 3:
				image = image.squeeze()
			if pred.ndim > 3:
				pred = pred.squeeze()
			if target is not None and target.ndim > 3:
				target = target.squeeze()
			
			# 获取中间切片
			slice_idx = image.shape[0] // 2
			image_slice = image[slice_idx]
			pred_slice = pred[slice_idx]
			target_slice = target[slice_idx] if target is not None else None
			
			# 绘制原始图像
			axes[i, 0].imshow(image_slice, cmap='gray')
			axes[i, 0].set_title(f'Tier-{tier}: 原始图像')
			axes[i, 0].set_axis_off()
			
			# 绘制预测掩码
			axes[i, 1].imshow(image_slice, cmap='gray')
			axes[i, 1].imshow(pred_slice, cmap=self.vessel_cmap, alpha=0.5)
			axes[i, 1].set_title(f'Tier-{tier}: 预测掩码')
			axes[i, 1].set_axis_off()
			
			# 绘制目标掩码 (如果有)
			if has_targets:
				axes[i, 2].imshow(image_slice, cmap='gray')
				axes[i, 2].imshow(target_slice, cmap=self.vessel_cmap, alpha=0.5)
				axes[i, 2].set_title(f'Tier-{tier}: 目标掩码')
				axes[i, 2].set_axis_off()
		
		# 设置总标题
		if title:
			fig.suptitle(title)
		
		plt.tight_layout()
		
		# 保存或返回
		if save_path:
			plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
			plt.close(fig)
		else:
			return fig