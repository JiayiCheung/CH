import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import time
from typing import Dict, Tuple, Any, Optional, List


class BaseStage(nn.Module):
	"""处理阶段基类，所有具体阶段继承自此类"""
	
	def __init__(self, name: str, device: str):
		"""
		初始化基础处理阶段

		参数:
			name: 阶段名称
			device: 设备(例如'cuda:0')
		"""
		super().__init__()
		self.name = name
		self.device = device
		self.training = True
		
		# 性能监控
		self.compute_time = 0
		self.transfer_time = 0
		self.batch_count = 0
		
		print(f"[{self.name}] 初始化在 {self.device}")
	
	def train(self, mode=True):
		"""设置训练模式"""
		self.training = mode
		return super().train(mode)
	
	def eval(self):
		"""设置评估模式"""
		self.training = False
		return super().eval()
	
	def reset_stats(self):
		"""重置性能统计"""
		self.compute_time = 0
		self.transfer_time = 0
		self.batch_count = 0
	
	def get_stats(self):
		"""获取性能统计"""
		avg_compute = self.compute_time / max(1, self.batch_count)
		avg_transfer = self.transfer_time / max(1, self.batch_count)
		return {
			'avg_compute_ms': avg_compute * 1000,
			'avg_transfer_ms': avg_transfer * 1000,
			'batch_count': self.batch_count
		}
	
	def forward(self, *args, **kwargs):
		"""前向处理(由子类实现)"""
		raise NotImplementedError("子类必须实现forward方法")
	
	def backward(self, grad_output, saved_tensors=None):
		"""反向传播(由子类实现)"""
		raise NotImplementedError("子类必须实现backward方法")
	
	def get_state_dict_prefix(self):
		"""获取带前缀的参数字典(由子类实现)"""
		raise NotImplementedError("子类必须实现get_state_dict_prefix方法")


class FrontendStage(BaseStage):
	"""
	前端处理阶段 (GPU 0)

	职责:
	- 输入预处理
	- 3D FFT变换
	- 柱坐标映射
	"""
	
	def __init__(self, model, device):
		"""
		初始化前端处理阶段

		参数:
			model: VesselSegmenter模型
			device: 设备(例如'cuda:0')
		"""
		super().__init__("FrontendStage", device)
		
		# 提取模型组件
		self.fft_utils = model.ch_branch.fft_utils
		self.cylindrical_mapping = model.ch_branch.cylindrical_mapping
		
		# 移动到指定设备
		self.fft_utils.to(device)
		self.cylindrical_mapping.to(device)
		
		# 当前tier
		self.current_tier = None
	
	def set_tier(self, tier):
		"""设置当前tier"""
		self.current_tier = tier
	
	def forward(self, x, amp_enabled=False):
		"""
		前向处理

		参数:
			x: 输入数据 [B, C, D, H, W]
			amp_enabled: 是否启用混合精度

		返回:
			tuple: (cylindrical_spectrum, x)
		"""
		# 记录开始时间
		start_time = time.time()
		
		# 确保数据在正确的设备上
		transfer_start = time.time()
		x = x.to(self.device)
		self.transfer_time += time.time() - transfer_start
		
		with autocast(enabled=amp_enabled):
			# 执行3D FFT
			spectrum = self.fft_utils.fft3d(x, apply_window=True)
			
			# 执行柱坐标映射
			cylindrical_spectrum = self.cylindrical_mapping.cartesian_to_cylindrical(spectrum)
		
		# 更新计算时间和批次计数
		self.compute_time += time.time() - start_time
		self.batch_count += x.size(0)
		
		return cylindrical_spectrum, x
	
	def backward(self, grad_output, saved_tensors=None):
		"""
		反向传播

		参数:
			grad_output: 输出梯度
			saved_tensors: 保存的中间结果

		返回:
			输入梯度
		"""
		# 在实际实现中，这里需要实现FFT和柱坐标映射的反向传播
		# 由于这些操作在PyTorch中已经是可微分的，大部分情况下我们可以使用自动微分
		
		# 简化实现：直接返回None表示梯度传播到输入结束
		return None
	
	def get_state_dict_prefix(self):
		"""
		获取带前缀的参数字典

		返回:
			带前缀的参数字典
		"""
		state_dict = {}
		
		# 提取并添加前缀
		for name, param in self.fft_utils.state_dict().items():
			state_dict[f'ch_branch.fft_utils.{name}'] = param
		
		for name, param in self.cylindrical_mapping.state_dict().items():
			state_dict[f'ch_branch.cylindrical_mapping.{name}'] = param
		
		return state_dict


class CHProcessingStage(BaseStage):
	"""
	CH处理阶段 (GPU 1)

	职责:
	- CH分解
	- CH系数注意力
	"""
	
	def __init__(self, model, device):
		"""
		初始化CH处理阶段

		参数:
			model: VesselSegmenter模型
			device: 设备(例如'cuda:1')
		"""
		super().__init__("CHProcessingStage", device)
		
		# 提取模型组件
		self.ch_transform = model.ch_branch.ch_transform
		self.ch_attention = model.ch_branch.ch_attention
		self.tier_params = model.tier_params
		
		# 移动到指定设备
		self.ch_transform.to(device)
		self.ch_attention.to(device)
		
		# 当前tier
		self.current_tier = None
	
	def set_tier(self, tier):
		"""设置当前tier"""
		self.current_tier = tier
	
	def forward(self, data, amp_enabled=False):
		"""
		前向处理

		参数:
			data: 输入数据(cylindrical_spectrum, input_reference)
			amp_enabled: 是否启用混合精度

		返回:
			tuple: (enhanced_coeffs, (cylindrical_spectrum, ch_coeffs))
		"""
		# 记录开始时间
		start_time = time.time()
		
		cylindrical_spectrum, _ = data
		
		# 确保数据在正确的设备上
		transfer_start = time.time()
		cylindrical_spectrum = cylindrical_spectrum.to(self.device)
		self.transfer_time += time.time() - transfer_start
		
		with autocast(enabled=amp_enabled):
			# 获取tier特定的r_scale
			r_scale = 1.0
			if self.current_tier is not None and self.current_tier in self.tier_params:
				r_scale = self.tier_params[self.current_tier].get('r_scale', 1.0)
			
			# 执行CH分解
			ch_coeffs = self.ch_transform.decompose(cylindrical_spectrum, r_scale=r_scale)
			
			# 应用CH系数注意力
			if self.current_tier is not None:
				enhanced_coeffs = self.ch_attention.tier_specific_enhancement(ch_coeffs, self.current_tier)
			else:
				enhanced_coeffs = self.ch_attention(ch_coeffs)
		
		# 更新计算时间和批次计数
		self.compute_time += time.time() - start_time
		self.batch_count += cylindrical_spectrum.size(0)
		
		return enhanced_coeffs, (cylindrical_spectrum, ch_coeffs)
	
	def backward(self, grad_output, saved_tensors=None):
		"""
		反向传播

		参数:
			grad_output: 输出梯度
			saved_tensors: 保存的中间结果

		返回:
			输入梯度
		"""
		# 简化实现：直接返回None表示梯度传播到此阶段结束
		return None
	
	def get_state_dict_prefix(self):
		"""
		获取带前缀的参数字典

		返回:
			带前缀的参数字典
		"""
		state_dict = {}
		
		# 提取并添加前缀
		for name, param in self.ch_transform.state_dict().items():
			state_dict[f'ch_branch.ch_transform.{name}'] = param
		
		for name, param in self.ch_attention.state_dict().items():
			state_dict[f'ch_branch.ch_attention.{name}'] = param
		
		return state_dict


class SpatialFusionStage(BaseStage):
	"""
	空间处理与融合阶段 (GPU 2)

	职责:
	- 空间分支处理
	- 特征融合
	"""
	
	def __init__(self, model, device):
		"""
		初始化空间处理与融合阶段

		参数:
			model: VesselSegmenter模型
			device: 设备(例如'cuda:2')
		"""
		super().__init__("SpatialFusionStage", device)
		
		# 提取模型组件
		self.spatial_branch = model.spatial_branch
		self.attention_fusion = model.attention_fusion
		self.edge_enhance = model.edge_enhance
		self.ch_branch = {
			'ch_transform': model.ch_branch.ch_transform,
			'fft_utils': model.ch_branch.fft_utils,
			'cylindrical_mapping': model.ch_branch.cylindrical_mapping,
			'cylindrical_dims': model.ch_branch.cylindrical_dims
		}
		self.tier_params = model.tier_params
		
		# 移动到指定设备
		self.spatial_branch.to(device)
		self.attention_fusion.to(device)
		self.edge_enhance.to(device)
		
		# 当前tier
		self.current_tier = None
	
	def set_tier(self, tier):
		"""设置当前tier"""
		self.current_tier = tier
	
	def forward(self, data, input_x, amp_enabled=False):
		"""
		前向处理

		参数:
			data: 输入数据(enhanced_coeffs, ch_tensors)
			input_x: 原始输入
			amp_enabled: 是否启用混合精度

		返回:
			tuple: (fused, (ch_features, spatial_features))
		"""
		# 记录开始时间
		start_time = time.time()
		
		enhanced_coeffs, ch_tensors = data
		
		# 确保数据在正确的设备上
		transfer_start = time.time()
		enhanced_coeffs = enhanced_coeffs.to(self.device)
		input_x = input_x.to(self.device)
		self.transfer_time += time.time() - transfer_start
		
		with autocast(enabled=amp_enabled):
			# 重构CH特征
			ch_features = self._reconstruct_ch_features(enhanced_coeffs, ch_tensors[0])
			
			# 执行空间分支处理
			edge_feat = self.edge_enhance(input_x)
			spatial_features = self.spatial_branch(input_x)
			
			# 特征融合
			fused = self.attention_fusion(ch_features, spatial_features)
		
		# 更新计算时间和批次计数
		self.compute_time += time.time() - start_time
		self.batch_count += input_x.size(0)
		
		return fused, (ch_features, spatial_features)
	
	def _reconstruct_ch_features(self, enhanced_coeffs, cylindrical_spectrum):
		"""
		从CH系数重构特征

		参数:
			enhanced_coeffs: 增强的CH系数
			cylindrical_spectrum: 原始柱坐标频谱

		返回:
			重构的CH特征
		"""
		# 获取tier特定的r_scale
		r_scale = 1.0
		if self.current_tier is not None and self.current_tier in self.tier_params:
			r_scale = self.tier_params[self.current_tier].get('r_scale', 1.0)
		
		# 获取柱坐标维度
		r_samples, theta_samples, z_samples = self.ch_branch['cylindrical_dims']
		
		# 执行逆CH变换
		reconstructed_cylindrical = self.ch_branch['ch_transform'].reconstruct(
			enhanced_coeffs,
			(r_samples, theta_samples, z_samples),
			r_scale=r_scale
		)
		
		# 执行柱坐标逆映射
		input_shape = cylindrical_spectrum.shape[2:5]
		reconstructed_spectrum = self.ch_branch['cylindrical_mapping'].cylindrical_to_cartesian(
			reconstructed_cylindrical,
			input_shape
		)
		
		# 执行3D逆FFT
		ch_features = self.ch_branch['fft_utils'].ifft3d(reconstructed_spectrum)
		
		return ch_features
	
	def backward(self, grad_output, saved_tensors=None):
		"""
		反向传播

		参数:
			grad_output: 输出梯度
			saved_tensors: 保存的中间结果

		返回:
			输入梯度的元组
		"""
		# 简化实现：直接返回None表示梯度传播到此阶段结束
		return None, None
	
	def get_state_dict_prefix(self):
		"""
		获取带前缀的参数字典

		返回:
			带前缀的参数字典
		"""
		state_dict = {}
		
		# 提取并添加前缀
		for name, param in self.spatial_branch.state_dict().items():
			state_dict[f'spatial_branch.{name}'] = param
		
		for name, param in self.attention_fusion.state_dict().items():
			state_dict[f'attention_fusion.{name}'] = param
		
		for name, param in self.edge_enhance.state_dict().items():
			state_dict[f'edge_enhance.{name}'] = param
		
		return state_dict


class BackendStage(BaseStage):
	"""
	后端处理阶段 (GPU 3)

	职责:
	- 多尺度融合
	- 分割头处理
	- 损失计算
	"""
	
	def __init__(self, model, device):
		"""
		初始化后端处理阶段

		参数:
			model: VesselSegmenter模型
			device: 设备(例如'cuda:3')
		"""
		super().__init__("BackendStage", device)
		
		# 提取模型组件
		self.multiscale_fusion = model.multiscale_fusion
		self.seg_head_first = model.seg_head_first
		self.seg_head_tail = model.seg_head_tail
		
		# 移动到指定设备
		self.multiscale_fusion.to(device)
		if self.seg_head_first is not None:
			self.seg_head_first.to(device)
		self.seg_head_tail.to(device)
		
		# 保存tier特征
		self.tier_features = {}
		
		# 当前tier
		self.current_tier = None
	
	def set_tier(self, tier):
		"""设置当前tier"""
		self.current_tier = tier
	
	def clear_tier_features(self):
		"""清除tier特征缓存"""
		self.tier_features.clear()
	
	def forward(self, data, amp_enabled=False):
		"""
		前向处理

		参数:
			data: 输入数据(fused, fusion_tensors)
			amp_enabled: 是否启用混合精度

		返回:
			分割结果
		"""
		# 记录开始时间
		start_time = time.time()
		
		fused, _ = data
		
		# 确保数据在正确的设备上
		transfer_start = time.time()
		fused = fused.to(self.device)
		self.transfer_time += time.time() - transfer_start
		
		with autocast(enabled=amp_enabled):
			# 保存当前tier特征
			if self.current_tier is not None:
				self.tier_features[self.current_tier] = fused
			
			# 多尺度融合 (如果有多个tier)
			if len(self.tier_features) > 1:
				final = self.multiscale_fusion(self.tier_features)
			else:
				final = fused
			
			# 分割头处理
			if self.seg_head_first is None:
				# 延迟构建分割头
				self._build_seg_head(final.shape[1], final)
			
			# 确保分割头在正确的设备上
			if self.seg_head_first.weight.device != self.device:
				self.seg_head_first = self.seg_head_first.to(self.device)
			
			logits = self.seg_head_tail(self.seg_head_first(final))
		
		# 更新计算时间和批次计数
		self.compute_time += time.time() - start_time
		self.batch_count += fused.size(0)
		
		return logits
	
	def _build_seg_head(self, in_c, ref):
		"""
		构建分割头

		参数:
			in_c: 输入通道数
			ref: 参考张量(用于获取设备和dtype)
		"""
		self.seg_head_first = nn.Conv3d(in_c, 32, 3, padding=1, bias=False)
		self.seg_head_first.to(ref.device, dtype=ref.dtype)
	
	def backward(self, grad_output, saved_tensors=None):
		"""
		反向传播

		参数:
			grad_output: 输出梯度
			saved_tensors: 保存的中间结果

		返回:
			输入梯度
		"""
		# 简化实现：直接返回None表示梯度传播到此阶段结束
		return None
	
	def compute_metrics(self, outputs, targets):
		"""
		计算评估指标

		参数:
			outputs: 模型输出
			targets: 目标值

		返回:
			指标字典
		"""
		from utils.metrics import SegmentationMetrics
		
		# 确保数据在正确的设备上
		outputs = outputs.to(self.device)
		targets = targets.to(self.device)
		
		# 二值化预测
		preds = (outputs > 0.5).float()
		
		# 计算指标
		metrics = SegmentationMetrics.evaluate_all(preds, targets, include_advanced=False)
		
		return metrics
	
	def get_state_dict_prefix(self):
		"""
		获取带前缀的参数字典

		返回:
			带前缀的参数字典
		"""
		state_dict = {}
		
		# 提取并添加前缀
		for name, param in self.multiscale_fusion.state_dict().items():
			state_dict[f'multiscale_fusion.{name}'] = param
		
		if self.seg_head_first is not None:
			for name, param in self.seg_head_first.state_dict().items():
				state_dict[f'seg_head_first.{name}'] = param
		
		for name, param in self.seg_head_tail.state_dict().items():
			state_dict[f'seg_head_tail.{name}'] = param
		
		return state_dict