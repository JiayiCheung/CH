import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CHAttention(nn.Module):
	"""优化的CH系数注意力机制"""
	
	def __init__(self, max_n, max_k, max_l):
		"""
		初始化CH系数注意力机制

		参数:
			max_n: 最大角谐波阶数
			max_k: 最大径向阶数
			max_l: 最大轴向阶数
		"""
		super().__init__()
		
		# 系数维度
		self.max_n = max_n
		self.max_k = max_k
		self.max_l = max_l
		
		# 输入特征维度
		ch_dims = (2 * max_n + 1, max_k, 2 * max_l + 1)
		self.ch_dims = ch_dims
		
		# 降维投影
		self.projection = nn.Sequential(
			nn.Conv3d(2, 8, kernel_size=1),  # 复数表示为2通道(实部+虚部)
			nn.InstanceNorm3d(8),
			nn.ReLU(inplace=True)
		)
		
		# 轻量级注意力网络 (使用深度可分离卷积减少参数)
		self.attention = nn.Sequential(
			# 深度可分离卷积
			nn.Conv3d(8, 8, kernel_size=3, padding=1, groups=8),
			nn.Conv3d(8, 16, kernel_size=1),
			nn.InstanceNorm3d(16),
			nn.ReLU(inplace=True),
			# 第二层深度可分离卷积
			nn.Conv3d(16, 16, kernel_size=3, padding=1, groups=16),
			nn.Conv3d(16, 2, kernel_size=1),  # 输出实部和虚部的注意力权重
			nn.Sigmoid()  # 输出范围[0,1]
		)
		
		# 频率编码层 (注入物理先验知识)
		self.frequency_encoding = self._create_frequency_encoding()
	
	def _create_frequency_encoding(self):
		"""
		创建频率编码层，编码物理先验知识

		返回:
			频率编码张量
		"""
		# 创建编码张量
		freq_encoding = torch.ones(
			(1, 2, *self.ch_dims),
			dtype=torch.float32
		)
		
		# 在编码中注入先验知识
		for n_idx, n in enumerate(range(-self.max_n, self.max_n + 1)):
			for k_idx, k in enumerate(range(1, self.max_k + 1)):
				for l_idx, l in enumerate(range(-self.max_l, self.max_l + 1)):
					# 低频增强
					if abs(n) <= 1 and k <= 2 and abs(l) <= 2:
						# 增强低频分量 (对应全局结构)
						freq_encoding[0, :, n_idx, k_idx, l_idx] = 1.2
					
					# 血管方向性增强 (n=±1 对应90°/270°血管方向)
					if abs(n) == 1:
						freq_encoding[0, :, n_idx, k_idx, l_idx] = 1.3
					
					# 降低高频噪声
					if abs(n) > 3 or k > self.max_k - 1 or abs(l) > self.max_l - 1:
						freq_encoding[0, :, n_idx, k_idx, l_idx] = 0.8
		
		# 注册为缓冲区，不作为模型参数更新
		return nn.Parameter(freq_encoding, requires_grad=True)
	
	def forward(self, ch_coeffs):
		"""
		应用注意力机制

		参数:
			ch_coeffs: CH系数 [B, C, 2*max_n+1, max_k, 2*max_l+1]

		返回:
			增强后的CH系数
		"""
		B, C = ch_coeffs.shape[:2]
		
		# 分离实部和虚部
		real = ch_coeffs.real
		imag = ch_coeffs.imag
		
		# 处理每个通道
		enhanced_coeffs = []
		
		for c in range(C):
			# 创建输入特征 [B, 2, 2*max_n+1, max_k, 2*max_l+1]
			coeffs_input = torch.stack([real[:, c], imag[:, c]], dim=1)
			
			# 应用频率编码
			encoded = coeffs_input * self.frequency_encoding.to(coeffs_input.device)
			
			# 特征提取与注意力生成
			features = self.projection(encoded)
			attention_weights = self.attention(features)
			
			# 分离实部和虚部注意力
			real_attention = attention_weights[:, 0:1]
			imag_attention = attention_weights[:, 1:2]
			
			# 应用注意力
			enhanced_real = real[:, c:c + 1] * real_attention
			enhanced_imag = imag[:, c:c + 1] * imag_attention
			
			# 残差连接
			output_real = enhanced_real + real[:, c:c + 1]
			output_imag = enhanced_imag + imag[:, c:c + 1]
			
			# 创建复数输出
			enhanced_c = torch.complex(output_real, output_imag)
			enhanced_coeffs.append(enhanced_c)
		
		# 合并通道
		return torch.cat(enhanced_coeffs, dim=1)
	
	def tier_specific_enhancement(self, ch_coeffs, tier):
		"""
		根据tier应用特定的增强

		参数:
			ch_coeffs: CH系数
			tier: tier级别 (0, 1, 2)

		返回:
			增强后的CH系数
		"""
		# 基础增强
		enhanced = self.forward(ch_coeffs)
		
		# 应用tier特定的后处理
		if tier == 0:  # 器官级别
			# 强调全局结构，降低高频成分
			for n_idx, n in enumerate(range(-self.max_n, self.max_n + 1)):
				if abs(n) > 2:
					enhanced[:, :, n_idx] *= 0.8  # 降低高频角谐波
				
				for k_idx, k in enumerate(range(1, self.max_k + 1)):
					if k > 3:
						enhanced[:, :, :, k_idx] *= 0.8  # 降低高频径向分量
		
		elif tier == 2:  # 细节级别
			# 强调细节，增强高频成分
			for n_idx, n in enumerate(range(-self.max_n, self.max_n + 1)):
				if abs(n) >= 2:
					enhanced[:, :, n_idx] *= 1.2  # 增强高频角谐波
				
				for k_idx, k in enumerate(range(1, self.max_k + 1)):
					if k >= 3:
						enhanced[:, :, :, k_idx] *= 1.2  # 增强高频径向分量
		
		# tier-1保持默认平衡设置
		
		return enhanced