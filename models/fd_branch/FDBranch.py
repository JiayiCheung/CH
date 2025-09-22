import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Optional
from .fft_utils import ComplexLinear3D


class OrderedCutpoints(nn.Module):
	"""
	保证频率切割点有序：0 < r1 < r2 < r3 < 1
	通过参数化方式实现切割点的可学习性
	"""
	
	def __init__(self, init_cuts: Tuple[float, float, float] = (0.15, 0.3, 0.45)):
		super().__init__()
		# 使用未约束参数
		self.raw_r1 = nn.Parameter(torch.tensor(self._inverse_sigmoid(init_cuts[0])))
		self.raw_r2 = nn.Parameter(torch.tensor(self._inverse_sigmoid(init_cuts[1])))
		self.raw_r3 = nn.Parameter(torch.tensor(self._inverse_sigmoid(init_cuts[2])))
	
	def _inverse_sigmoid(self, x: float) -> float:
		"""sigmoid的逆函数，用于初始化"""
		x = max(min(x, 0.999), 0.001)  # 避免数值问题
		return math.log(x / (1 - x))
	
	def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""返回有序的切割点"""
		# 使用sigmoid确保范围在(0,1)内
		r1 = torch.sigmoid(self.raw_r1) * 0.35  # 限制在[0, 0.35]
		
		# r2在r1之后，使用剩余范围的一部分
		remaining_1 = 1.0 - r1
		r2 = r1 + torch.sigmoid(self.raw_r2) * remaining_1 * 0.6  # 限制r2-r1的最大值
		
		# r3在r2之后，使用剩余范围，但确保小于1
		remaining_2 = 1.0 - r2
		r3 = r2 + torch.sigmoid(self.raw_r3) * remaining_2 * 0.9  # 确保r3 < 1
		
		return r1, r2, r3


class DepthwiseSeparableConv3d(nn.Module):
	"""
	深度可分离3D卷积
	分别对每个输入通道应用空间卷积，然后使用1x1x1卷积混合通道
	"""
	
	def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
		super().__init__()
		self.depthwise = nn.Conv3d(
			channels,
			channels,
			kernel_size=kernel_size,
			padding=kernel_size // 2 * dilation,
			groups=channels,  # 分组卷积使每个通道单独卷积
			dilation=dilation,
			bias=False
		)
		
		self.pointwise = nn.Conv3d(
			channels,
			channels,
			kernel_size=1,
			bias=False
		)
		
		# 初始化接近于恒等映射
		nn.init.dirac_(self.depthwise.weight)
		nn.init.xavier_uniform_(self.pointwise.weight)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.depthwise(x)
		x = self.pointwise(x)
		return x


class ComplexBlock(nn.Module):
	"""
	频域增强模块，直接在复数频谱上操作
	使用ComplexLinear3D实现通道混合和特征提取
	"""
	
	def __init__(self, channels: int, expansion_ratio: int = 2):
		super().__init__()
		hidden_dim = channels * expansion_ratio
		
		# 复数线性层 (扩展通道)
		self.expand = ComplexLinear3D(channels, hidden_dim)
		
		# 复数线性层 (压缩回原通道)
		self.reduce = ComplexLinear3D(hidden_dim, channels)
		
		# 学习缩放参数
		self.gamma = nn.Parameter(torch.tensor(0.1))
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		x: 复数频谱 [B, C, D, H, W//2+1]
		"""
		identity = x
		
		# 通道扩展
		x = self.expand(x)
		
		# 非线性激活 (分别应用于实部和虚部)
		x = torch.complex(F.gelu(x.real), F.gelu(x.imag))
		
		# 通道压缩
		x = self.reduce(x)
		
		# 残差连接
		return identity + self.gamma * x


class FDBranch(nn.Module):
	"""
	频域特征增强分支（Frequency-Domain Branch v2.0）

	对输入3D体积进行频域变换，在频域中分段增强不同频率的特征，再逆变换回空间域。
	特别适用于医学影像中的血管增强，保持几何结构不变同时提高特征表达。

	参数:
		in_channels (int): 输入通道数
		out_channels (int): 输出通道数，默认与输入相同
		init_cuts (tuple): 初始频段切割点，默认为(0.15, 0.3, 0.45)
		init_scales (tuple): 初始频段增益系数，默认为(1.0, 1.2, 1.3, 0.8)
		transition_width (float): 频段过渡区宽度，默认为0.1
		expansion_ratio (int): 频域增强模块的通道扩展比例，默认为2

	输入:
		x (torch.Tensor): 形状为[B, C, D, H, W]的实数张量

	输出:
		torch.Tensor: 与输入形状相同的增强后实数张量
	"""
	
	def __init__(
			self,
			in_channels: int = 1,
			out_channels: Optional[int] = None,
			init_cuts: Tuple[float, float, float] = (0.15, 0.3, 0.45),
			init_scales: Tuple[float, float, float, float] = (1.0, 1.2, 1.3, 0.8),
			transition_width: float = 0.1,
			expansion_ratio: int = 2
	):
		super().__init__()
		
		# 设置输入/输出通道
		out_channels = out_channels or in_channels
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.transition_width = transition_width
		
		# 频段切割点
		self.cutpoints = OrderedCutpoints(init_cuts)
		
		# 频段增益系数（通过参数化确保在合理范围内）
		raw_scales = [self._inverse_scale(s) for s in init_scales]
		self.raw_band_scales = nn.Parameter(torch.tensor(raw_scales))
		
		# 频段处理模块 - 改用复数频域处理
		# 低频增强模块
		self.lf_enhance = ComplexBlock(in_channels, expansion_ratio)
		
		# 中频增强模块
		self.mf_enhance = ComplexBlock(in_channels, expansion_ratio)
		
		# 高频增强模块
		self.hf_enhance = ComplexBlock(in_channels, expansion_ratio)
		
		# 输出投影层（仅当输入/输出通道不同时使用）
		self.out_proj = (
			nn.Identity() if in_channels == out_channels else
			nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
		)
	
	def _inverse_scale(self, scale: float) -> float:
		"""将尺度[0.5,2.0]转换为未约束参数"""
		scale = min(max(scale, 0.5), 2.0)
		# 使用映射：scale = 0.5 + 1.5 * sigmoid(raw)
		normalized = (scale - 0.5) / 1.5
		return math.log(normalized / (1 - normalized))
	
	def _get_band_scales(self) -> torch.Tensor:
		"""获取约束后的频段增益系数"""
		# 转换为[0.5, 2.0]范围内的尺度
		return 0.5 + 1.5 * torch.sigmoid(self.raw_band_scales)
	
	
	
	
	def _create_freq_grid(self, shape: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
		"""创建归一化频率网格"""
		D, H, W_half = shape
		
		# 创建频率坐标
		kz = torch.fft.fftfreq(D, device=device)[:, None, None]
		ky = torch.fft.fftfreq(H, device=device)[None, :, None]
		kx = torch.fft.rfftfreq(2 * (W_half - 1), device=device)[None, None, :]
		
		# 计算归一化频率半径 [0,1]
		k_radius = torch.sqrt(kz ** 2 + ky ** 2 + kx ** 2)
		
		# 归一化到[0,1]
		k_max = math.sqrt(0.5 ** 2 + 0.5 ** 2 + 0.5 ** 2)  # 最大理论半径
		k_radius = k_radius / k_max
		
		return k_radius
	
	def _create_band_masks(
			self, k_radius: torch.Tensor
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""创建四个频段的平滑掩码"""
		# 获取有序的频率切割点
		r1, r2, r3 = self.cutpoints()
		tw = self.transition_width
		
		# 低频带掩码：[0, r1+tw]，平滑过渡
		lf_mask = 1.0 - self._sigmoid_mask(k_radius, r1, tw)
		
		# 中频带掩码：[r1-tw, r2+tw]，平滑过渡
		mf_mask = self._sigmoid_mask(k_radius, r1, tw) - self._sigmoid_mask(k_radius, r2, tw)
		
		# 高频带掩码：[r2-tw, r3+tw]，平滑过渡
		hf_mask = self._sigmoid_mask(k_radius, r2, tw) - self._sigmoid_mask(k_radius, r3, tw)
		
		# 极高频带掩码：[r3-tw, 1.0]，平滑过渡
		vhf_mask = self._sigmoid_mask(k_radius, r3, tw)
		
		return lf_mask, mf_mask, hf_mask, vhf_mask
	
	def _sigmoid_mask(self, x: torch.Tensor, center: torch.Tensor, width: float) -> torch.Tensor:
		"""创建基于sigmoid的平滑过渡掩码"""
		# 缩放系数使过渡区域宽度接近指定宽度
		scale = 4.0 / width
		return torch.sigmoid(scale * (x - center))
	
	def _apply_energy_conservation(
			self,
			orig_bands: List[torch.Tensor],
			enhanced_bands: List[torch.Tensor]
	) -> List[torch.Tensor]:
		"""应用分组能量守恒"""
		# 分组：低频, 中频+高频, 极高频
		groups = [
			[0],  # 低频
			[1, 2],  # 中频+高频
			[3]  # 极高频
		]
		
		results = enhanced_bands.copy()
		
		for group in groups:
			# 计算原始组能量
			orig_energy = sum(torch.sum(torch.abs(orig_bands[i]) ** 2) for i in group)
			
			# 计算增强后组能量
			enhanced_energy = sum(torch.sum(torch.abs(enhanced_bands[i]) ** 2) for i in group)
			
			# 计算能量比例，应用到增强后的频段
			if enhanced_energy > 0:
				scale = torch.sqrt(orig_energy / (enhanced_energy + 1e-8))
				for i in group:
					results[i] = enhanced_bands[i] * scale
		
		return results
	
	def get_band_edges(self):
		"""返回当前动态频带边界"""
		r1, r2, r3 = self.cutpoints()
		return [0.0, float(r1.detach().cpu()), float(r2.detach().cpu()), float(r3.detach().cpu()), 1.0]
	
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		频域增强前向传播

		输入:
			x: [B,C,D,H,W] 实数空间域张量

		输出:
			[B,C,D,H,W] 实数空间域张量，频域增强后
		"""
		# 保存原始形状，用于逆变换
		orig_shape = x.shape
		B, C, D, H, W = orig_shape
		
		# 1. 执行FFT变换到频域
		spectrum = torch.fft.rfftn(x, dim=(-3, -2, -1))
		
		# 2. 创建频率网格和频段掩码
		k_radius = self._create_freq_grid((D, H, spectrum.shape[-1]), spectrum.device)
		lf_mask, mf_mask, hf_mask, vhf_mask = self._create_band_masks(k_radius)
		
		# 添加批次和通道维度，扩展掩码以便应用
		masks = [m.unsqueeze(0).unsqueeze(0) for m in [lf_mask, mf_mask, hf_mask, vhf_mask]]
		
		# 3. 分离不同频段的频谱
		lf_spectrum = spectrum * masks[0]
		mf_spectrum = spectrum * masks[1]
		hf_spectrum = spectrum * masks[2]
		vhf_spectrum = spectrum * masks[3]
		
		# 保存原始频段，用于能量守恒
		orig_bands = [lf_spectrum, mf_spectrum, hf_spectrum, vhf_spectrum]
		
		# 4. 获取频段增益系数
		band_scales = self._get_band_scales()
		
		# 5. 分别处理每个频段
		# 低频：复数频域增强
		enhanced_lf = self.lf_enhance(lf_spectrum) * band_scales[0]
		
		# 中频：复数频域增强
		enhanced_mf = self.mf_enhance(mf_spectrum) * band_scales[1]
		
		# 高频：复数频域增强
		enhanced_hf = self.hf_enhance(hf_spectrum) * band_scales[2]
		
		# 极高频：简单缩放（保持原设计）
		enhanced_vhf = vhf_spectrum * band_scales[3]
		
		# 6. 应用分组能量守恒
		enhanced_bands = self._apply_energy_conservation(
			orig_bands,
			[enhanced_lf, enhanced_mf, enhanced_hf, enhanced_vhf]
		)
		
		# 7. 组合增强后的频段
		enhanced_spectrum = sum(enhanced_bands)
		
		# 8. 逆变换回空间域
		enhanced_spatial = torch.fft.irfftn(enhanced_spectrum, s=orig_shape[-3:], dim=(-3, -2, -1))
		
		# 9. 应用输出投影（如果需要）
		output = self.out_proj(enhanced_spatial)
		
		return output