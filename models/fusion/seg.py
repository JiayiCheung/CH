import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
	"""双卷积模块: Conv -> InstanceNorm -> LeakyReLU -> Conv -> InstanceNorm -> LeakyReLU"""
	
	def __init__(self, in_channels, out_channels, mid_channels=None):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
		self.norm1 = nn.InstanceNorm3d(mid_channels)
		self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
		self.norm2 = nn.InstanceNorm3d(out_channels)
		self.act = nn.LeakyReLU(0.01, inplace=True)
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.norm1(x)
		x = self.act(x)
		x = self.conv2(x)
		x = self.norm2(x)
		return self.act(x)


class DownsampleBlock(nn.Module):
	"""下采样模块: MaxPool -> ConvBlock"""
	
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.pool = nn.MaxPool3d(2)
		self.conv = ConvBlock(in_channels, out_channels)
	
	def forward(self, x):
		x = self.pool(x)
		return self.conv(x)


class UpsampleBlock(nn.Module):
	"""上采样模块: Upsample -> Conv -> Concat -> ConvBlock"""
	
	def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
		super().__init__()
		self.bilinear = bilinear
		
		if bilinear:
			# 使用双线性插值上采样
			self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
			self.conv1 = nn.Conv3d(in_channels, in_channels // 2, kernel_size=1)
			# 上采样后的通道数
			up_channels = in_channels // 2
		else:
			# 使用转置卷积上采样
			self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
			self.conv1 = nn.Identity()
			up_channels = in_channels // 2
		
		# 拼接后的通道数 = 上采样后的通道数 + 跳跃连接的通道数
		self.conv2 = ConvBlock(up_channels + skip_channels, out_channels)
	
	def forward(self, x1, x2):
		x1 = self.up(x1)
		x1 = self.conv1(x1)
		
		# 处理输入尺寸不匹配的情况
		diff_z = x2.size()[2] - x1.size()[2]
		diff_y = x2.size()[3] - x1.size()[3]
		diff_x = x2.size()[4] - x1.size()[4]
		
		x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
		                diff_y // 2, diff_y - diff_y // 2,
		                diff_z // 2, diff_z - diff_z // 2])
		
		# 拼接跳跃连接
		x = torch.cat([x2, x1], dim=1)
		return self.conv2(x)


class Seg(nn.Module):
	"""可配置深度的U-Net分割头架构"""
	
	def __init__(self, in_channels, out_channels, depth=4, base_features=32, feature_scale=2, bilinear=True):
		"""
		初始化可配置的分割头

		参数:
			in_channels (int): 输入通道数
			out_channels (int): 输出通道数
			depth (int): U-Net的深度，即下采样/上采样的层数
			base_features (int): 第一层的特征数
			feature_scale (int): 每层特征数的缩放因子
			bilinear (bool): 是否使用双线性插值进行上采样
		"""
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.depth = depth
		self.bilinear = bilinear
		
		# 计算每层的特征数
		features = [int(base_features * (feature_scale ** i)) for i in range(depth + 1)]
		
		# 初始卷积块
		self.inc = ConvBlock(in_channels, features[0])
		
		# 下采样路径 - 动态创建下采样层
		self.down_layers = nn.ModuleList()
		for i in range(depth):
			self.down_layers.append(DownsampleBlock(features[i], features[i + 1]))
		
		# 底部瓶颈卷积
		self.bottleneck = ConvBlock(features[-1], features[-1])
		
		# 上采样路径 - 动态创建上采样层
		self.up_layers = nn.ModuleList()
		for i in range(depth):
			# 注意索引顺序是倒序的
			in_ch = features[depth - i]
			skip_ch = features[depth - i - 1]
			out_ch = features[depth - i - 1]
			
			self.up_layers.append(UpsampleBlock(
				in_channels=in_ch,
				skip_channels=skip_ch,
				out_channels=out_ch,
				bilinear=bilinear
			))
		
		# 输出层
		self.outc = nn.Sequential(
			nn.InstanceNorm3d(features[0]),
			nn.ReLU(inplace=True),
			nn.Conv3d(features[0], out_channels, kernel_size=1)
		)
		
		# 初始化权重
		self._init_weights()
	
	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.InstanceNorm3d):
				if m.weight is not None:
					nn.init.constant_(m.weight, 1)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
	
	def forward(self, x):
		# 保存编码器特征以便跳跃连接
		encoder_features = [self.inc(x)]
		
		# 编码器路径
		for i, down in enumerate(self.down_layers):
			encoder_features.append(down(encoder_features[-1]))
		
		# 瓶颈
		x = self.bottleneck(encoder_features[-1])
		
		# 解码器路径
		for i, up in enumerate(self.up_layers):
			# 注意索引是倒序的
			skip_feature = encoder_features[-(i + 2)]  # 获取对应的编码器特征
			x = up(x, skip_feature)
		
		# 输出层
		logits = self.outc(x)
		
		return logits