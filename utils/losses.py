import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
	"""
	Dice损失函数
	"""
	
	def __init__(self, smooth=1.0):
		super(DiceLoss, self).__init__()
		self.smooth = smooth
	
	def forward(self, inputs, targets):
		# 展平输入和目标
		inputs_flat = inputs.view(-1)
		targets_flat = targets.view(-1)
		
		# 计算交集
		intersection = (inputs_flat * targets_flat).sum()
		
		# 计算Dice系数
		dice = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
		
		return 1 - dice





class FocalDiceLoss(nn.Module):
	"""
	Focal Dice损失
	增加对难分类样本的关注
	"""
	
	def __init__(self, gamma=1.0, smooth=1.0):
		super(FocalDiceLoss, self).__init__()
		self.gamma = gamma
		self.smooth = smooth
	
	def forward(self, inputs, targets):
		# 展平输入和目标
		inputs_flat = inputs.view(-1)
		targets_flat = targets.view(-1)
		
		# 计算Dice系数
		intersection = (inputs_flat * targets_flat).sum()
		dice = (2. * intersection + self.smooth) / (
				inputs_flat.sum() + targets_flat.sum() + self.smooth)
		
		# 应用Focal调制
		focal_factor = (1 - dice) ** self.gamma
		
		return focal_factor * (1 - dice)





class CombinedLoss(nn.Module):
    """
    简化版组合损失（Dice + Focal BCE），接受 logits 输入

    参数:
        alpha: Dice loss 与 Focal BCE 的加权系数（默认 0.5）
        gamma: Focal loss 的聚焦因子（默认 2.0）

    用法:
        logits = model(x)
        loss = CombinedLoss()(logits, targets)
    """

    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        targets = targets.float()

        # 1. Focal BCE with logits
        bce = self.bce(logits, targets)
        pt = torch.exp(-bce)  # pt = p or 1-p
        focal_factor = (1 - pt) ** self.gamma
        focal_bce = focal_factor * bce
        focal_bce = focal_bce.mean()

        # 2. Dice loss (soft dice on sigmoid probs)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice = (2. * intersection + 1e-5) / (probs.sum() + targets.sum() + 1e-5)
        dice_loss = 1 - dice

        # 3. Final loss
        return self.alpha * dice_loss + (1 - self.alpha) * focal_bce





class DiceCELossWithLogits(nn.Module):
    """
    L = w_dice * Dice( sigmoid/softmax(logits), y ) + w_ce * CE(logits, y)

    - 二分类：CE = BCEWithLogitsLoss(pos_weight)；targets 形状可为 [B,1,D,H,W] 或 [B,D,H,W]
    - 多分类：CE = CrossEntropyLoss(weight, ignore_index)；targets 形状为 [B,D,H,W] (long)
    - 仅接受 logits；模型末尾不要再 sigmoid/softmax

    参数:
        w_dice, w_ce: 两个项的权重
        smooth: Dice 平滑项
        include_background: 多分类时是否计入背景通道（默认计入）
        pos_weight: 二分类正类权重（标量或 1D Tensor），用于严重不平衡
        class_weights: 多分类各类权重（形状 [C] 的 1D Tensor）
        ignore_index: 多分类的忽略标签（如 255）；Dice 也会对应地忽略
    """
    def __init__(self,
                 w_dice: float = 0.5,
                 w_ce: float = 0.5,
                 smooth: float = 1e-5,
                 include_background: bool = True,
                 pos_weight=None,
                 class_weights=None,
                 ignore_index=None):
        super().__init__()
        self.w_dice = w_dice
        self.w_ce = w_ce
        self.smooth = smooth
        self.include_background = include_background
        self.pos_weight = pos_weight
        self.class_weights = class_weights
        self.ignore_index = ignore_index

    @staticmethod
    def _dice_on_probs(probs: torch.Tensor, target_onehot: torch.Tensor, smooth: float) -> torch.Tensor:
        # probs/target_onehot: [B,C,D,H,W]
        dims = tuple(range(2, probs.ndim))  # D,H,W
        inter = (probs * target_onehot).sum(dims)
        denom = probs.sum(dims) + target_onehot.sum(dims)
        dice_per_c = (2.0 * inter + smooth) / (denom + smooth)  # [B,C]
        return 1.0 - dice_per_c.mean()  # scalar

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C = logits.shape[:2]
        device = logits.device

        if C == 1:
            # -------- Binary --------
            # 统一 targets 形状为 [B,1,D,H,W]
            if targets.ndim == logits.ndim - 1:
                targets = targets.unsqueeze(1)
            targets = targets.float()

            # CE
            if self.pos_weight is None:
                ce = F.binary_cross_entropy_with_logits(logits, targets)
            else:
                pos_w = self.pos_weight
                if not torch.is_tensor(pos_w):
                    pos_w = torch.tensor([float(pos_w)], device=device)
                ce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_w)

            # Dice（在概率上算）
            probs = torch.sigmoid(logits)
            dice = self._dice_on_probs(probs, targets, self.smooth)

            return self.w_dice * dice + self.w_ce * ce

        else:
            # -------- Multi-class --------
            # CE（CrossEntropyLoss 内置 log-softmax）
            ce = F.cross_entropy(
                logits,
                targets.long(),
                weight=self.class_weights,
                ignore_index=self.ignore_index if self.ignore_index is not None else -100  # PyTorch 默认 -100
            )

            # Dice：softmax 概率 + one-hot
            probs = torch.softmax(logits, dim=1)
            num_classes = C
            tgt = targets.long()

            # 忽略标签 mask（同时用于 dice）
            if self.ignore_index is not None:
                mask = (tgt != self.ignore_index).unsqueeze(1).float()  # [B,1,D,H,W]
            else:
                mask = None

            onehot = F.one_hot(tgt.clamp_min(0), num_classes=num_classes)  # [B,D,H,W,C]
            onehot = onehot.permute(0, 4, 1, 2, 3).float()                 # [B,C,D,H,W]

            if not self.include_background and num_classes > 1:
                probs  = probs[:, 1:, ...]
                onehot = onehot[:, 1:, ...]

            if mask is not None:
                probs  = probs  * mask
                onehot = onehot * mask

            dice = self._dice_on_probs(probs, onehot, self.smooth)
            return self.w_dice * dice + self.w_ce * ce
