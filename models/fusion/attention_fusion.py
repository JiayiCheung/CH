import torch
import torch.nn as nn
import torch.nn.functional as F


class BandPooling(nn.Module):
    """
    频带池化（支持 3D 径向频带 或 1D-沿最后一维频率的带通）.
    输入:
      x : [B, C, D, H, W] 空域张量，或已在频域的半谱张量（最后一维为 Wf）
      band_edges:
        - 3D 径向比例边界: [K, 2]，值域在 [0,1]（左闭右开），按归一化径向频率构造 [K, Df, Hf, Wf] 掩模
        - 1D 掩模 / 边界: [K, 2]（索引或比例→索引）或 [K, Wf]（0/1掩模），仅沿最后一维分带
          * 若传入全谱长度 [K, Wfull] (Wfull = 2*(Wf-1))，将自动裁切到半谱长度 Wf
    输出:
      tokens: [B, K, C_token] （默认 C_token=C；若 concat real/imag 则 2*C）
    """
    def __init__(self, use_magnitude: bool = True, eps: float = 1e-6):
        super().__init__()
        self.use_magnitude = use_magnitude
        self.eps = eps

    @staticmethod
    def _to_freq(x: torch.Tensor):
        """
        将空域/频域混合输入统一为 3D 半谱:
          - 若 x 是实数空域: rFFTN → [B,C,Df,Hf,Wf]
          - 若 x 已是复数频域半谱: 原样返回
          - 若 x 已是实数半谱（极少见）: 直接返回
        """
        if torch.is_complex(x):
            return x
        # 实数空域 → 3D 半谱
        return torch.fft.rfftn(x, dim=(-3, -2, -1))

    @staticmethod
    def _mag_or_ri(X: torch.Tensor, use_magnitude: bool) -> torch.Tensor:
        """
        复数频谱 → 实数表示:
          - use_magnitude=True: |X|
          - use_magnitude=False: concat(real, imag) 到通道维
        """
        if X.is_complex():
            if use_magnitude:
                return X.abs()
            # concat real/imag
            B, C, Df, Hf, Wf = X.shape
            Xri = torch.view_as_real(X)              # [B,C,Df,Hf,Wf,2]
            Xri = Xri.movedim(-1, 1).reshape(B, C * 2, Df, Hf, Wf)
            return Xri
        return X

    @staticmethod
    def _build_mask_1d(band_edges, Wf: int, device, dtype):
        """
        仅沿最后一维的 1D 掩模:
          - [K,2]（索引或比例）→ [K,Wf] 0/1
          - [K,Wf] → 校正为 0/1
          - [K,Wfull] (Wfull=2*(Wf-1)) → 自动裁切到 [:, :Wf]
        """
        be = torch.as_tensor(band_edges, device=device)
        if be.ndim == 2 and be.shape[1] == 2:
            # 边界可能是比例或索引
            if be.is_floating_point() and be.max() <= 1.0001 and be.min() >= -1e-6:
                idx = torch.clamp((be * (Wf - 1)).round().long(), 0, Wf - 1)
            else:
                idx = be.round().long().clamp(0, Wf - 1)
            K = idx.size(0)
            mask = torch.zeros((K, Wf), device=device, dtype=dtype)
            for k in range(K):
                l, r = int(idx[k, 0]), int(idx[k, 1])
                r = max(r, l)  # 防御; 右开
                r = min(r, Wf)
                if r > l:
                    mask[k, l:r] = 1
            return mask
        if be.ndim == 2:
            # [K, W?]
            K, Wany = be.shape
            if Wany == Wf:
                return (be > 0).to(dtype).to(device)
            # 全谱长度 → 裁半谱
            Wfull = 2 * (Wf - 1)
            if Wany == Wfull:
                return (be[:, :Wf] > 0).to(dtype).to(device)
        raise ValueError(
            f"[BandPooling] 1D band_edges 需 [K,2] 或 [K,Wf]/[K,Wfull], got {tuple(be.shape)}"
        )

    @staticmethod
    def _build_mask_radial3d(band_edges, Df: int, Hf: int, Wf: int, device, dtype):
        """
        3D 径向掩模: band_edges 为比例边界 [K,2], 值域 [0,1].
        按归一化半径 r∈[0,1] 生成 [K,Df,Hf,Wf] 的 0/1 掩模。
        """
        be = torch.as_tensor(band_edges, device=device, dtype=torch.float32)
        if not (be.ndim == 2 and be.shape[1] == 2):
            raise ValueError("[BandPooling] 径向模式需 [K,2] 比例边界")
        # 频率坐标（注意 Wf 为半谱）
        fd = torch.fft.fftfreq(Df, d=1.0).to(device)
        fh = torch.fft.fftfreq(Hf, d=1.0).to(device)
        fw = torch.fft.rfftfreq(2 * (Wf - 1), d=1.0).to(device)
        zz, yy, xx = torch.meshgrid(fd, fh, fw, indexing="ij")
        r = torch.sqrt(zz * zz + yy * yy + xx * xx)
        r = (r / r.max().clamp_min(1e-12)).clamp(0, 1)

        masks = []
        for k in range(be.size(0)):
            l, rgt = float(be[k, 0]), float(be[k, 1])
            masks.append(((r >= l) & (r < rgt)).to(dtype))
        return torch.stack(masks, dim=0).to(device)  # [K,Df,Hf,Wf]

    def forward(self, x: torch.Tensor, band_edges) -> torch.Tensor:
        """
        生成频带 token: [B, K, C_token]
        """
        device = x.device
        # 统一为频域半谱
        X = self._to_freq(x)                    # [B,C,Df,Hf,Wf] (complex or real)
        B, C, Df, Hf, Wf = X.shape

        # 实数化：幅值或 real/imag 拼接
        Xr = self._mag_or_ri(X, self.use_magnitude)  # [B,C',Df,Hf,Wf]
        B, Cprime, Df, Hf, Wf = Xr.shape

        # 自动判断 band_edges 类型:
        be = torch.as_tensor(band_edges, device=device)
        # case A: 3D 径向比例 [K,2] （优先 3D）
        radial_mode = (be.ndim == 2 and be.shape[1] == 2 and
                       be.is_floating_point() and be.max() <= 1.0001 and be.min() >= -1e-6)

        if radial_mode:
            mask = self._build_mask_radial3d(be, Df, Hf, Wf, device, Xr.dtype)  # [K,Df,Hf,Wf]
            K = mask.size(0)
            m = mask.view(1, 1, K, Df, Hf, Wf)
        else:
            # 1D（沿最后一维）模式
            mask1d = self._build_mask_1d(be, Wf=Wf, device=device, dtype=Xr.dtype)  # [K,Wf]
            K = mask1d.size(0)
            m = mask1d.view(1, 1, K, 1, 1, Wf)  # 广播到 D/H

        # 频带聚合 → token
        # Xr: [B,C',Df,Hf,Wf], m: [1,1,K,Df,Hf,Wf] 或 [1,1,K,1,1,Wf]
        Xm = Xr.unsqueeze(2) * m                               # [B,C',K,Df,Hf,Wf]
        num = Xm.sum(dim=(-3, -2, -1))                         # [B,C',K]
        den = m.sum(dim=(-3, -2, -1)).clamp_min(self.eps)      # [1,1,K]
        tok = (num / den).transpose(1, 2).contiguous()         # [B,K,C']
        return tok


class MultiHeadAttention(nn.Module):
    """标准多头注意力"""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key, value, attn_mask=None):
        B = query.size(0)
        q = self.q_proj(query).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key  ).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, -1, self.embed_dim)
        return self.out_proj(out)


class FreqGuidedCrossAttn(nn.Module):
    """
    频域引导的跨模态注意力:
      - fd_feat: [B,C_fd,D,H,W]  (空间域)
      - sp_feat: [B,C_sp,D',H',W'](空间域)
      - band_edges: 同 BandPooling 说明
    返回:
      [B, out_channels, D, H, W]
    """
    def __init__(
        self,
        fd_channels: int,
        sp_channels: int,
        out_channels: int = None,
        num_bands: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        fdbranch=None
    ):
        super().__init__()
        self.out_channels = out_channels or (fd_channels + sp_channels)
        self.fdbranch = fdbranch
        self.num_bands = num_bands

        hidden = 128
        self.fd_proj = nn.Linear(fd_channels, hidden)
        self.sp_proj = nn.Linear(sp_channels, hidden)

        self.band_pooling = BandPooling(use_magnitude=True, eps=1e-6)

        self.cross_attn = MultiHeadAttention(embed_dim=hidden, num_heads=num_heads, dropout=dropout)
        self.norm_q = nn.LayerNorm(hidden)
        self.norm_o = nn.LayerNorm(hidden)

        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.Dropout(dropout),
        )
        self.out_proj = nn.Linear(hidden, self.out_channels)

        self.res_proj = None
        if self.out_channels != sp_channels:
            self.res_proj = nn.Linear(sp_channels, self.out_channels)

    def forward(self, fd_feat, sp_feat, band_edges=None):
        # 动态频带边界
        if band_edges is None and self.fdbranch is not None:
            band_edges = self.fdbranch.get_band_edges()

        # 允许 1D 边界向 [K,2] 转换
        if band_edges is not None:
            be = torch.as_tensor(band_edges, device=fd_feat.device, dtype=torch.float32)
            if be.ndim == 1 and be.numel() >= 2:
                band_edges = torch.stack([be[:-1], be[1:]], dim=1)

        # 尺寸对齐: 以 fd_feat 的 D,H,W 为准
        if fd_feat.shape[2:] != sp_feat.shape[2:]:
            sp_feat = F.interpolate(sp_feat, size=fd_feat.shape[2:], mode="trilinear", align_corners=False)

        B, C_fd, D, H, W = fd_feat.shape
        _, C_sp, _, _, _ = sp_feat.shape

        # 1) 频带 token
        fd_tokens = self.band_pooling(fd_feat, band_edges)          # [B,K,C_fd] (或 [B,K,2*C_fd] 若 concat RI)
        if self.num_bands is not None:
            assert fd_tokens.size(1) == self.num_bands, \
                f"[FreqGuidedCrossAttn] num_bands={self.num_bands}, but got {fd_tokens.size(1)} bands from band_edges."
        fd_tokens = self.fd_proj(fd_tokens)                          # [B,K,hidden]

        # 2) 空间序列化
        sp_tokens = sp_feat.flatten(2).transpose(1, 2)               # [B,N,C_sp]
        sp_tokens_res = sp_tokens
        sp_tokens = self.sp_proj(sp_tokens)                          # [B,N,hidden]

        # 3) Cross-Attention (Q=space, K/V=freq)
        attn_out = self.cross_attn(
            query=self.norm_q(sp_tokens),
            key=fd_tokens,
            value=fd_tokens
        )                                                            # [B,N,hidden]
        x = sp_tokens + attn_out
        x = x + self.mlp(self.norm_o(x))                             # Transformer 块

        # 4) 输出与残差
        out = self.out_proj(x)                                       # [B,N,out_ch]
        if self.res_proj is not None:
            res = self.res_proj(sp_tokens_res)                       # [B,N,out_ch]
            out = out + res
        else:
            # 只有在 out_channels == sp_channels 时才能直接相加
            assert sp_tokens_res.size(-1) == out.size(-1), \
                "[FreqGuidedCrossAttn] out_channels 必须等于 sp_channels 才能直接残差相加"
            out = out + sp_tokens_res

        # 5) 还原体素形状
        out = out.transpose(1, 2).reshape(B, self.out_channels, D, H, W)
        return out


