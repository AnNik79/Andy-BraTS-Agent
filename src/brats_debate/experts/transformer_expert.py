import inspect
import torch
from torch import nn
import torch.nn.functional as F
from .base import Expert, ResidualBlock, ChannelSummary, group_norm, maybe_pool


class Attention3d(nn.Module):
    """Explicit QKV attention so MPS uses matmul/softmax rather than nested-transformer kernels."""

    def __init__(self, dim, heads, dropout=.1):
        super().__init__()
        if dim % heads:
            raise ValueError("transformer_dim must be divisible by transformer_heads")
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.norm = group_norm(dim)
        self.qkv = nn.Conv3d(dim, dim * 3, 1)
        self.proj = nn.Conv3d(dim, dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        b, c, d, h, w = x.shape
        n = d * h * w
        if n <= 1:
            return residual
        qkv = self.qkv(x).reshape(b, 3, self.heads, c // self.heads, n)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = self.drop((q.transpose(-2, -1) @ k * self.scale).softmax(-1))
        out = (v @ attn.transpose(-2, -1)).reshape(b, c, d, h, w)
        return residual + self.proj(out)


class WindowAttention3d(nn.Module):
    def __init__(self, dim, heads, window, dropout=.1):
        super().__init__()
        self.window = window
        self.attn = Attention3d(dim, heads, dropout)

    def forward(self, x):
        window = self.window
        if min(x.shape[2:]) <= window:
            return self.attn(x)
        b, c, d, h, w = x.shape
        pad = [(window - s % window) % window for s in (d, h, w)]
        if any(pad):
            x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]))
        pd, ph, pw = x.shape[2:]
        windows = (x.view(b, c, pd // window, window, ph // window, window, pw // window, window)
                     .permute(0, 2, 4, 6, 1, 3, 5, 7)
                     .reshape(-1, c, window, window, window))
        windows = self.attn(windows)
        x = (windows.view(b, pd // window, ph // window, pw // window, c, window, window, window)
             .permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(b, c, pd, ph, pw))
        return x[:, :, :d, :h, :w]


class TransformerExpert(Expert):
    """Hierarchical 3D encoder-decoder with windowed mid-level and global bottleneck attention.

    TransBTS / nnFormer inspired: local residual convs + attention on downsampled tokens.
    Attention is only applied where the token count is tractable.
    """

    def __init__(self, in_channels, classes, width=16, feature_channels=4, dropout=.1,
                 dim=48, heads=4, layers=2, grid=(4, 4, 4), window=4):
        super().__init__()
        del grid  # kept in the signature for config compatibility; hierarchy uses /4 and /8.
        self.stem = ResidualBlock(in_channels, width, dropout)
        self.enc1 = ResidualBlock(width, width * 2, dropout)
        self.enc2 = ResidualBlock(width * 2, dim, dropout)
        self.mid_attn = WindowAttention3d(dim, heads, window, dropout)
        self.enc3 = ResidualBlock(dim, dim, dropout)
        self.bot_attn = nn.ModuleList([Attention3d(dim, heads, dropout) for _ in range(layers)])
        self.dec2 = ResidualBlock(dim + dim, dim, dropout)
        self.dec1 = ResidualBlock(dim + width * 2, width * 2, dropout)
        self.dec0 = ResidualBlock(width * 2 + width, width, dropout)
        self.head = nn.Conv3d(width, classes, 1)
        self.compact = ChannelSummary(width, feature_channels)

    def raw_forward(self, image):
        s0 = self.stem(image)
        s1 = self.enc1(maybe_pool(s0))
        s2 = self.mid_attn(self.enc2(maybe_pool(s1)))
        b = self.enc3(maybe_pool(s2))
        for attn in self.bot_attn:
            b = attn(b)
        d2 = self.dec2(torch.cat([s2, F.interpolate(b, size=s2.shape[2:], mode="trilinear", align_corners=False)], 1))
        d1 = self.dec1(torch.cat([s1, F.interpolate(d2, size=s1.shape[2:], mode="trilinear", align_corners=False)], 1))
        f = self.dec0(torch.cat([s0, F.interpolate(d1, size=s0.shape[2:], mode="trilinear", align_corners=False)], 1))
        return self.head(f), self.compact(f), {}


class SwinExpert(Expert):
    def __init__(self, in_channels, classes, feature_channels, feature_size, crop_size, dropout):
        super().__init__()
        try:
            from monai.networks.nets import SwinUNETR
        except ImportError as exc:
            raise ImportError("Install brats-debate[monai] to use transformer_backend: swin") from exc
        if any(s % 32 or s < 64 for s in crop_size):
            raise ValueError("Swin crop_size must be multiples of 32 and >=64 for its instance normalization")
        kwargs = dict(in_channels=in_channels, out_channels=classes, feature_size=feature_size,
                      drop_rate=dropout, use_checkpoint=True)
        if "img_size" in inspect.signature(SwinUNETR).parameters:
            kwargs["img_size"] = tuple(crop_size)
        self.network = SwinUNETR(**kwargs)
        self.compact = ChannelSummary(feature_size, feature_channels)
        self._decoder_feature = None
        self.network.decoder1.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self._decoder_feature = output

    def raw_forward(self, image):
        logits = self.network(image)
        features = self.compact(self._decoder_feature)
        self._decoder_feature = None
        return logits, features, {}
