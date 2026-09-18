import math
import torch
from torch import nn
import torch.nn.functional as F


def uncertainty(probabilities):
    p = probabilities.float().clamp_min(1e-8)
    top = p.topk(2, dim=1).values
    entropy = -(p * p.log()).sum(1, keepdim=True) / math.log(p.shape[1])
    return {"entropy": entropy, "top_probability": top[:, :1], "margin": top[:, :1] - top[:, 1:2]}


def pack_output(logits, features, extra=None, probabilities=None):
    p = logits.float().softmax(1) if probabilities is None else probabilities
    stats = uncertainty(p.detach())
    return {"logits": logits, "probabilities": p, "segmentation": p.argmax(1),
            "confidence": stats["top_probability"], "uncertainty": stats["entropy"],
            "features": features, "extra": {**stats, **(extra or {})}}


class Expert(nn.Module):
    def segmentation_logits(self, image):
        """Minimal validation interface; implementations can skip auxiliary feature heads."""
        return self.raw_forward(image)[0]

    def forward(self, image):
        logits, features, extra = self.raw_forward(image)
        return pack_output(logits, features, extra)

    @torch.no_grad()
    def predict(self, image, mc_samples=1):
        states = {module: module.training for module in self.modules()}
        self.eval()
        try:
            if mc_samples == 1:
                result = self(image)
                result["extra"]["mutual_information"] = torch.zeros_like(result["uncertainty"])
                return result
            if mc_samples > 1:
                for module in self.modules():
                    if isinstance(module, (nn.Dropout, nn.Dropout3d)):
                        module.train()
            outputs = [self(image) for _ in range(mc_samples)]
            p = torch.stack([o["probabilities"] for o in outputs]).mean(0)
            extra = {k: torch.stack([o["extra"][k] for o in outputs]).mean(0)
                     for k in outputs[0]["extra"] if k not in ("entropy", "top_probability", "margin")}
            extra["mutual_information"] = (uncertainty(p)["entropy"] -
                                            torch.stack([o["uncertainty"] for o in outputs]).mean(0)).clamp_min(0)
            return pack_output(p.clamp_min(1e-8).log(), torch.stack([o["features"] for o in outputs]).mean(0), extra, p)
        finally:
            for module, state in states.items():
                module.training = state

    @torch.no_grad()
    def analyze(self, image, *, pool=True, retain_spatial=True):
        """Observational analysis interface. Segmentation computation is unchanged."""
        from ..analysis.extract import analyze_segmentation_expert
        return analyze_segmentation_expert(self, image, pool=pool, retain_spatial=retain_spatial)


def group_norm(channels):
    """GroupNorm valid at batch size 1, including 2-voxel bottlenecks.

    Prefer 8 groups, but keep at least two values per group so training
    GroupNorm does not reject 1x1x1 or very small spatial maps.
    """
    for groups in (8, 4, 2, 1):
        if channels % groups == 0 and (groups == 1 or channels // groups >= 2):
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=.1, dilation=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            group_norm(out_channels), nn.GELU(), nn.Dropout3d(dropout),
            nn.Conv3d(out_channels, out_channels, 3, padding=dilation, dilation=dilation),
            group_norm(out_channels), nn.GELU())

    def forward(self, x):
        return self.layers(x)


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with GroupNorm and a learned residual skip."""

    def __init__(self, in_channels, out_channels, dropout=.1, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            group_norm(out_channels), nn.GELU(), nn.Dropout3d(dropout),
            nn.Conv3d(out_channels, out_channels, 3, padding=dilation, dilation=dilation),
            group_norm(out_channels))
        self.skip = nn.Identity() if in_channels == out_channels else nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1), group_norm(out_channels))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.block(x) + self.skip(x))


def maybe_pool(x):
    # Pool only when every axis can keep >=2 voxels. A 1x1x1 bottleneck
    # breaks training GroupNorm at batch size 1 and is not useful context.
    return F.avg_pool3d(x, 2) if min(x.shape[2:]) >= 4 else x


class ResidualUNet(nn.Module):
    """nnU-Net-like 3D U-Net: residual blocks, skips, GroupNorm, depth downsamples."""

    def __init__(self, in_channels, width, dropout=.1, depth=3):
        super().__init__()
        if depth < 2:
            raise ValueError("ResidualUNet depth must be >=2")
        self.depth = depth
        self.out_channels = width
        self.stem = ResidualBlock(in_channels, width, dropout)
        downs, chans = nn.ModuleList(), [width]
        src = width
        for _ in range(depth):
            dst = src * 2
            downs.append(ResidualBlock(src, dst, dropout))
            chans.append(dst)
            src = dst
        self.downs = downs
        ups = nn.ModuleList()
        for skip in reversed(chans[:-1]):
            ups.append(ResidualBlock(src + skip, skip, dropout))
            src = skip
        self.ups = ups

    def forward(self, x):
        skips = [self.stem(x)]
        h = skips[0]
        for block in self.downs:
            h = block(maybe_pool(h))
            skips.append(h)
        h = skips.pop()
        for block in self.ups:
            skip = skips.pop()
            h = F.interpolate(h, size=skip.shape[2:], mode="trilinear", align_corners=False)
            h = block(torch.cat([skip, h], 1))
        return h


class ChannelSummary(nn.Module):
    """Deterministic compression of trained decoder activations; no unsupervised random head."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if out_channels > in_channels or out_channels < 1:
            raise ValueError("feature_channels must be between 1 and the decoder width")
        self.out_channels = out_channels

    def forward(self, x):
        return torch.cat([chunk.mean(1, keepdim=True) for chunk in torch.tensor_split(x, self.out_channels, dim=1)], 1)
