"""Analysis-only pooling. Never used inside expert forward or fusion."""
import torch
import torch.nn.functional as F


def pool_patch(spatial):
    """Deterministic global mean over spatial axes: [B, C, D, H, W] -> [B, C]."""
    if spatial.ndim != 5:
        raise ValueError("Expected a 5D spatial feature map [B, C, D, H, W]")
    return spatial.float().mean(dim=(2, 3, 4))


def pool_region(spatial, mask):
    """Mean features inside a boolean mask, aligned to the feature grid if needed."""
    if spatial.ndim != 5:
        raise ValueError("Expected a 5D spatial feature map [B, C, D, H, W]")
    mask = torch.as_tensor(mask, device=spatial.device, dtype=spatial.dtype)
    if mask.ndim == 3:
        mask = mask[None, None]
    elif mask.ndim == 4:
        mask = mask[:, None]
    if mask.shape[-3:] != spatial.shape[-3:]:
        mask = F.interpolate(mask, size=spatial.shape[-3:], mode="nearest")
    weights = mask.clamp_min(0)
    denom = weights.sum(dim=(2, 3, 4)).clamp_min(1e-8)
    return (spatial.float() * weights).sum(dim=(2, 3, 4)) / denom


def pool_patient(patch_vectors):
    """Mean of patch (or region) vectors for one patient: [N, C] or list -> [C]."""
    stacked = torch.stack([torch.as_tensor(v).reshape(-1) for v in patch_vectors], 0).float()
    return stacked.mean(0)


def upsample_spatial(spatial, size):
    """Optional analysis interpolation to a volume grid. Not a model output."""
    return F.interpolate(spatial.float(), size=tuple(size), mode="trilinear", align_corners=False)
