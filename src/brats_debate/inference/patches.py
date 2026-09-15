from itertools import product
from functools import lru_cache
import torch
import torch.nn.functional as F
from ..experts.base import pack_output


def window_starts(shape, patch, overlap=.5):
    axes = []
    for total, size in zip(shape, patch):
        last = max(0, total - size)
        starts = list(range(0, last + 1, max(1, int(size * (1 - overlap)))))
        if starts[-1] != last:
            starts.append(last)
        axes.append(starts)
    return list(product(*axes))


def patch_slices(start, patch):
    return tuple(slice(a, a + n) for a, n in zip(start, patch))


@torch.no_grad()
def sliding_predict(model, image, patch, device, overlap=.5, mc_samples=1,
                    proposal=None, max_patches=None):
    """CPU accumulation. Specialist abstention is explicit outside evaluated patches."""
    if image.ndim != 5 or image.shape[0] != 1:
        raise ValueError("Sliding inference requires one [1,C,X,Y,Z] patient")
    shape = tuple(image.shape[2:])
    pad = [max(0, n - s) for s, n in zip(shape, patch)]
    image = F.pad(image.detach().cpu(), (0, pad[2], 0, pad[1], 0, pad[0]))
    spatial = image.shape[2:]
    starts = window_starts(spatial, patch, overlap)
    if proposal is not None:
        proposal = F.pad(proposal.float().cpu(), (0, pad[2], 0, pad[1], 0, pad[0]))
        scored = [(float(proposal[(..., *patch_slices(s, patch))].mean()), s) for s in starts]
        starts = [s for score, s in sorted(scored, reverse=True) if score > 0]
        if not starts:
            # A single background check gives an independent negative example.
            starts = [tuple(max(0, (s - p) // 2) for s, p in zip(spatial, patch))]
        if max_patches is not None:
            starts = starts[:max(1, int(max_patches))]
    accum, extra_accum, counts = None, {}, torch.zeros((1, 1, *spatial))
    features = None
    for start in starts:
        sl = patch_slices(start, patch)
        out = model.predict(image[(..., *sl)].to(device), mc_samples)
        if accum is None:
            accum = torch.zeros((1, out["probabilities"].shape[1], *spatial))
            features = torch.zeros((1, out["features"].shape[1], *spatial))
        accum[(..., *sl)] += out["probabilities"].detach().cpu()
        features[(..., *sl)] += out["features"].detach().cpu()
        counts[(..., *sl)] += 1
        for key, value in out["extra"].items():
            if key in ("entropy", "top_probability", "margin", "boundary_logits"):
                continue
            if key not in extra_accum:
                extra_accum[key] = torch.zeros((1, value.shape[1], *spatial))
            extra_accum[key][(..., *sl)] += value.detach().cpu()
        del out
    covered = counts > 0
    probabilities = accum / counts.clamp_min(1)
    # Uniform probabilities denote missing evidence; availability masks exclude them from fusion.
    probabilities = torch.where(covered, probabilities, torch.full_like(probabilities, 1 / probabilities.shape[1]))
    trim = tuple(slice(0, s) for s in shape)
    p = probabilities[(..., *trim)]
    extras = {key: (value / counts.clamp_min(1))[(..., *trim)] for key, value in extra_accum.items()}
    extras["availability"] = covered[(..., *trim)].float()
    features = (features / counts.clamp_min(1))[(..., *trim)]
    return pack_output(p.clamp_min(1e-8).log(), features, extras, p)


@torch.inference_mode()
def sliding_probabilities(model, image, patch, device, overlap=.5, event=None,
                          batch_size=1, accumulation_device="cpu", cached_geometry=False):
    """Validation-only path: one contiguous probability patch copied to CPU at a time.

    No full-volume GPU allocation, auxiliary features, entropy, MC stacks, or GPU metric lists.
    """
    if image.device.type != "cpu" or image.ndim != 5 or image.shape[0] != 1:
        raise ValueError("Validation requires one CPU [1,C,X,Y,Z] patient")
    event = event or (lambda *args, **kwargs: None)
    if batch_size != 1 or accumulation_device != "cpu" or cached_geometry:
        return _batched_probabilities(model, image, tuple(patch), device, overlap, event,
                                      batch_size, accumulation_device)
    shape = image.shape[2:]
    padding = [max(0, p - s) for p, s in zip(patch, shape)]
    image = F.pad(image.detach(), (0, padding[2], 0, padding[1], 0, padding[0]))
    starts = window_starts(image.shape[2:], patch, overlap)
    counts = torch.zeros((1, 1, *image.shape[2:]), device="cpu")
    accumulated = None
    for index, start in enumerate(starts, 1):
        sl = patch_slices(start, patch)
        event("validation patch forward begin", patch=index, patches=len(starts))
        # Materialize strided CPU window before transfer; GPU input/output layout is explicit.
        gpu_input = image[(..., *sl)].contiguous().to(device)
        logits = model.segmentation_logits(gpu_input)
        probabilities = logits.float().softmax(1)
        del logits, gpu_input
        event("validation patch forward complete", patch=index, patches=len(starts))
        event("validation probability CPU transfer begin", patch=index, patches=len(starts),
              bytes=probabilities.numel() * probabilities.element_size())
        cpu_probabilities = probabilities.detach().contiguous().cpu()
        event("validation probability CPU transfer complete", patch=index, patches=len(starts))
        del probabilities
        if accumulated is None:
            accumulated = torch.zeros((1, cpu_probabilities.shape[1], *image.shape[2:]), device="cpu")
        accumulated[(..., *sl)] += cpu_probabilities
        counts[(..., *sl)] += 1
        del cpu_probabilities
        if index % 10 == 0 or index == len(starts):
            event("validation patch progress", patch=index, patches=len(starts))
    accumulated.div_(counts)
    trim = tuple(slice(0, s) for s in shape)
    return accumulated[(..., *trim)]


@lru_cache(maxsize=4)
def _window_geometry(shape, patch, overlap):
    """Small bounded CPU cache; tensors are read-only and never part of a graph."""
    starts = tuple(window_starts(shape, patch, overlap))
    counts = torch.zeros((1, 1, *shape), dtype=torch.float32)
    for start in starts:
        counts[(..., *patch_slices(start, patch))] += 1
    return starts, counts


@torch.inference_mode()
def _batched_probabilities(model, image, patch, device, overlap, event, batch_size, accumulation_device):
    if batch_size < 1 or accumulation_device not in ("cpu", "model"):
        raise ValueError("Positive window batch size and cpu/model accumulation required")
    shape = tuple(image.shape[2:])
    padding = [max(0, p - s) for p, s in zip(patch, shape)]
    image = image.detach()
    if any(padding):
        image = F.pad(image, (0, padding[2], 0, padding[1], 0, padding[0]))
    starts, cpu_counts = _window_geometry(tuple(image.shape[2:]), patch, overlap)
    destination = device if accumulation_device == "model" else "cpu"
    accumulated = None
    for offset in range(0, len(starts), batch_size):
        windows = starts[offset:offset + batch_size]
        index = offset + len(windows)
        event("validation patch forward begin", patch=index, patches=len(starts))
        batch = torch.cat([image[(..., *patch_slices(s, patch))] for s in windows], 0).to(device)
        p = model.segmentation_logits(batch).float().softmax(1).detach()
        event("validation patch forward complete", patch=index, patches=len(starts))
        if accumulation_device == "cpu":
            event("validation probability CPU transfer begin", patch=index, patches=len(starts),
                  bytes=p.numel() * p.element_size())
            p = p.contiguous().cpu()
            event("validation probability CPU transfer complete", patch=index, patches=len(starts))
        if accumulated is None:
            accumulated = torch.zeros((1, p.shape[1], *image.shape[2:]), device=destination)
        # Preserve window summation order, including the last partial batch.
        for j, start in enumerate(windows):
            accumulated[(..., *patch_slices(start, patch))] += p[j:j+1]
        del batch, p
        if index // 10 != offset // 10 or index == len(starts):
            event("validation patch progress", patch=index, patches=len(starts))
    accumulated.div_(cpu_counts.to(destination))
    if accumulation_device == "model":
        event("validation probability CPU transfer begin", patch=len(starts), patches=len(starts),
              bytes=accumulated.numel() * accumulated.element_size())
        accumulated = accumulated.detach().cpu()
        event("validation probability CPU transfer complete", patch=len(starts), patches=len(starts))
    return accumulated[(..., *(slice(0, s) for s in shape))]
