"""Observational representation extraction for the four voxel experts.

Hooks capture an existing submodule tensor during the unchanged forward pass.
No architecture, head, or loss is modified. Standardization of shapes is not
performed here; analysis pooling lives in pooling.py.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import torch
from ..experts import build_expert
from ..experts.boundary_expert import BoundaryExpert
from ..experts.cnn_expert import CNNExpert
from ..experts.highres_expert import HighResolutionExpert
from ..experts.transformer_expert import SwinExpert, TransformerExpert


REPO = Path(__file__).resolve().parents[3]
FROZEN_CNN_BEST = REPO / "outputs/brats2023_gli_cnn_curve4/checkpoints/cnn/best.pt"
FROZEN_CNN_EPOCH = 15
FROZEN_CNN_MEAN_DICE = 0.8621752521506791


@dataclass(frozen=True)
class RepresentationSpec:
    name: str
    module: torch.nn.Module
    capture: str
    description: str


def representation_spec(model):
    """Specialty representation: one late tensor that reflects the expert's role."""
    if isinstance(model, CNNExpert):
        return RepresentationSpec(
            name="encoder_bottleneck",
            module=model.backbone.downs[-1],
            capture="output",
            description="Deepest ResidualUNet encoder block (bottleneck); local/multi-scale structure.",
        )
    if isinstance(model, TransformerExpert):
        return RepresentationSpec(
            name="global_bottleneck_attention",
            module=model.bot_attn[-1],
            capture="output",
            description="Last global bottleneck attention map; long-range / hierarchical context.",
        )
    if isinstance(model, SwinExpert):
        return RepresentationSpec(
            name="swin_decoder_feature",
            module=model.network.decoder1,
            capture="output",
            description="SwinUNETR decoder1 feature; global transformer context before the head.",
        )
    if isinstance(model, BoundaryExpert):
        return RepresentationSpec(
            name="geometry_modulated_prehead",
            module=model.head,
            capture="input",
            description="Geometry-modulated decoder features immediately before the class head.",
        )
    if isinstance(model, HighResolutionExpert):
        return RepresentationSpec(
            name="native_resolution_prehead",
            module=model.fuse,
            capture="output",
            description="Fused local+dilated native-resolution features immediately before the class head.",
        )
    raise TypeError(f"Unsupported segmentation expert: {type(model)}")


@contextmanager
def _capture_tensor(module, capture):
    box = {}

    def hook(_module, inputs, output):
        tensor = inputs[0] if capture == "input" else output
        box["tensor"] = tensor.detach()

    handle = module.register_forward_hook(hook)
    try:
        yield box
    finally:
        handle.remove()


@torch.no_grad()
def analyze_segmentation_expert(model, image, *, pool=True, retain_spatial=True):
    """Run the expert's existing forward while capturing its specialty representation."""
    spec = representation_spec(model)
    was_training = model.training
    model.eval()
    try:
        with _capture_tensor(spec.module, spec.capture) as box:
            output = model(image)
        spatial = box["tensor"]
        result = {
            "logits": output["logits"],
            "probabilities": output["probabilities"],
            "segmentation": output["segmentation"],
            "confidence": output["confidence"],
            "uncertainty": output["uncertainty"],
            "compact_decoder_features": output["features"],
            "extra": output["extra"],
            "representation_name": spec.name,
            "representation_description": spec.description,
            "spatial_representation": spatial if retain_spatial else None,
            "spatial_shape": tuple(spatial.shape),
        }
        if pool:
            from .pooling import pool_patch
            result["pooled_patch"] = pool_patch(spatial)
            result["pooled_dim"] = int(result["pooled_patch"].shape[-1])
        return result
    finally:
        model.train(was_training)


def load_frozen_cnn(cfg, checkpoint=FROZEN_CNN_BEST, map_location="cpu"):
    """Load the finalized epoch-15 CNN. Does not train or alter the file."""
    path = Path(checkpoint)
    data = torch.load(path, map_location=map_location, weights_only=True)
    if data.get("name") != "cnn":
        raise ValueError(f"Expected CNN checkpoint, found {data.get('name')}")
    model = build_expert("cnn", cfg)
    model.load_state_dict(data["state_dict"])
    model.eval().requires_grad_(False)
    return model, data
