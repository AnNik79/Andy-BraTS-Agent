from ..config import REASONING_EXPERT, SEGMENTATION_EXPERTS
from .cnn_expert import CNNExpert
from .transformer_expert import TransformerExpert, SwinExpert
from .boundary_expert import BoundaryExpert
from .highres_expert import HighResolutionExpert


def build_expert(name, cfg):
    if name == REASONING_EXPERT:
        raise ValueError("The LLM is the reasoning expert, not a voxel network. Use LLMReasoningExpert.from_config(cfg).")
    if name not in SEGMENTATION_EXPERTS:
        raise ValueError(f"Unknown segmentation expert: {name}")
    model = cfg["model"]
    kwargs = dict(in_channels=len(cfg["modalities"]), classes=len(cfg["label_mapping"]),
                  width=model["width"], feature_channels=model["feature_channels"], dropout=model["dropout"],
                  depth=model.get("unet_depth", 3))
    if name == "transformer":
        backend = model["transformer_backend"]
        if backend == "swin":
            return SwinExpert(kwargs["in_channels"], kwargs["classes"], kwargs["feature_channels"],
                              model["swin_feature_size"], cfg["crop_size"], model["dropout"])
        if backend not in ("hierarchical", "tiny"):
            raise ValueError("Unknown transformer backend")
        return TransformerExpert(in_channels=kwargs["in_channels"], classes=kwargs["classes"],
                                 width=kwargs["width"], feature_channels=kwargs["feature_channels"],
                                 dropout=kwargs["dropout"], dim=model["transformer_dim"],
                                 heads=model["transformer_heads"], layers=model["transformer_layers"],
                                 grid=model["transformer_grid"], window=model.get("transformer_window", 4))
    return {"cnn": CNNExpert, "boundary": BoundaryExpert, "highres": HighResolutionExpert}[name](**kwargs)
