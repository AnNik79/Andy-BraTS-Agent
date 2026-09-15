import torch
from ..config import SEGMENTATION_EXPERTS


def controller_features(image, outputs, debate):
    values = [image.float()]
    for name in SEGMENTATION_EXPERTS:
        out = outputs[name]
        values.extend([out["probabilities"], out["uncertainty"], out["confidence"], out["extra"]["margin"],
                       out["features"], out["extra"].get("availability", torch.ones_like(out["confidence"]))])
    values.extend([debate["map"], debate["js_divergence"], debate["variance"],
                   debate["prediction_entropy"], debate["votes"] / len(SEGMENTATION_EXPERTS),
                   outputs["boundary"]["extra"]["boundary_probability"],
                   outputs["boundary"]["extra"]["distance_to_boundary"] / 16.])
    return torch.cat(values, 1)


def feature_channels(cfg):
    classes, compact = len(cfg["label_mapping"]), cfg["model"]["feature_channels"]
    n = len(SEGMENTATION_EXPERTS)
    return len(cfg["modalities"]) + n * (classes + compact + 4) + 5 + 2 * classes


def stack_evidence(outputs):
    p = torch.stack([outputs[n]["probabilities"] for n in SEGMENTATION_EXPERTS], 1)
    availability = torch.cat([outputs[n]["extra"].get("availability", torch.ones_like(outputs[n]["confidence"]))
                              for n in SEGMENTATION_EXPERTS], 1)
    return p, availability
