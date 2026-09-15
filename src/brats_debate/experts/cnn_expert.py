from .base import Expert, ResidualUNet, ChannelSummary
from torch import nn


class CNNExpert(Expert):
    """Local-structure / voxel-level 3D residual U-Net (nnU-Net-inspired, not nnU-Net)."""

    def __init__(self, in_channels, classes, width=16, feature_channels=4, dropout=.1, depth=3):
        super().__init__()
        self.backbone = ResidualUNet(in_channels, width, dropout, depth)
        self.head = nn.Conv3d(width, classes, 1)
        self.compact = ChannelSummary(width, feature_channels)

    def encode_decode(self, x):
        return self.backbone(x)

    def raw_forward(self, image):
        features = self.encode_decode(image)
        return self.head(features), self.compact(features), {}

    def segmentation_logits(self, image):
        return self.head(self.encode_decode(image))
