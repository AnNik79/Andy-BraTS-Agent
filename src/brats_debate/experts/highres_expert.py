import torch
from torch import nn
from .base import Expert, ResidualBlock, ChannelSummary


class HighResolutionExpert(Expert):
    """Fine-structure specialist: HighResNet-style residual dilated streams, no pooling.

    A local (dilation 1) path is fused with a wider-context dilated path so the
    expert keeps native-resolution maps while still seeing beyond a 3x3 neighborhood.
    """

    def __init__(self, in_channels, classes, width=16, feature_channels=4, dropout=.1, depth=3):
        super().__init__()
        del depth  # shared factory kwargs; highres does not downsample.
        self.stem = ResidualBlock(in_channels, width, dropout, dilation=1)
        self.local = nn.Sequential(
            ResidualBlock(width, width, dropout, dilation=1),
            ResidualBlock(width, width, dropout, dilation=1),
        )
        self.context = nn.Sequential(
            ResidualBlock(width, width, dropout, dilation=2),
            ResidualBlock(width, width, dropout, dilation=4),
            ResidualBlock(width, width, dropout, dilation=2),
        )
        self.fuse = ResidualBlock(width * 2, width, dropout, dilation=1)
        self.head = nn.Conv3d(width, classes, 1)
        self.compact = ChannelSummary(width, feature_channels)

    def raw_forward(self, image):
        stem = self.stem(image)
        features = self.fuse(torch.cat([self.local(stem), self.context(stem)], 1))
        return self.head(features), self.compact(features), {}
