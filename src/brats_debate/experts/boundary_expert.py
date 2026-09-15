from torch import nn
from .base import Expert, ResidualUNet, ResidualBlock, ChannelSummary


class BoundaryExpert(Expert):
    """Professor's Sum Medical role: boundary / medical-geometry specialist.

    Internal checkpoint key remains "boundary". Independent residual U-Net plus a
    dilated geometry stream. Boundary/distance heads supervise that stream, and
    the predicted interface map modulates the features used for class logits so
    geometry actually changes the segmentation representation.
    """

    def __init__(self, in_channels, classes, width=16, feature_channels=4, dropout=.1, depth=3):
        super().__init__()
        self.backbone = ResidualUNet(in_channels, width, dropout, depth)
        self.geometry = nn.Sequential(
            ResidualBlock(width, width, dropout, dilation=2),
            ResidualBlock(width, width, dropout, dilation=1),
        )
        self.fuse = nn.Conv3d(width, width, 1)
        self.boundary_head = nn.Conv3d(width, 1, 1)
        self.distance_head = nn.Conv3d(width, 1, 1)
        self.head = nn.Conv3d(width, classes, 1)
        self.compact = ChannelSummary(width, feature_channels)

    def encode_decode(self, x):
        return self.backbone(x)

    def raw_forward(self, image):
        base = self.encode_decode(image)
        geom = self.geometry(base)
        boundary_logits = self.boundary_head(geom)
        boundary = boundary_logits.sigmoid()
        extra = {"boundary_logits": boundary_logits, "boundary_probability": boundary,
                 "boundary_confidence": (2 * boundary - 1).abs(),
                 "distance_to_boundary": self.distance_head(geom).sigmoid() * 16.}
        features = base + self.fuse(geom) * boundary
        return self.head(features), self.compact(features), extra
