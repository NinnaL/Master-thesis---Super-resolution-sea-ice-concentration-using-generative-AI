import torch
import torch.nn as nn
import torch.nn.functional as F


class EncDec(nn.Module):
    def __init__(self, in_channels, features=64):
        super().__init__()
 
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features, 1, kernel_size=3, padding=1),
        )

    def forward(self, x, target_size):
        # Normalise AMSR2 brightness temperatures to ~[-1, 2]
        x = (x - 150.0) / 50.0
 
        out = self.features(x)
 
        # Snap to exact target size — corrects minor size differences
        # from batch collation without introducing meaningful upsampling
        out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
        return out