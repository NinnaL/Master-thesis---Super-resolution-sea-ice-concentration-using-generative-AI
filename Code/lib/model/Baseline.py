import torch.nn as nn
import torch.nn.functional as F


# SRCNN architecture adapted
class EncDec(nn.Module):
    def __init__(self, in_channels, features=64):
        super(EncDec, self).__init__()

        self.features = nn.Sequential(
            # feature extraction without downsampling
            nn.Conv2d(in_channels, features, 9, padding=4),
            nn.ReLU(),
            nn.Conv2d(features, features, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features, features, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features, features, 3, padding=1),
            nn.ReLU(),
        )

        self.upsample = nn.Sequential(
            # upsampling
            nn.Conv2d(features, features*25, 3, padding=1), # 5x
            nn.PixelShuffle(5),                             # (B, features*25, H, W) -> (B, features, 5H, 5W)
            nn.ReLU(),
            nn.Conv2d(features, features*25, 3, padding=1), # 5x
            nn.PixelShuffle(5),                             # (B, features*25, 5H, 5W) -> (B, features, 25H, 25W)
            nn.ReLU(),
            nn.Conv2d(features, 1, 3, padding=1)
        )

    def forward(self, x):

        out = self.upsample(self.features(x))    # (B, 1, H*25, W*25)

        return out
