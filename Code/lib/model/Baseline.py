import torch.nn as nn
import torch.nn.functional as F

class EncDec(nn.Module):
    def __init__(self, return_features=False):
        super(EncDec, self).__init__()
        self.return_features = return_features
        self.encoder = nn.Sequential(
            # encoder (downsampling)
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128 -> 64
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 -> 32
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 -> 16
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 16 -> 8
        )
    
        # bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            # decoder (upsampling)
            nn.Upsample(16),  # 8 -> 16
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(32),  # 16 -> 32
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(64),  # 32 -> 64
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(128),  # 64 -> 128
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x, return_features=False):
        x_enc = self.encoder(x)
        x_bottleneck = self.bottleneck_conv(x_enc)
        x_dec = self.decoder(x_bottleneck)

        if return_features:
            return x_bottleneck
            
        return x_dec