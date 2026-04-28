import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features, features, kernel_size=3, padding=1))
    
    def forward(self, x):
        return x + self.block(x)  # Residual connection

class ASPP(nn.Module):
    def __init__(self, features, rates=(6, 12, 18)):
        super().__init__()

        # For local context - 1x1 conv
        self.first_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=1),
            nn.ReLU())
        
        # For multi-scale context - dilated convolutions at rates
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, padding=rate, dilation=rate),
                nn.ReLU()) for rate in rates])
        
        # Fuse all branches: 1x1 + 3 dilated convs = 4 x features -> features
        self.fuse = nn.Sequential(
            nn.Conv2d(features * 4, features, kernel_size=1),
            nn.ReLU())
        
    def forward(self, x):
        out = [self.first_conv(x)] + [conv(x) for conv in self.dilated_convs]
        return self.fuse(torch.cat(out, dim=1))

class FusionNetASPP(nn.Module):
    name = 'FusionNetASPP'
    def __init__(self, in_channels, features=64):
        super().__init__()

        self.first_conv = nn.Sequential(
                            nn.Conv2d(2, features, kernel_size=9, padding=4), 
                            nn.ReLU())
        
        self.conv       = nn.Sequential(
                            nn.Conv2d(features+2, features, kernel_size=3, padding=1),
                            nn.ReLU())
        
        self.res_blocks = ResBlock(features)

        self.aspp       = ASPP(features)

        self.reduces    = nn.ModuleList([
                            nn.Sequential(
                            nn.Conv2d(features*2, features, kernel_size=3, padding=1),
                            nn.ReLU()) for _ in range(6)])

        self.final_conv = nn.Sequential(
                            nn.Conv2d(features, 1, kernel_size=3, padding=1))

    def forward(self, x, target_size):
        # Encoder
        s1 = self.first_conv(x[:, -2:])     # 89 GHz HH and HV
        s1 = self.res_blocks(s1)            # Add residual blocks after first conv

        s2 = self.conv(torch.cat([s1, x[:, -4:-2]], dim=1))    # 36.5 GHz HH and HV
        s2 = self.res_blocks(s2)            # Add residual blocks
        
        s3 = self.conv(torch.cat([s2, x[:, -6:-4]], dim=1))    # 23.8 GHz HH and HV
        s3 = self.res_blocks(s3)            # Add residual blocks
        
        s4 = self.conv(torch.cat([s3, x[:, -8:-6]], dim=1))    # 18.7 GHz HH and HV
        s4 = self.res_blocks(s4)            # Add residual blocks
        
        s5 = self.conv(torch.cat([s4, x[:, -10:-8]], dim=1))   # 10.7 GHz HH and HV
        s5 = self.res_blocks(s5)            # Add residual blocks
        
        s6 = self.conv(torch.cat([s5, x[:, -12:-10]], dim=1))  # 7.3 GHz HH and HV
        s6 = self.res_blocks(s6)            # Add residual blocks
        
        s7 = self.conv(torch.cat([s6, x[:, -14:-12]], dim=1))  # 6.9 GHz HH and HV
        s7 = self.res_blocks(s7)            # Add residual blocks

        # ASPP bottleneck
        d = self.aspp(s7)

        # Decoder
        skips = [s6, s5, s4, s3, s2, s1]  # Collect skip connections

        for i, skip in enumerate(skips):
            d = self.res_blocks(d)
            d = self.reduces[i](torch.cat([d, skip], dim=1))  # Concatenate skip connection
        
        # Final residual refinement and convolution
        out = self.res_blocks(d)
        out = self.final_conv(out)

        # Snap to exact target size — corrects minor size differences
        out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
        
        return out