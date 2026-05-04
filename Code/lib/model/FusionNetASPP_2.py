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

class FusionNetASPP_2(nn.Module):
    """
    FusionNetASPP with two stride-2 downsampling steps in the encoder,
    matched by two transposed convolution upsampling steps in the decoder.

    Downsampling is placed at s4 (18.7 GHz) and s6 (7.3 GHz) —
    matching the physical resolution drop between AMSR2 frequency groups:
      s1-s3: full res  (H, W)       — 89, 36.5, 23.8 GHz fine-scale
      s4-s5: half res  (H/2, W/2)  — 18.7, 10.7 GHz mid-scale
      s6-s7: quarter   (H/4, W/4)  — 7.3, 6.9 GHz coarse-scale + ASPP

    Decoder unwinds with transposed convolutions:
      (H/4, W/4) → upsample → (H/2, W/2) → upsample → (H, W)
    Skip connections from encoder stages are concatenated at matching
    spatial resolutions before each reduce step.
    """
    name = 'FusionNetASPP_2'
    def __init__(self, in_channels=14, features=64):
        super().__init__()
        ### Encoder ###
        self.first_conv = nn.Sequential(
                            nn.Conv2d(2, features, kernel_size=9, padding=4), 
                            nn.ReLU())
        
        self.conv       = nn.Sequential(
                            nn.Conv2d(features+2, features, kernel_size=3, padding=1),
                            nn.ReLU())
        
        self.conv_stride= nn.Sequential(     
                            nn.Conv2d(features+2, features, kernel_size=3, stride=2, padding=1),
                            nn.ReLU()) # Downsample by 2
        
        self.res_blocks = ResBlock(features)

        ### ASPP ###
        self.aspp       = ASPP(features)

        ### Decoder ###
        self.up         = nn.ConvTranspose2d(features, features, kernel_size=3, stride=2, padding=1, output_padding=1)  # Upsample by 2

        self.reduces    = nn.Sequential(
                            nn.Conv2d(features*2, features, kernel_size=3, padding=1),
                            nn.ReLU())

        self.final_conv = nn.Conv2d(features, 1, kernel_size=3, padding=1)

    def forward(self, x, target_size):
        # ── Encoder — full resolution (H, W) ─────────────────────────────────
        s1 = self.first_conv(x[:, -2:])     # 89 GHz HH and HV
        s1 = self.res_blocks(s1)            # Add residual blocks after first conv

        s2 = self.conv(torch.cat([s1, x[:, -4:-2]], dim=1))    # 36.5 GHz HH and HV
        s2 = self.res_blocks(s2)            # Add residual blocks
        
        s3 = self.conv(torch.cat([s2, x[:, -6:-4]], dim=1))    # 23.8 GHz HH and HV
        s3 = self.res_blocks(s3)            # Add residual blocks
        
        # ── Encoder — stride-2 → (H/2, W/2) ─────────────────────────────────        
        s4 = self.conv_stride(torch.cat([s3, x[:, -8:-6]], dim=1))    # 18.7 GHz HH and HV
        s4 = self.res_blocks(s4)            # Add residual blocks
        
        x5 = F.interpolate(x[:, -10:-8], size=s4.shape[2:], mode='bilinear', align_corners=False)  # Downsample input for 10.7 GHz
        s5 = self.conv1(torch.cat([s4, x5], dim=1))   # 10.7 GHz HH and HV
        s5 = self.res_blocks(s5)            # Add residual blocks
        
        # ── Encoder — stride-2 → (H/4, W/4) ─────────────────────────────────
        x6 = F.interpolate(x[:, -12:-10], size=s5.shape[2:], mode='bilinear', align_corners=False)  # Downsample input for 7.3 GHz
        s6 = self.conv_stride(torch.cat([s5, x6], dim=1))  # 7.3 GHz HH and HV
        s6 = self.res_blocks(s6)            # Add residual blocks
        
        x7 = F.interpolate(x[:, -14:-12], size=s6.shape[2:], mode='bilinear', align_corners=False)  # Downsample input for 6.9 GHz
        s7 = self.conv(torch.cat([s6, x7], dim=1))  # 6.9 GHz HH and HV
        s7 = self.res_blocks(s7)            # Add residual blocks

        # ── ASPP bottleneck at (H/4, W/4) ────────────────────────────────────
        d = self.aspp(s7)

        # ── Decoder — (H/4, W/4) → (H/2, W/2) ───────────────────────────────
        d = self.res_blocks(d)
        d = self.up(d)
        d = F.interpolate(d, size=s6.shape[2:], mode='nearest')                 # snap odd dims
        d = self.reduces(torch.cat([d, s6], dim=1))                             # fuse s6 skip

        d = self.res_blocks(d)
        d = F.interpolate(d, size=s5.shape[2:], mode='nearest')
        d = self.reduces(torch.cat([d, s5], dim=1))                             # fuse s5 skip

        # ── Decoder — (H/2, W/2) → (H, W) ───────────────────────────────────
        d = self.res_blocks(d)
        d = self.up(d)
        d = F.interpolate(d, size=s4.shape[2:], mode='nearest')                 # snap odd dims
        d = self.reduces(torch.cat([d, s4], dim=1))                             # fuse s4 skip

        d = self.res_blocks(d)
        d = F.interpolate(d, size=s3.shape[2:], mode='nearest')
        d = self.reduces(torch.cat([d, s3], dim=1))                             # fuse s3 skip

        d = self.res_blocks(d)
        d = F.interpolate(d, size=s2.shape[2:], mode='nearest')
        d = self.reduces(torch.cat([d, s2], dim=1))                             # fuse s2 skip

        d = self.res_blocks(d)
        d = F.interpolate(d, size=s1.shape[2:], mode='nearest')
        d = self.reduces(torch.cat([d, s1], dim=1))                             # fuse s1 skip

        # ── Output ────────────────────────────────────────────────────────────
        d   = self.res_blocks(d)
        out = self.final_conv(d)
        out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
        return out