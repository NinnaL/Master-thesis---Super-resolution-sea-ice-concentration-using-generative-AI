"""
diffusion_model.py
------------------
Self-contained module defining all model classes, SDE helpers, data helpers,
sampler, and a load_models() convenience function.

Import this from the notebook:
    from diffusion_model import load_models, Euler_Maruyama_sampler, prepare_cond, prepare_sic, valid_mask, EMA

Author: Ninna Juul Ligaard, MSc thesis, DTU/DMI 2026
"""

import os
import sys
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants ─────────────────────────────────────────────────────────────────
SIC_SENTINEL_MIN = 254   # 254=missing, 255=land
SIGMA            = 25.0  # must match training script


# ══════════════════════════════════════════════════════════════════════════════
# Building blocks
# ══════════════════════════════════════════════════════════════════════════════

class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class RMSNorm(nn.Module):
    """RMS layer norm over the channel dimension."""
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        rms = x.pow(2).mean(dim=1, keepdim=True).add(1e-8).sqrt()
        return x / rms * self.scale


class SpatialSelfAttention(nn.Module):
    """Multi-head self-attention at the U-Net bottleneck."""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm = RMSNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens     = self.norm(x).flatten(2).transpose(1, 2)
        out, _     = self.attn(tokens, tokens, tokens)
        return x + out.transpose(1, 2).reshape(B, C, H, W)

class EMA:
    """
    Exponential moving average of model weights.
    EMA shadow weights produce sharper, more stable samples at inference.
    """
    def __init__(self, model, decay=0.999):
        self.decay  = decay
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1.0 - self.decay) * v.float()

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.float() for k, v in sd.items()}

# ══════════════════════════════════════════════════════════════════════════════
# ScoreNet
# ══════════════════════════════════════════════════════════════════════════════

class ScoreNet(nn.Module):
    """
    Conditional score network.
    Conditioning via input concatenation: [noisy_sic, condition] → 2 channels.
    Bottleneck self-attention + RMSNorm throughout.
    """
    def __init__(self, marginal_prob_std, channels=(32, 64, 128, 256), embed_dim=256):
        super().__init__()
        self.act   = nn.SiLU()
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        self.marginal_prob_std = marginal_prob_std
        C = channels

        self.conv1  = nn.Conv2d(2,    C[0], 3, padding=1,           bias=False)
        self.conv2  = nn.Conv2d(C[0], C[1], 3, stride=2, padding=1, bias=False)
        self.conv3  = nn.Conv2d(C[1], C[2], 3, stride=2, padding=1, bias=False)
        self.conv4  = nn.Conv2d(C[2], C[3], 3, stride=2, padding=1, bias=False)

        self.dense1 = Dense(embed_dim, C[0])
        self.dense2 = Dense(embed_dim, C[1])
        self.dense3 = Dense(embed_dim, C[2])
        self.dense4 = Dense(embed_dim, C[3])

        self.rnorm1 = RMSNorm(C[0])
        self.rnorm2 = RMSNorm(C[1])
        self.rnorm3 = RMSNorm(C[2])
        self.rnorm4 = RMSNorm(C[3])

        self.self_attn = SpatialSelfAttention(C[3], num_heads=8)

        self.tconv4 = nn.ConvTranspose2d(C[3],     C[2], 3, stride=2, padding=1, output_padding=1, bias=False)
        self.tconv3 = nn.ConvTranspose2d(C[2] * 2, C[1], 3, stride=2, padding=1, output_padding=1, bias=False)
        self.tconv2 = nn.ConvTranspose2d(C[1] * 2, C[0], 3, stride=2, padding=1, output_padding=1, bias=False)
        self.tconv1 = nn.Conv2d(C[0] * 2, 1, 3, padding=1)

        self.dense5 = Dense(embed_dim, C[2])
        self.dense6 = Dense(embed_dim, C[1])
        self.dense7 = Dense(embed_dim, C[0])

        self.drnorm4 = RMSNorm(C[2])
        self.drnorm3 = RMSNorm(C[1])
        self.drnorm2 = RMSNorm(C[0])

    def forward(self, x, t, y):
        x     = torch.cat([x, y], dim=1)
        embed = self.act(self.embed(t))

        h1 = self.act(self.rnorm1(self.conv1(x)  + self.dense1(embed)))
        h2 = self.act(self.rnorm2(self.conv2(h1) + self.dense2(embed)))
        h3 = self.act(self.rnorm3(self.conv3(h2) + self.dense3(embed)))
        h4 = self.act(self.rnorm4(self.conv4(h3) + self.dense4(embed)))

        h4 = self.self_attn(h4)

        h  = self.act(self.drnorm4(self.tconv4(h4) + self.dense5(embed)))
        h  = F.interpolate(h, size=h3.shape[2:], mode='nearest')
        h  = self.act(self.drnorm3(self.tconv3(torch.cat([h, h3], dim=1)) + self.dense6(embed)))
        h  = F.interpolate(h, size=h2.shape[2:], mode='nearest')
        h  = self.act(self.drnorm2(self.tconv2(torch.cat([h, h2], dim=1)) + self.dense7(embed)))
        h  = F.interpolate(h, size=h1.shape[2:], mode='nearest')

        out = self.tconv1(torch.cat([h, h1], dim=1))
        out = out / self.marginal_prob_std(t)[:, None, None, None]
        return out


# ══════════════════════════════════════════════════════════════════════════════
# SDE helpers
# ══════════════════════════════════════════════════════════════════════════════

def marginal_prob_std(t, sigma, device):
    t = torch.as_tensor(t, device=device, dtype=torch.float32)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma, device):
    return torch.as_tensor(sigma ** t, device=device, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def valid_mask(x):
    """True where SIC is a real measurement (not NaN or sentinel 254/255)."""
    return ~torch.isnan(x) & (x < SIC_SENTINEL_MIN)

def prepare_cond(y):
    """Clamp FusionNet output to valid [0,100] range, no normalisation."""
    return y.clamp(0.0, 100.0)

def prepare_sic(x):
    """Zero out sentinel/NaN pixels, keep valid SIC in original [0,100] range."""
    return torch.where(valid_mask(x), x, torch.zeros_like(x))

# ══════════════════════════════════════════════════════════════════════════════
# Sampler
# ══════════════════════════════════════════════════════════════════════════════

def Euler_Maruyama_sampler(score_model, marginal_prob_std_fn, diffusion_coeff_fn,
                            y, batch_size, num_steps=500, device='cuda', eps=1e-3,
                            verbose=True):
    """y in [0,100]. Returns samples in [0,100]."""
    from tqdm import tqdm
    t          = torch.ones(batch_size, device=device)
    init_x     = (torch.randn(batch_size, 1, y.shape[-2], y.shape[-1], device=device)
                  * marginal_prob_std_fn(t)[:, None, None, None])
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size  = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in tqdm(time_steps, desc='denoising', leave=False, disable=not verbose):
            bts    = torch.ones(batch_size, device=device) * time_step
            g      = diffusion_coeff_fn(bts)
            mean_x = x + (g**2)[:, None, None, None] * score_model(x, bts, y) * step_size
            x      = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
    return mean_x


# ══════════════════════════════════════════════════════════════════════════════
# Convenience loader
# ══════════════════════════════════════════════════════════════════════════════

def load_models(code_dir, fusion_ckpt_base, postfix, diff_ckpt, best_ckpt, device):
    """
    Load FusionNetASPP and ScoreNet (EMA weights if available).

    Returns
    -------
    fusion_model  : nn.Module  — frozen FusionNetASPP
    score_model   : nn.Module  — DataParallel ScoreNet with EMA weights
    mdl           : class      — the FusionNetASPP class (for .name attribute)
    marginal_prob_std_fn, diffusion_coeff_fn : functools.partial
    fusion_info   : dict       — epoch, val_rmse from fusion checkpoint
    diff_info     : dict       — epoch, val_loss, ema_val_loss from diff checkpoint
    """
    sys.path.append(code_dir)
    from lib.model.FusionNetASPP import FusionNetASPP as mdl

    # ── FusionNetASPP ──────────────────────────────────────────────────────────
    fusion_ckpt_path = os.path.join(fusion_ckpt_base, f'{mdl.name.lower()}/best_model_{postfix}.pth')
    fusion_ckpt      = torch.load(fusion_ckpt_path, map_location=device, weights_only=True)
    fusion_model     = mdl(
        in_channels=fusion_ckpt['in_channels'],
        features=fusion_ckpt['features'],
    ).to(device)
    fusion_model.load_state_dict(fusion_ckpt['model_state_dict'])
    fusion_model.eval()

    fusion_info = {
        'epoch':    fusion_ckpt.get('epoch'),
        'val_rmse': fusion_ckpt.get('val_rmse'),
    }
    print(f'{mdl.name}  epoch={fusion_info["epoch"]}  val_rmse={fusion_info["val_rmse"]:.2f}%')

    # ── ScoreNet ───────────────────────────────────────────────────────────────
    _mps_fn = functools.partial(marginal_prob_std, sigma=SIGMA, device=device)
    _dc_fn  = functools.partial(diffusion_coeff,   sigma=SIGMA, device=device)

    inner_model = ScoreNet(marginal_prob_std=_mps_fn).to(device)

    # Only wrap in DataParallel on CUDA — it doesn't support CPU
    if device.type == 'cuda':
        score_model = torch.nn.DataParallel(inner_model)
    else:
        score_model = inner_model

    infer_path = best_ckpt if os.path.exists(best_ckpt) else diff_ckpt
    ckpt       = torch.load(infer_path, map_location=device, weights_only=True)

    if 'ema_state_dict' in ckpt:
        inner = score_model.module if hasattr(score_model, 'module') else score_model
        inner.load_state_dict(ckpt['ema_state_dict'])
        print(f'ScoreNet: loaded EMA weights from {infer_path}')
    else:
        from collections import OrderedDict
        state_dict = ckpt.get('state_dict', ckpt)
        # Strip 'module.' prefix if checkpoint was saved with DataParallel
        # but we're loading into a plain ScoreNet (CPU)
        if not hasattr(score_model, 'module'):
            state_dict = OrderedDict(
                (k.replace('module.', '', 1), v) for k, v in state_dict.items()
            )
        score_model.load_state_dict(state_dict)
        print(f'ScoreNet: loaded regular weights from {infer_path}')

    diff_info = {
        'epoch':         ckpt.get('epoch'),
        'avg_loss':      ckpt.get('avg_loss'),
        'val_loss':      ckpt.get('val_loss'),
        'ema_val_loss':  ckpt.get('ema_val_loss'),
        'best_val_loss': ckpt.get('best_val_loss'),
    }
    if diff_info['epoch'] is not None:
        print(f'ScoreNet: epoch={diff_info["epoch"]}  '
              f'train={diff_info["avg_loss"]:.5f}  '
              f'val={diff_info["val_loss"]:.5f}  '
              f'ema_val={diff_info["ema_val_loss"]:.5f}')

    score_model.eval()

    return fusion_model, score_model, mdl, _mps_fn, _dc_fn, fusion_info, diff_info
