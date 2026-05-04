"""
diffusion_predict.py
--------------------
Loads FusionNetASPP (frozen condition encoder) and a trained diffusion
ScoreNet, runs two Euler-Maruyama samples on one validation batch, and
saves a 4-panel plot to the same folder as the diffusion checkpoint.

Author: Ninna Juul Ligaard, MSc thesis, DTU/DMI 2026

Usage:
    python Code/diffusion_predict.py
"""

import os
import sys
import functools
import itertools

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Paths — edit these ────────────────────────────────────────────────────────
CODE_DIR    = '/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI/Code'
CACHE_DIR   = '/dmidata/projects/asip-cms/ninna_msc/zarr_cache'
FUSION_CKPT = '/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI/outputs/training/fusionnetaspp/best_model_1.pth'
DIFF_CKPT   = '/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI/outputs/diffusion/ckpt_diff_fusionnetaspp.pth'

# ── Inference config ──────────────────────────────────────────────────────────
BATCH_SIZE  = 32
NUM_WORKERS = 4
SAMPLE_IDX  = 2     # which sample in the batch to plot
NUM_STEPS   = 500   # Euler-Maruyama steps
SIGMA       = 25.0  # SDE noise schedule

sys.path.append(CODE_DIR)
from lib.model.FusionNetASPP import FusionNetASPP
from lib.dataset.dataloader import AMSR2Dataset, collate_pad_to_max

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ── ScoreNet (2D U-Net score model) ──────────────────────────────────────────
#@title Define the loss function (double click to expand or collapse)

def loss_fn(model, x, y, marginal_prob_std, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
  z = torch.randn_like(x).to(device)
  std = marginal_prob_std(random_t)
  perturbed_x = x + z * std[:, None, None, None]
  score = model(perturbed_x, y, random_t)
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
  return loss

def masked_loss_fn(model, x,  y, marginal_prob_std, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """

  mask = ~torch.isnan(x)  # mask is True where x is not NaN
  x = torch.nan_to_num(x)  
  y = torch.nan_to_num(y)  
  
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
  z = torch.randn_like(x)
  std = marginal_prob_std(random_t)
  perturbed_x = x + z * std[:, None, None, None] # add one dimension here
  score = model(perturbed_x, t=random_t, y=y)#, mask=mask_obs)
  
  # Step 2: Compute the loss as usual
  loss = (score * std[:, None, None, None] + z) ** 2 # add one dimension here
  
  # Step 3: Apply the mask
  loss = loss * mask.float()  # Only consider non-NaN pixels
  
  # Step 4: Compute the mean loss over valid pixels
  loss = torch.sum(loss, dim=(1, 2, 3))  # Sum over spatial dimensions
  valid_pixel_count = torch.sum(mask, dim=(1, 2, 3))  # Count valid pixels
  loss = loss / valid_pixel_count  # Normalize loss by valid pixel count
  loss = torch.mean(loss)  # Average over the batch
  return loss
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


class ScoreNet(nn.Module):
    def __init__(self, marginal_prob_std, channels=(32, 64, 128, 256), embed_dim=256):
        super().__init__()
        self.act   = nn.SiLU()
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        self.marginal_prob_std = marginal_prob_std

        # Encoder
        self.conv1  = nn.Conv2d(2,           channels[0], 3, padding=1,           bias=False)
        self.conv2  = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=False)
        self.conv3  = nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False)
        self.conv4  = nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.dense2 = Dense(embed_dim, channels[1])
        self.dense3 = Dense(embed_dim, channels[2])
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm1 = nn.GroupNorm(4,  channels[0])
        self.gnorm2 = nn.GroupNorm(32, channels[1])
        self.gnorm3 = nn.GroupNorm(32, channels[2])
        self.gnorm4 = nn.GroupNorm(32, channels[3])

        # Decoder
        self.tconv4  = nn.ConvTranspose2d(channels[3],     channels[2], 3, stride=2, padding=1, output_padding=1, bias=False)
        self.tconv3  = nn.ConvTranspose2d(channels[2] * 2, channels[1], 3, stride=2, padding=1, output_padding=1, bias=False)
        self.tconv2  = nn.ConvTranspose2d(channels[1] * 2, channels[0], 3, stride=2, padding=1, output_padding=1, bias=False)
        self.tconv1  = nn.Conv2d(channels[0] * 2, 1, 3, padding=1)
        self.dense5  = Dense(embed_dim, channels[2])
        self.dense6  = Dense(embed_dim, channels[1])
        self.dense7  = Dense(embed_dim, channels[0])
        self.tgnorm4 = nn.GroupNorm(32, channels[2])
        self.tgnorm3 = nn.GroupNorm(32, channels[1])
        self.tgnorm2 = nn.GroupNorm(4,  channels[0])

    def forward(self, x, t, y):
        x     = torch.cat([x, y], dim=1)   # (B, 2, H, W)
        embed = self.act(self.embed(t))

        # Encoder
        h1 = self.act(self.gnorm1(self.conv1(x)  + self.dense1(embed)))
        h2 = self.act(self.gnorm2(self.conv2(h1) + self.dense2(embed)))
        h3 = self.act(self.gnorm3(self.conv3(h2) + self.dense3(embed)))
        h4 = self.act(self.gnorm4(self.conv4(h3) + self.dense4(embed)))

        # Decoder — F.interpolate before each skip cat to handle odd dims
        h  = self.act(self.tgnorm4(self.tconv4(h4) + self.dense5(embed)))
        h  = F.interpolate(h, size=h3.shape[2:], mode='nearest')
        h  = self.act(self.tgnorm3(self.tconv3(torch.cat([h, h3], dim=1)) + self.dense6(embed)))
        h  = F.interpolate(h, size=h2.shape[2:], mode='nearest')
        h  = self.act(self.tgnorm2(self.tconv2(torch.cat([h, h2], dim=1)) + self.dense7(embed)))
        h  = F.interpolate(h, size=h1.shape[2:], mode='nearest')
        out = self.tconv1(torch.cat([h, h1], dim=1))

        out = out / self.marginal_prob_std(t)[:, None, None, None]
        return out


# ── SDE helpers ───────────────────────────────────────────────────────────────
def marginal_prob_std(t, sigma):
    t = torch.tensor(t, device=device)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma):
    return torch.tensor(sigma ** t, device=device)


marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=SIGMA)
diffusion_coeff_fn   = functools.partial(diffusion_coeff,   sigma=SIGMA)


# ── Euler-Maruyama sampler ────────────────────────────────────────────────────
def Euler_Maruyama_sampler(score_model, marginal_prob_std, diffusion_coeff,
                            y, batch_size, num_steps=500, device='cuda', eps=1e-3):
    t      = torch.ones(batch_size, device=device)
    init_x = (torch.randn(batch_size, 1, y.shape[-2], y.shape[-1], device=device)
               * marginal_prob_std(t)[:, None, None, None])
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size  = time_steps[0] - time_steps[1]
    x = init_x
    y = y.nan_to_num()
    with torch.no_grad():
        for time_step in tqdm(time_steps, desc='sampling', leave=False):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g      = diffusion_coeff(batch_time_step)
            mean_x = x + (g ** 2)[:, None, None, None] * score_model(x, batch_time_step, y) * step_size
            x      = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
    return mean_x


# ── Load FusionNetASPP (frozen) ───────────────────────────────────────────────
fusion_ckpt  = torch.load(FUSION_CKPT, map_location=device, weights_only=True)
fusion_model = FusionNetASPP(
    in_channels=fusion_ckpt['in_channels'],
    features=fusion_ckpt['features'],
).to(device)
fusion_model.load_state_dict(fusion_ckpt['model_state_dict'])
fusion_model.eval()
print(f'FusionNetASPP  epoch={fusion_ckpt["epoch"]}  val_rmse={fusion_ckpt["val_rmse"]:.2f}%')


# ── Load ScoreNet ─────────────────────────────────────────────────────────────
score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)
ckpt = torch.load(DIFF_CKPT, map_location=device, weights_only=True)
if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    score_model.load_state_dict(ckpt['state_dict'])
    print(f'Diffusion model  epoch={ckpt.get("epoch", "?")}  avg_loss={ckpt.get("avg_loss", float("nan")):.5f}')
else:
    score_model.load_state_dict(ckpt)
    print('Diffusion model loaded (legacy checkpoint — no epoch info)')
score_model.eval()


# ── Training and validation dataloader ─────────────────────────────────────────────────────
train_dataset = AMSR2Dataset(CACHE_DIR, split='train')
val_dataset   = AMSR2Dataset(CACHE_DIR, split='val')

data_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=False,
    persistent_workers=True, collate_fn=collate_pad_to_max,
)
val_dataloader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=False,
    persistent_workers=True, collate_fn=collate_pad_to_max,
)

amsr2, sic, mask = next(itertools.islice(val_dataloader, 1, None))
amsr2 = amsr2.to(device)
sic   = sic.to(device)
mask  = mask.to(device)

target_size = (sic.shape[-2], sic.shape[-1])


# ── FusionNetASPP condition ───────────────────────────────────────────────────
with torch.no_grad():
    y = fusion_model(amsr2, target_size=target_size)   # (B, 1, H, W)


#@title Training (double click to expand or collapse)

from IPython.display import clear_output
from torch.optim import Adam
from tqdm.autonotebook import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import warnings
warnings.filterwarnings('ignore')

n_epochs = 100
lr       = 5e-4

score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)

optimizer          = Adam(score_model.parameters(), lr=lr)
scheduler          = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400, 450, 600, 800, 1000, 1200, 1400], gamma=0.5)
accumulation_steps = 16
start_epoch        = 0

if os.path.exists(DIFF_CKPT):
    ckpt = torch.load(DIFF_CKPT, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        score_model.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt['epoch'] + 1
        # Replay scheduler steps to restore correct lr
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resumed from epoch {ckpt['epoch']}  loss={ckpt['avg_loss']:.5f}  lr={optimizer.param_groups[0]['lr']:.2e}")
    else:
        # Old checkpoint — raw state dict, no epoch info
        score_model.load_state_dict(ckpt)
        print(f"Resumed from checkpoint (no epoch info — starting scheduler from 0)")
else:
    print('No checkpoint found, training from scratch.')

optimizer.zero_grad()

tqdm_epoch = tqdm(range(start_epoch, start_epoch + n_epochs))
for epoch in tqdm_epoch:
    avg_loss  = 0.
    num_items = 0

    for i, (amsr2, sic, mask) in enumerate(data_loader):
        amsr2 = amsr2.to(device)
        sic   = sic.to(device)
        mask  = mask.to(device)

        x           = sic
        target_size = (x.shape[-2], x.shape[-1])

        with torch.no_grad():
            y = fusion_model(amsr2, target_size)

        loss = masked_loss_fn(score_model, x, y, marginal_prob_std_fn)
        if torch.isnan(loss):
            continue

        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_loss  += loss.item() * accumulation_steps * x.shape[0]
        num_items += x.shape[0]

    tqdm_epoch.set_description(f'Epoch {epoch} | avg loss: {avg_loss / max(num_items, 1):.5f}  lr={optimizer.param_groups[0]["lr"]:.2e}')
    torch.save({
        'state_dict': score_model.state_dict(),
        'epoch':      epoch,
        'avg_loss':   avg_loss / max(num_items, 1),
    }, DIFF_CKPT)
    scheduler.step()


# ── Two independent diffusion samples ────────────────────────────────────────
print('Running diffusion sample #1 ...')
x_estim  = Euler_Maruyama_sampler(
    score_model, marginal_prob_std_fn, diffusion_coeff_fn,
    y.nan_to_num(), batch_size=sic.shape[0],
    num_steps=NUM_STEPS, device=str(device),
)
print('Running diffusion sample #2 ...')
x_estim2 = Euler_Maruyama_sampler(
    score_model, marginal_prob_std_fn, diffusion_coeff_fn,
    y.nan_to_num(), batch_size=sic.shape[0],
    num_steps=NUM_STEPS, device=str(device),
)


# ── Prepare numpy arrays ──────────────────────────────────────────────────────
idx      = SAMPLE_IDX
mask_np  = mask[idx, 0].cpu().numpy().astype(bool)
sic_np   = np.where(mask_np, np.nan, sic[idx, 0].cpu().numpy())
y_np     = np.clip(np.where(mask_np, np.nan, y[idx, 0].detach().cpu().numpy()), 0, 100)
est1_np  = np.clip(np.where(mask_np, np.nan, x_estim[idx, 0].detach().cpu().numpy()),  0, 100)
est2_np  = np.clip(np.where(mask_np, np.nan, x_estim2[idx, 0].detach().cpu().numpy()), 0, 100)


# ── Colormap ──────────────────────────────────────────────────────────────────
try:
    import cmcrameri as cmc
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        return mcolors.LinearSegmentedColormap.from_list(
            f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
            cmap(np.linspace(minval, maxval, n)))
    cmap = truncate_colormap(cmc.cm.oslo, minval=0.2, maxval=1, n=100)
except ImportError:
    cmap = plt.cm.Blues_r.copy()
    cmap.set_bad('lightgray')
    print('cmcrameri not found — using Blues_r colormap instead')


# ── Plot ──────────────────────────────────────────────────────────────────────
vmin, vmax = 0, 100
fig, axes  = plt.subplots(1, 4, figsize=(22, 5))
titles     = ['FusionNetASPP (condition)', 'Target SIC',
              'Diffusion sample #1',       'Diffusion sample #2']
arrays     = [y_np, sic_np, est1_np, est2_np]

for ax, arr, title in zip(axes, arrays, titles):
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(title, fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, label='SIC (%)')

plt.suptitle('FusionNetASPP + Diffusion refinement', fontsize=13)
plt.tight_layout()


# ── Save to same folder as diffusion checkpoint ───────────────────────────────
save_dir  = os.path.dirname(DIFF_CKPT)
save_path = os.path.join(save_dir, f'diffusion_prediction_sample{SAMPLE_IDX}.png')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved → {save_path}')