"""
diffusion_SR_ASIP_from_AMSR2.py
--------------------
Trains a score-based diffusion model conditioned on FusionNetASPP predictions,
then runs inference on one validation batch and saves a 4-panel plot to the
same folder as the diffusion checkpoint.

Author: Ninna Juul Ligaard, MSc thesis, DTU/DMI 2026

Usage:
    python Code/diffusion_SR_ASIP_from_AMSR2.py            # train then predict
    python Code/diffusion_SR_ASIP_from_AMSR2.py --predict  # predict only (skip training)
"""

import os
import sys
import json
import argparse
import functools
import itertools
import warnings
warnings.filterwarnings('ignore')

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

postfix = '3'

# ── Paths ─────────────────────────────────────────────────────────────────────
CODE_DIR    = '/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI/Code'
CACHE_DIR   = '/dmidata/projects/asip-cms/ninna_msc/zarr_cache'
FUSION_CKPT_BASE = '/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI/outputs/training'
OUTPUT_DIR   = '/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI/outputs/diffusion'

# ── Training config ───────────────────────────────────────────────────────────
N_EPOCHS           = 500
BATCH_SIZE         = 32
NUM_WORKERS        = 4
LR                 = 5e-4
ACCUMULATION_STEPS = 4
SIGMA              = 25
MILESTONES         = [100, 200, 300, 400, 450]
EMA_DECAY          = 0.999

# ── Inference config ──────────────────────────────────────────────────────────
NUM_STEPS   = 500
SAMPLE_IDX  = 2
SIC_SENTINEL_MIN = 254 #254 = missing, 255=land

# ── Imports from diffusion_model.py ──────────────────────────────────────────
# All model classes, SDE helpers, data helpers, sampler, and EMA live there.
sys.path.insert(0, CODE_DIR)
from lib.model.diffusion_model import (
    ScoreNet,
    marginal_prob_std, diffusion_coeff,
    SIGMA, SIC_SENTINEL_MIN,
    valid_mask, prepare_sic, prepare_cond,
    Euler_Maruyama_sampler, EMA,
)

from lib.model.FusionNetASPP import FusionNetASPP as mdl
from lib.dataset.dataloader import AMSR2Dataset, collate_pad_to_max

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ── SDE partials ─────────────────────────────────
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=SIGMA, device=device)
diffusion_coeff_fn   = functools.partial(diffusion_coeff,   sigma=SIGMA, device=device)

# ── Load conditioning model ───────────────────────────────────────────────────
FUSION_CKPT = os.path.join(FUSION_CKPT_BASE, f'{mdl.name.lower()}/best_model_{postfix}.pth')
fusion_ckpt  = torch.load(FUSION_CKPT, map_location=device, weights_only=True)
fusion_model = mdl(
    in_channels=fusion_ckpt['in_channels'],
    features=fusion_ckpt['features'],
).to(device)
fusion_model.load_state_dict(fusion_ckpt['model_state_dict'])
fusion_model.eval()
print(f'{mdl.name}  epoch={fusion_ckpt["epoch"]}  val_rmse={fusion_ckpt["val_rmse"]:.2f}%')
 
# ── Derived output paths ─────────────────────────────────────────
DIFF_CKPT    = f'{OUTPUT_DIR}/ckpt_{mdl.name.lower()}_{postfix}.pth'
BEST_CKPT    = f'{OUTPUT_DIR}/best_ckpt_{mdl.name.lower()}_{postfix}.pth'
HISTORY_PATH = f'{OUTPUT_DIR}/history_{mdl.name.lower()}_{postfix}.json'

# ── Masked loss ───────────────────────────────────────────────────────────────
def masked_loss_fn(model, x, y, marginal_prob_std, eps=1e-5):
    """
    Score-matching loss with sentinel masking and SNR weighting.
      - Only valid SIC pixels [0,100] contribute to the loss
      - SNR weight 1/(std²+0.1) upweights low-noise timesteps that
        most determine output sharpness and fine ice-edge detail
    """
    mask = valid_mask(x)
    x    = prepare_sic(x)
    y    = prepare_cond(y)
 
    random_t    = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    z           = torch.randn_like(x)
    std         = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]
 
    score      = model(perturbed_x, t=random_t, y=y)
    loss       = (score * std[:, None, None, None] + z) ** 2
    snr_weight = 1.0 / (std[:, None, None, None] ** 2 + 10.0) 
    loss       = loss * snr_weight * mask.float()
    loss       = torch.sum(loss, dim=(1,2,3)) / torch.sum(mask, dim=(1,2,3)).clamp(min=1)
    return torch.mean(loss)

# ── Training history ───────────────────────────────────────────────────────────────────
def load_history(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {'epoch': [], 'train_loss': [], 'val_loss': [], 'ema_val_loss': [], 'lr': []}

def save_history(history, path):
    with open(path, 'w') as f: json.dump(history, f, indent=2)
 
def plot_history(history, save_dir):
    epochs = history['epoch']
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(epochs, history['train_loss'],   label='Train')
    axes[0].plot(epochs, history['val_loss'],     label='Val (train weights)')
    if history.get('ema_val_loss'):
        axes[0].plot(epochs, history['ema_val_loss'], label='Val (EMA weights)', linestyle='--')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('SNR-weighted loss')
    axes[0].set_title('Loss curves'); axes[0].legend(); axes[0].grid(alpha=0.3)
    if len(epochs) > 10: axes[0].set_yscale('log')
    axes[1].plot(epochs, history['lr'], color='tab:orange')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('LR')
    axes[1].set_title('Learning rate'); axes[1].set_yscale('log'); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'history_{mdl.name.lower()}_{postfix}.png'), dpi=120, bbox_inches='tight')
    plt.close()

# ── Dataloaders ───────────────────────────────────────────────────────────────
train_dataset = AMSR2Dataset(CACHE_DIR, split='train')
val_dataset   = AMSR2Dataset(CACHE_DIR, split='val')

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=False,
    persistent_workers=True, collate_fn=collate_pad_to_max,
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=False,
    persistent_workers=True, collate_fn=collate_pad_to_max,
)
print(f'Train: {len(train_dataset)} samples  Val: {len(val_dataset)} samples')


# ── ScoreNet + optimizer + EMA ──────────────────────────────────────────────────
score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)

optimizer  = Adam(score_model.parameters(), lr=LR)
scheduler  = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.5)
ema = EMA(score_model.module, decay=EMA_DECAY)
start_epoch = 0
best_val = float('inf')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Resume ────────────────────────────────────────────────────────────────────
if os.path.exists(DIFF_CKPT):
    ckpt = torch.load(DIFF_CKPT, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        score_model.load_state_dict(ckpt['state_dict'])
        # if 'ema_state_dict' in ckpt:
        #     ema.load_state_dict(ckpt['ema_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val    = ckpt.get('best_val_loss', float('inf'))
        for _ in range(start_epoch):
            scheduler.step()
        print(f'Resumed  epoch={start_epoch-1}  '
              f'train={ckpt.get("avg_loss",float("nan")):.5f}  '
              f'val={ckpt.get("val_loss",float("nan")):.5f}  '
              f'lr={optimizer.param_groups[0]["lr"]:.2e}')
    else:
        score_model.load_state_dict(ckpt)
        print('Resumed from legacy checkpoint')
else:
    print('No checkpoint — training from scratch')


# ── Parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--predict', action='store_true',
                    help='Skip training and run inference only')
args, _ = parser.parse_known_args()


# ── Training loop ─────────────────────────────────────────────────────────────
if not args.predict:
    history = load_history(HISTORY_PATH)
    print(f'\nStarting training from epoch {start_epoch} ...')
    optimizer.zero_grad()

    for epoch in range(start_epoch, start_epoch + N_EPOCHS):
        # Training
        score_model.train()
        avg_loss  = 0.
        num_items = 0
        pending_grad = False  # track if we have unstepped gradients from accumulation

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
        for i, (amsr2, sic, _) in enumerate(pbar):
            amsr2 = amsr2.to(device)
            sic   = sic.to(device)

            with torch.no_grad():
                y = fusion_model(amsr2, target_size=(sic.shape[-2], sic.shape[-1]))

            loss = masked_loss_fn(score_model, sic, y, marginal_prob_std_fn)
            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            (loss / ACCUMULATION_STEPS).backward()
            pending_grad = True   # ← track that we have unstepped gradients

            if (i + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                ema.update(score_model.module)
                pending_grad = False

            avg_loss  += loss.item() * sic.shape[0]
            num_items += sic.shape[0]
            pbar.set_postfix(loss=f'{avg_loss / max(num_items, 1):.5f}')

        epoch_train_loss = avg_loss / max(num_items, 1)

        # Flush remaining accumulated gradients
        if pending_grad:
            optimizer.step()
            optimizer.zero_grad()
            ema.update(score_model.module)
            pending_grad = False

        # Validation
        score_model.eval()
        val_loss  = 0.
        val_items = 0
        with torch.no_grad():
            for amsr2, sic, _ in val_loader:
                amsr2 = amsr2.to(device)
                sic   = sic.to(device)
                y     = fusion_model(amsr2, target_size=(sic.shape[-2], sic.shape[-1]))
                loss  = masked_loss_fn(score_model, sic, y, marginal_prob_std_fn)
                if not torch.isnan(loss):
                    val_loss  += loss.item() * sic.shape[0]
                    val_items += sic.shape[0]
        epoch_val_loss = val_loss / max(val_items, 1)

        # EMA validation loss 
        # Temporarily load EMA weights to get the val loss that reflects what inference will actually see
        train_state = {k: v.clone() for k, v in score_model.state_dict().items()}
        score_model.module.load_state_dict(ema.state_dict())
        ema_val_loss  = 0.
        ema_val_items = 0
        with torch.no_grad():
            for amsr2, sic, _ in val_loader:
                amsr2 = amsr2.to(device)
                sic   = sic.to(device)
                y     = fusion_model(amsr2, target_size=(sic.shape[-2], sic.shape[-1]))
                loss  = masked_loss_fn(score_model, sic, y, marginal_prob_std_fn)
                if not torch.isnan(loss):
                    ema_val_loss  += loss.item() * sic.shape[0]
                    ema_val_items += sic.shape[0]
        epoch_ema_val_loss = ema_val_loss / max(ema_val_items, 1)
        score_model.load_state_dict(train_state)  # restore training weights
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch:>4}  train={epoch_train_loss:.5f}  '
              f'val={epoch_val_loss:.5f}  ema_val={epoch_ema_val_loss:.5f}  '
              f'lr={current_lr:.2e}')
        
        # ── Save ──────────────────────────────────────────────────────────────
        ckpt_dict = {
            'state_dict':     score_model.state_dict(),
            'ema_state_dict': ema.state_dict(),
            'epoch':          epoch,
            'avg_loss':       epoch_train_loss,
            'val_loss':       epoch_val_loss,
            'ema_val_loss':   epoch_ema_val_loss,
            'best_val_loss':  best_val,
        }
        torch.save(ckpt_dict, DIFF_CKPT)
 
        if epoch_ema_val_loss < best_val:   # track best on EMA val loss
            best_val = epoch_ema_val_loss
            ckpt_dict['best_val_loss'] = best_val
            torch.save(ckpt_dict, BEST_CKPT)
            print(f'  ↳ New best ema_val={best_val:.5f} → {BEST_CKPT}')
 
        history['epoch'].append(epoch)
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['ema_val_loss'].append(epoch_ema_val_loss)
        history['lr'].append(current_lr)
        save_history(history, HISTORY_PATH)
        if epoch % 10 == 0 or epoch == start_epoch + N_EPOCHS - 1:
            plot_history(history, os.path.dirname(DIFF_CKPT))
 
    print(f'Training complete  best_val={best_val:.5f}')


# ── Inference ─────────────────────────────────────────────────────────────────
print('\nRunning inference ...')
infer_path = BEST_CKPT if os.path.exists(BEST_CKPT) else DIFF_CKPT
ckpt       = torch.load(infer_path, map_location=device, weights_only=True)
 
# Load EMA shadow weights for inference
if 'ema_state_dict' in ckpt:
    score_model.module.load_state_dict(ckpt['ema_state_dict'])
    print(f'Using EMA weights from {infer_path}')
else:
    score_model.load_state_dict(ckpt.get('state_dict', ckpt))
    print(f'Using regular weights from {infer_path}')
 
score_model.eval()
 
amsr2, sic, mask = next(itertools.islice(val_loader, 1, None))
amsr2 = amsr2.to(device)
sic   = sic.to(device)
mask  = mask.to(device)
 
with torch.no_grad():
    y      = fusion_model(amsr2, target_size=(sic.shape[-2], sic.shape[-1]))
    y_cond = prepare_cond(y)
 
print('Sample #1 ...')
x_estim  = Euler_Maruyama_sampler(
    score_model, marginal_prob_std_fn, diffusion_coeff_fn,
    y_cond, batch_size=sic.shape[0], num_steps=NUM_STEPS, device=str(device),
)
print('Sample #2 ...')
x_estim2 = Euler_Maruyama_sampler(
    score_model, marginal_prob_std_fn, diffusion_coeff_fn,
    y_cond, batch_size=sic.shape[0], num_steps=NUM_STEPS, device=str(device),
)
 
# ── Plot ──────────────────────────────────────────────────────────────────────
idx       = SAMPLE_IDX
sic_raw   = sic[idx, 0].cpu().numpy()
plot_mask = mask[idx, 0].cpu().numpy().astype(bool) | (sic_raw >= SIC_SENTINEL_MIN)
 
sic_np   = np.where(plot_mask, np.nan, sic_raw)
y_np     = np.where(plot_mask, np.nan, np.clip(y[idx, 0].detach().cpu().numpy(), 0, 100))
est1_np  = np.where(plot_mask, np.nan, np.clip(x_estim[idx,  0].detach().cpu().numpy(), 0, 100))
est2_np  = np.where(plot_mask, np.nan, np.clip(x_estim2[idx, 0].detach().cpu().numpy(), 0, 100))
diff_np  = np.where(plot_mask, np.nan, np.abs(est1_np - est2_np))
 
try:
    import cmcrameri as cmc
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        return mcolors.LinearSegmentedColormap.from_list(
            f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
            cmap(np.linspace(minval, maxval, n)))
    sic_cmap = truncate_colormap(cmc.cm.oslo, minval=0.2, maxval=1, n=100)
except ImportError:
    sic_cmap = plt.cm.Blues_r.copy()
    sic_cmap.set_bad('lightgray')
 
fig, axes = plt.subplots(1, 5, figsize=(28, 5))
panels = [
    (y_np,    'FusionNetASPP (condition)', sic_cmap, 0, 100, 'SIC (%)'),
    (sic_np,  'Target SIC',               sic_cmap, 0, 100, 'SIC (%)'),
    (est1_np, 'Diffusion sample #1',       sic_cmap, 0, 100, 'SIC (%)'),
    (est2_np, 'Diffusion sample #2',       sic_cmap, 0, 100, 'SIC (%)'),
    (diff_np, '|Sample1 − Sample2|',       'hot_r',  0,  30, 'ΔSIC (%)'),
]
for ax, (arr, title, cm, vn, vx, label) in zip(axes, panels):
    im = ax.imshow(arr, cmap=cm, vmin=vn, vmax=vx, interpolation='nearest')
    ax.set_title(title, fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, label=label)
 
plt.suptitle(f'{mdl.name} — Diffusion refinement', fontsize=13)
plt.tight_layout()
 
save_path = os.path.join(OUTPUT_DIR, f'prediction_{mdl.name.lower()}_sample{SAMPLE_IDX}.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved → {save_path}')