import os
import sys
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
import matplotlib.pyplot as plt
 
from lib.dataset.dataloader import AMSR2Dataset, collate_crop_to_min, collate_pad_to_max
# from lib.model.Baseline import EncDec
from lib.model.FusionNet import FusionNet


### Configurations ###
CACHE_DIR  = '/dmidata/projects/asip-cms/ninna_msc/zarr_cache'
# OUTPUT_DIR = "/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI/outputs/training/baseline"
OUTPUT_DIR = "/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI/outputs/training/fusionnet"

### Parameters ###
NUM_EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 1e-4 
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4
SEED = 42
FEATURES = 32

AMSR2_IN_CHANNELS = 14 # The two 89.9 CHz channels for the baseline model
GRAD_CLIP_NORM    = 1.0

postfix = '1'

### Setup ###
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

### Dataset and DataLoader ###
train_dataset = AMSR2Dataset(CACHE_DIR, split='train')
val_dataset   = AMSR2Dataset(CACHE_DIR, split='val')

# For testing
# train_dataset = Subset(train_dataset, range(1000))
# val_dataset   = Subset(val_dataset,   range(200))

### Collate ###
# Pad to max
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=False,
    persistent_workers=True,
    collate_fn=collate_pad_to_max,
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=False,
    persistent_workers=True,
    collate_fn=collate_pad_to_max,
)

# # Crop to min
# train_loader = DataLoader(
#     train_dataset, batch_size=BATCH_SIZE, shuffle=True,
#     num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
#     prefetch_factor=4, collate_fn=collate_crop_to_min,
# )
# val_loader = DataLoader(
#     val_dataset, batch_size=BATCH_SIZE, shuffle=False,
#     num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
#     prefetch_factor=4, collate_fn=collate_crop_to_min,
# )

### Model, loss, optimizer ###
# model = EncDec(in_channels=AMSR2_IN_CHANNELS, features=FEATURES).to(device)
model = FusionNet(in_channels=AMSR2_IN_CHANNELS, features=FEATURES).to(device)
criterion = nn.MSELoss() # L2
# criterion = nn.L1Loss() # MAE for more robustness towards outliers
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, min_lr=1e-7, patience=5, verbose=True)

# ### DataLoader timing test ###
# print("Timing DataLoader...")
# import time

# t0 = time.time()
# for i, (amsr2, sic, mask) in enumerate(train_loader):
#     if i == 0:
#         print(f"  First batch: {time.time()-t0:.2f}s  shapes: amsr2={tuple(amsr2.shape)}  sic={tuple(sic.shape)}")
#     if i >= 9:
#         break
# t_total = time.time() - t0
# print(f"  10 batches: {t_total:.2f}s  ({t_total/10:.2f}s per batch)")
# print(f"  Estimated epoch time: {t_total/10 * len(train_loader):.1f}s  ({len(train_loader)} batches)")

### Save model summary ###
summary_path = os.path.join(OUTPUT_DIR, f'model_summary_{postfix}.txt')
 
class _Tee:
    def __init__(self, f): self._f = f
    def write(self, s): sys.__stdout__.write(s); self._f.write(s)
    def flush(self): sys.__stdout__.flush(); self._f.flush()
 
with open(summary_path, 'w') as f:
    f.write(f"Samples  : train={len(train_dataset)}  val={len(val_dataset)}\n")
    f.write(f"Model    : EncDec_simple  |  in_channels={AMSR2_IN_CHANNELS}  |  features={FEATURES}\n\n")
    sys.stdout = _Tee(f)
    summary(model.features, input_size=(AMSR2_IN_CHANNELS, 199, 212), device=str(device))
    sys.stdout = sys.__stdout__
 

### Metricss ### (Masked for only applying loss to valid pixels)
def masked_rmse(pred, target, mask):
	return torch.sqrt(torch.mean((pred[~mask] - target[~mask])**2)).item()

def masked_mae(pred, target, mask):
    return torch.mean(torch.abs(pred[~mask] - target[~mask])).item()

### Training / validation loop ###
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    skipped = 0
    t_load, t_forward, t_backward = 0.0, 0.0, 0.0
	
    for batch_idx, (amsr2, sic, mask) in enumerate(dataloader):
        t0 = time.time()
        amsr2, sic, mask = amsr2.to(device), sic.to(device), mask.to(device)
        t_load += time.time() - t0

        target_size = (sic.shape[-2], sic.shape[-1])
        valid = ~mask
        if not valid.any():
            continue

        t0 = time.time()
        optimizer.zero_grad() 
        pred = model(amsr2, target_size=target_size)
        loss = criterion(pred[valid], sic[valid])  # Only compute loss on valid pixels
        t_forward += time.time() - t0

        # NaN loss guard - if loss is non-finite skip this batch to avoid corrupting training with bad gradients
        if not torch.isfinite(loss):
            skipped += 1
            optimizer.zero_grad()  # clear any gradients just in case
            print(f'  NaN loss at batch {batch_idx}')
            continue

        t0 = time.time()
        loss.backward()
        # Gradient clipping - caps gradient norm before the optimizer step
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        
        # NaN grad norm guard - if grad norm is non-finite skip optimizer step to avoid corrupting training with bad weights
        if not torch.isfinite(total_norm):
            skipped += 1
            t_backward += time.time() - t0
            optimizer.zero_grad()  # clear any gradients just in case
            print(f'  NaN grad norm at batch {batch_idx}')
            continue
        optimizer.step()
        t_backward += time.time() - t0

        total_loss += loss.item()

    n = max(len(dataloader)-skipped, 1)  # avoid division by zero
    if skipped:
        print(f"  Warning: {skipped} batches skipped due to non-finite loss")
    # print(f"  load={t_load/n:.2f}s  forward={t_forward/n:.2f}s  backward={t_backward/n:.2f}s  per batch")
    return total_loss / n

@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    n_processed = 0
    total_loss, total_rmse, total_mae = 0.0, 0.0, 0.0

    for amsr2, sic, mask in dataloader:
        amsr2, sic, mask = amsr2.to(device), sic.to(device), mask.to(device)
        target_size = (sic.shape[-2], sic.shape[-1])
        valid = ~mask
        if not valid.any():
            continue

        pred = model(amsr2, target_size=target_size)

        batch_loss = criterion(pred[valid], sic[valid]).item()  # Only compute loss on valid pixels
        if not np.isfinite(batch_loss):
            continue
        total_loss += batch_loss
        total_rmse += masked_rmse(pred, sic, mask)
        total_mae += masked_mae(pred, sic, mask)
        n_processed += 1

    n = max(n_processed, 1)  # avoid division by zero
    return total_loss / n, total_rmse / n, total_mae / n

### Training loop ###
history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'val_mae': []}
best_val_loss = float('inf')
best_ckpt_path = os.path.join(OUTPUT_DIR, f'best_baseline_model_{postfix}.pth')
history_path = os.path.join(OUTPUT_DIR, f'training_history_{postfix}.npy')

print("\nStarting training...")
print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Val RMSE':>10} {'Val MAE':>10} {'Time':>8} {'lr':>10}")

for epoch in range(1, NUM_EPOCHS + 1):
    start_time = time.time()
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_rmse, val_mae = validate_epoch(model, val_loader, criterion, device)
    epoch_time = time.time() - start_time

    scheduler.step(val_loss)  # Adjust learning rate based on validation loss
    current_lr = optimizer.param_groups[0]['lr']

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_rmse'].append(val_rmse)
    history['val_mae'].append(val_mae)

    # Saving history as a numpy file for easy loading and plotting later
    np.save(history_path, history)

    print(f"{epoch:>6} {train_loss:>12.4f} {val_loss:>10.4f} {val_rmse:>10.4f} {val_mae:>10.4f} {epoch_time:>7.1f}s  lr={current_lr:.2e}")

    # Save best model checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            # model config
            "in_channels": AMSR2_IN_CHANNELS,  # saved for safe reloading
            "features": FEATURES,
            # training config — for reproducibility and logging
            'num_epochs':           NUM_EPOCHS,
            'batch_size':           BATCH_SIZE,
            'learning_rate':        LEARNING_RATE,
            'weight_decay':         WEIGHT_DECAY,
            'grad_clip_norm':       GRAD_CLIP_NORM,
            'seed':                 SEED,
            'scheduler':            scheduler.state_dict(),
            'cache_dir':            CACHE_DIR,
            'collate':              'pad_to_max',
        }, best_ckpt_path)
        print(f"         ↳ saved best model (val_loss={val_loss:.4f})")
 
print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

### Saving ###
# Saving a sample prediction (first batch of validation batch)
# load best model weights
ckpoint = torch.load(best_ckpt_path, map_location=device, weights_only=True)
model.load_state_dict(ckpoint['model_state_dict'])
model.eval()

amsr2, sic, mask = next(iter(val_loader))
amsr2, sic, mask = amsr2.to(device), sic.to(device), mask.to(device)

with torch.no_grad():
    pred = model(amsr2, target_size=(sic.shape[-2], sic.shape[-1]))

idx = 0 # first sample in the batch
pred_np = pred[idx, 0].cpu().numpy() # (H,W)
target_np = sic[idx, 0].cpu().numpy() # (H,W)
mask_np = mask[idx, 0].cpu().numpy() # (H,W) bool

pred_np_masked = np.where(mask_np, np.nan, pred_np)
target_np_masked = np.where(mask_np, np.nan, target_np)

# Save .npy file
npy_path = os.path.join(OUTPUT_DIR, f'sample_prediction_{postfix}.npz')
np.savez(npy_path, prediction=pred_np_masked, target=target_np_masked, mask=mask_np)
print(f"Sample prediction saved to {npy_path}")

# Save .png visualization
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
vmin, vmax = 0, 100

im0 = axes[0].imshow(target_np_masked, vmin=vmin, vmax=vmax, cmap='Blues_r')
axes[0].set_title('Target SIC')
axes[0].axis('off')
plt.colorbar(im0, ax=axes[0], fraction=0.046, label='SIC (%)')

im1 = axes[1].imshow(pred_np_masked, vmin=vmin, vmax=vmax, cmap='Blues_r')
axes[1].set_title('Predicted SIC')
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], fraction=0.046, label='SIC (%)')

diff = pred_np_masked - target_np_masked
abs_max = np.nanmax(np.abs(diff))
im2 = axes[2].imshow(diff, vmin=-abs_max, vmax=abs_max, cmap='bwr')
axes[2].set_title('Prediction Error')
axes[2].axis('off')
plt.colorbar(im2, ax=axes[2], fraction=0.046, label='Error (%)')

plt.suptitle('Sample Prediction vs Target (Masked) - best baseline model')
plt.tight_layout()
png_path = os.path.join(OUTPUT_DIR, f'sample_prediction_{postfix}.png')
plt.savefig(png_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Sample prediction figure saved to {png_path}")

# Save training curves
history = np.load(history_path, allow_pickle=True).item() # load history dict
epochs = range(1, len(history['train_loss']) + 1)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(epochs, history['train_loss'], label='Train')
axes[0].plot(epochs, history['val_loss'], label='Val')
axes[0].set_title('Loss (MSE)')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, history['val_rmse'], color='darkorange')
axes[1].set_title('Val RMSE')
axes[1].set_xlabel('Epoch')
axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs, history['val_mae'], color='green')
axes[2].set_title('Val MAE')
axes[2].set_xlabel('Epoch')
axes[2].grid(True, alpha=0.3)

plt.suptitle('Training History - Baseline Model')
plt.tight_layout()
history_png_path = os.path.join(OUTPUT_DIR, f'training_curves_{postfix}.png')
plt.savefig(history_png_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Training curves figure saved to {history_png_path}")