import os
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from rasterio.enums import Resampling
import matplotlib.pyplot as plt

from lib.dataset.dataloader2 import AMSR2Dataset
from lib.model.Baseline import EncDec

### Configurations ###
DATA_DIRS = ['/dmidata/projects/asip-cms/tests/new_input_ncs/AMSR2', '/dmidata/projects/asip-cms/reproc']
TRAINING_INDEX_CSV = "/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI/training_index2.csv"
OUTPUT_DIR = "/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI/outputs/training/baseline"

TARGET_SIZE = (192, 192)

### Parameters ###
NUM_EPOCHS = 1
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4
SEED = 42

AMSR2_IN_CHANNELS = 14

### Setup ###
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

### Dataset and DataLoader ###
train_dataset = AMSR2Dataset(DATA_DIRS, TRAINING_INDEX_CSV, split='train', target_size=TARGET_SIZE, seed=SEED)
val_dataset   = AMSR2Dataset(DATA_DIRS, TRAINING_INDEX_CSV, split='val', target_size=TARGET_SIZE, seed=SEED)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

### Model, loss, optimizer ###
model = EncDec(in_channels=AMSR2_IN_CHANNELS).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

### Loss masks ### (For only applying loss to valid pixels)
def masked_rmse(pred, target, mask):
	valid_pred = pred[~mask]
	valid_target = target[~mask]
	return torch.sqrt(torch.mean((valid_pred - valid_target)**2)).item()

def masked_mae(pred, target, mask):
    return torch.mean(torch.abs(pred[~mask] - target[~mask])).item()

### Training / validation loop ###
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
	
    for amsr2, sic, mask in dataloader:
        amsr2, sic, mask = amsr2.to(device), sic.to(device), mask.to(device)
        optimizer.zero_grad() 
        pred = model(amsr2)
        loss = criterion(pred[~mask], sic[~mask])  # Only compute loss on valid pixels
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_rmse, total_mae = 0.0, 0.0, 0.0

    for amsr2, sic, mask in dataloader:
        amsr2, sic, mask = amsr2.to(device), sic.to(device), mask.to(device)
        pred = model(amsr2)
        total_loss += criterion(pred[~mask], sic[~mask]).item()  # Only compute loss on valid pixels
        total_rmse += masked_rmse(pred, sic, mask)
        total_mae += masked_mae(pred, sic, mask)
    return total_loss / len(dataloader), total_rmse / len(dataloader), total_mae / len(dataloader)

### Training loop ###
history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'val_mae': []}
best_val_loss = float('inf')
best_ckpt_path = os.path.join(OUTPUT_DIR, 'best_baseline_model.pth')

print("Starting training...")
print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Val RMSE':>10} {'Val MAE':>10}")

for epoch in range(1, NUM_EPOCHS + 1):
	train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
	val_loss, val_rmse, val_mae = validate_epoch(model, val_loader, criterion, device)

	history['train_loss'].append(train_loss)
	history['val_loss'].append(val_loss)
	history['val_rmse'].append(val_rmse)
	history['val_mae'].append(val_mae)

	print(f"{epoch:>6} {train_loss:>12.4f} {val_loss:>10.4f} {val_rmse:>10.4f} {val_mae:>10.4f}")

	# Save best model checkpoint
	if val_loss < best_val_loss:
		best_val_loss = val_loss
		torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss":             val_loss,
            "val_rmse":             val_rmse,
            "val_mae":              val_mae,
            "in_channels":          AMSR2_IN_CHANNELS,  # saved for safe reloading
        }, best_ckpt_path)
		print(f"         ↳ saved best model (val_loss={val_loss:.4f})")
 
print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

### Saving ###
# Saving history as a numpy file for easy loading and plotting later
history_path = os.path.join(OUTPUT_DIR, 'training_history.npy')
np.save(history_path, history)
print(f"Training history saved to {history_path}")

# Saving a sample prediction (first batch of validation batch)
# load best model weights
ckpoint = torch.load(best_ckpt_path, map_location=device, weights_only=True)
model.load_state_dict(ckpoint['model_state_dict'])
model.eval()

amsr2, sic, mask = next(iter(val_loader))
amsr2, sic, mask = amsr2.to(device), sic.to(device), mask.to(device)

with torch.no_grad():
    pred = model(amsr2)

idx = 0 # first sample in the batch
pred_np = pred[idx, 0].cpu().numpy() # (H,W)
target_np = sic[idx, 0].cpu().numpy() # (H,W)
mask_np = mask[idx, 0].cpu().numpy() # (H,W) bool

pred_np_masked = np.where(mask_np, np.nan, pred_np)
target_np_masked = np.where(mask_np, np.nan, target_np)

# Save .npy file
npy_path = os.path.join(OUTPUT_DIR, 'sample_prediction.npz')
np.save(npy_path, {'prediction': pred_np_masked, 'target': target_np_masked, 'mask': mask_np})
print(f"Sample prediction saved to {npy_path}")

# Save .png visualization
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
vmin, vmax = 0, 100

im0 = axes[0].imshow(target_np_masked, vmin=vmin, vmax=vmax, cmap='Blues')
axes[0].set_title('Target SIC')
axes[0].axis('off')
plt.colorbar(im0, ax=axes[0], fraction=0.046, label='SIC (%)')

im1 = axes[1].imshow(pred_np_masked, vmin=vmin, vmax=vmax, cmap='Blues')
axes[1].set_title('Predicted SIC')
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], fraction=0.046, label='SIC (%)')

diff = pred_np_masked - target_np_masked
abs_max = np.nanmax(np.abs(diff))
im2 = axes[2].imshow(diff, vmin=-abs_max, vmax=abs_max, cmap='RdBu_r')
axes[2].set_title('Prediction Error')
axes[2].axis('off')
plt.colorbar(im2, ax=axes[2], fraction=0.046, label='Error (%)')

plt.suptitle('Sample Prediction vs Target (Masked) - best baseline model')
plt.tight_layout()
png_path = os.path.join(OUTPUT_DIR, 'sample_prediction.png')
plt.savefig(png_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Sample prediction figure saved to {png_path}")

# Save training curves
epochs = range(1, NUM_EPOCHS + 1)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(epochs, history['train_loss'], label='Train')
axes[0].plot(epochs, history['val_loss'], label='Val')
axes[0].set_title('Loss (MSE)')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, history['val_rmse'], color='tab_orange')
axes[1].set_title('Val RMSE')
axes[1].set_xlabel('Epoch')
axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs, history['val_mae'], color='tab_green')
axes[2].set_title('Val MAE')
axes[2].set_xlabel('Epoch')
axes[2].grid(True, alpha=0.3)

plt.suptitle('Training History - Baseline Model')
plt.tight_layout()
history_png_path = os.path.join(OUTPUT_DIR, 'training_curves.png')
plt.savefig(history_png_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Training curves figure saved to {history_png_path}")