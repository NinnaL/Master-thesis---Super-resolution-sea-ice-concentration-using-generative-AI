'''python daily_composite.py 2020-03-15'''

import os
import re
import glob
import argparse
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import sys
import numpy as np
import xarray as xr
import torch
import matplotlib
matplotlib.use('Agg')  # no display needed — running headless / batch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
from scipy.ndimage import binary_erosion

# ── Add model code to path ────────────────────────────────────────────────────
CODE_DIR = '/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI/Code'
sys.path.insert(0, CODE_DIR)
from lib.model.FusionNetASPP import FusionNetASPP
from grid_config import target_area

# ── Config ─────────────────────────────────────────────────────────────────────
SWATH_ROOT = Path('/dmidata/projects/asip-cms/ninna_msc/AMSR2')       # year/month/day structure
OUTPUT_DIR = Path('/dmidata/projects/asip-cms/ninna_msc/output_full')
CKPT_DIR   = '/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI/outputs/training'
MODEL_NAME = 'FusionNetASPP'
POSTFIX    = 5
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TILE_SIZE   = 128
OVERLAP     = 16
STRIDE      = TILE_SIZE - OVERLAP
EDGE_TRIM   = 4
TRIM_PIXELS = 20

CHANNELS = [
    "btemp_6.9h", "btemp_6.9v",
    "btemp_7.3h", "btemp_7.3v",
    "btemp_10.7h", "btemp_10.7v",
    "btemp_18.7h", "btemp_18.7v",
    "btemp_23.8h", "btemp_23.8v",
    "btemp_36.5h", "btemp_36.5v",
    "btemp_89.0h", "btemp_89.0v",
]


def parse_swath_time(filename):
    match = re.search(r"_(\d{12})_", os.path.basename(filename))
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d%H%M")
    return datetime.fromtimestamp(os.path.getmtime(filename))


def load_swath(filepath):
    """Load a resampled swath .nc file into a (C, H, W) array in CHANNELS order,
    placing cropped data back into the full target grid if crop attrs are present."""
    with xr.open_dataset(filepath) as ds:
        arrs = [ds[ch].values.astype(np.float32) for ch in CHANNELS]
        tb = np.stack(arrs, axis=0)

        if ds.attrs.get('empty', False):
            H, W = ds.attrs['full_shape']
            return np.full((len(CHANNELS), H, W), np.nan, dtype=np.float32)

        if 'crop_y0' in ds.attrs and 'full_shape' in ds.attrs:
            y0 = ds.attrs['crop_y0']
            x0 = ds.attrs['crop_x0']
            H, W = ds.attrs['full_shape']
            full = np.full((len(CHANNELS), H, W), np.nan, dtype=np.float32)
            h, w = tb.shape[1], tb.shape[2]
            full[:, y0:y0 + h, x0:x0 + w] = tb
            return full

    return tb  # already full-grid, uncropped


def trim_edges(valid_mask, trim_pixels=15):
    if trim_pixels <= 0:
        return valid_mask
    structure = np.ones((3, 3), dtype=bool)
    return binary_erosion(valid_mask, structure=structure, iterations=trim_pixels)


def predict_sic(amsr2_full):
    H, W = amsr2_full.shape[1], amsr2_full.shape[2]
    sic_full   = np.full((H, W), np.nan, dtype=np.float32)
    count_full = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H, STRIDE):
        for x in range(0, W, STRIDE):
            y1 = min(y + TILE_SIZE, H)
            x1 = min(x + TILE_SIZE, W)
            tile = amsr2_full[:, y:y1, x:x1]

            th, tw = tile.shape[1], tile.shape[2]
            ph, pw = TILE_SIZE - th, TILE_SIZE - tw

            tile_in = np.nan_to_num(tile, nan=0.0)
            if ph > 0 or pw > 0:
                tile_in = np.pad(tile_in, ((0, 0), (0, ph), (0, pw)), constant_values=0.0)

            tile_tensor = torch.from_numpy(tile_in).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(tile_tensor, target_size=(TILE_SIZE, TILE_SIZE))

            pred_np = pred[0, 0].cpu().numpy()

            pred_np[:, :EDGE_TRIM]  = np.nan
            pred_np[:, -EDGE_TRIM:] = np.nan
            pred_np[:EDGE_TRIM, :]  = np.nan
            pred_np[-EDGE_TRIM:, :] = np.nan

            pred_np = pred_np[:th, :tw]
            valid_pred = ~np.isnan(pred_np)

            existing     = sic_full[y:y1, x:x1]
            existing_cnt = count_full[y:y1, x:x1]
            has_existing = ~np.isnan(existing)

            new_val = np.where(
                has_existing,
                (np.nan_to_num(existing) * existing_cnt + np.nan_to_num(pred_np)) /
                np.maximum(existing_cnt + 1, 1),
                pred_np)

            sic_full[y:y1, x:x1] = np.where(valid_pred, new_val, existing)
            count_full[y:y1, x:x1] += valid_pred.astype(np.float32)

    orig_nan_mask = np.any(np.isnan(amsr2_full), axis=0)
    sic_full[orig_nan_mask] = np.nan
    sic_full[count_full == 0] = np.nan
    return sic_full


def main():
    parser = argparse.ArgumentParser(description="Run SPICE SIC composite for a single day and save as PNG.")
    parser.add_argument('date', type=str, help="Date to process, format YYYY-MM-DD")
    args = parser.parse_args()

    date = datetime.strptime(args.date, "%Y-%m-%d")
    swath_dir = SWATH_ROOT / f"{date.year:04d}" / f"{date.month:02d}" / f"{date.day:02d}"

    if not swath_dir.exists():
        print(f"Error: {swath_dir} does not exist")
        sys.exit(1)

    swath_files = sorted(swath_dir.glob('*resampled.nc'), key=parse_swath_time)
    print(f"Found {len(swath_files)} swath file(s) for {args.date}")

    if len(swath_files) == 0:
        print("No swath files found — nothing to do.")
        sys.exit(0)

    # ── Load model ──────────────────────────────────────────────────────────────
    global model
    ckpt_path = os.path.join(CKPT_DIR, MODEL_NAME.lower(), f'best_model_{POSTFIX}.pth')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = FusionNetASPP(in_channels=ckpt['in_channels'], features=ckpt['features']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Model loaded from epoch {ckpt["epoch"]} — val_loss={ckpt["val_loss"]:.4f}')

    shp = gpd.read_file(f'{CODE_DIR}/lib/predict/arctic_shp/op_str_maps_circum_polar_40_EPSG3411.shp')

    # ── Run inference per swath, trim edges, composite by averaging overlaps ────
    H, W = target_area.height, target_area.width
    sic_sum   = np.zeros((H, W), dtype=np.float32)
    sic_count = np.zeros((H, W), dtype=np.float32)

    for f in swath_files:
        t = parse_swath_time(f)
        print(f"Processing {os.path.basename(f)} ({t})")

        amsr2_full = load_swath(f)
        sic_swath  = predict_sic(amsr2_full)

        valid = ~np.isnan(sic_swath)
        valid_trimmed = trim_edges(valid, trim_pixels=TRIM_PIXELS)

        sic_sum[valid_trimmed]   += sic_swath[valid_trimmed]
        sic_count[valid_trimmed] += 1

    sic_composite = np.where(sic_count > 0, sic_sum / np.maximum(sic_count, 1), np.nan)
    print("All swaths processed")

    # ── Plot ──────────────────────────────────────────────────────────────────────
    x_coords = np.linspace(target_area.area_extent[0],
                            target_area.area_extent[2],
                            target_area.width)
    y_coords = np.linspace(target_area.area_extent[3],
                            target_area.area_extent[1],
                            target_area.height)

    cmap = plt.cm.Blues_r.copy()
    cmap.set_bad('black')#('lightgray')
    norm = mcolors.Normalize(vmin=0, vmax=100)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('black')
    ax.pcolormesh(x_coords, y_coords, sic_composite,
                  cmap=cmap, norm=norm, shading='auto', zorder=3)
    shp.to_crs('EPSG:3411').plot(ax=ax, facecolor='#c8c8a0', edgecolor='gray',
             linewidth=0.1, zorder=4)
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                 ax=ax, fraction=0.035, pad=0.02, label='SIC (%)')

    valid_mask = ~np.isnan(sic_composite)
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)
    if rows.any() and cols.any():
        row_idx = np.where(rows)[0]
        col_idx = np.where(cols)[0]
        margin = 20
        INSET = 500  # extra pixels to pull inward from each edge, tune this

        y0 = max(row_idx.min() - margin + INSET, 0)
        y1 = min(row_idx.max() + margin - INSET, H - 1)
        x0 = max(col_idx.min() - margin + INSET, 0)
        x1 = min(col_idx.max() + margin - INSET, W - 1)

        ax.set_xlim(min(x_coords[x0], x_coords[x1]), max(x_coords[x0], x_coords[x1]))
        ax.set_ylim(min(y_coords[y0], y_coords[y1]), max(y_coords[y0], y_coords[y1]))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f'{args.date}', fontsize=12)
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"SPICE_SIC_full_{args.date}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()