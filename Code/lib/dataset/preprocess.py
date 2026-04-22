import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import torch
import torch.nn.functional as F
import zarr
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATA_DIRS = ['/dmidata/projects/asip-cms/tests/new_input_ncs/AMSR2', '/dmidata/projects/asip-cms/reproc']
TRAINING_INDEX_CSV = "/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI/training_index.csv"
CACHE_DIR = "/dmidata/projects/asip-cms/ninna_msc/zarr_cache"

VAL_SPLIT_RATIO = 0.2
SEED = 42
SCALE_FACTOR = 25


def build_filelist(csv_path, data_dirs):
    df = pd.read_csv(csv_path)
    timestamp = df['timestamp']
    year, month, day = timestamp.str.slice(0, 4), timestamp.str.slice(5, 7), timestamp.str.slice(8, 10)
    return pd.DataFrame([
        {
            'amsr2_path': os.path.join(data_dirs[0], y, m, d, amsr2),
            'sic_path':   os.path.join(data_dirs[1], y, m, d, sic),
        }
        for y, m, d, amsr2, sic in zip(year, month, day, df['amsr2_file'], df['sic_file'])
    ])


def get_channel_names(amsr2_path):
    with xr.open_dataset(amsr2_path, engine='h5netcdf') as ds:
        names = list(ds.data_vars)
    return names[:-1]  # drop swath_segmentation

def add_split_column(csv_path, val_ratio=VAL_SPLIT_RATIO, seed=SEED):
    df = pd.read_csv(csv_path)
    train_idx, val_idx = train_test_split(df.index.tolist(), test_size=val_ratio, random_state=seed, shuffle=True)
    df['split'] = 'train'
    df.loc[val_idx, 'split'] = 'val'
    df.to_csv(csv_path, index=False)
    n_train = (df['split'] == 'train').sum()
    n_val   = (df['split'] == 'val').sum()
    print(f'Added split column. Train: {n_train}, Val: {n_val}.')
    return df

def process_pair(amsr2_path, sic_path):
    # Load AMSR2 data
    with xr.open_dataset(amsr2_path, engine='h5netcdf') as ds:
        amsr2 = ds.to_array().values.astype(np.float32)  # (n_freq, H, W)
    amsr2 = amsr2[:-1]                                    # (14, H, W)
    amsr2_h, amsr2_w = amsr2.shape[-2], amsr2.shape[-1]

    # Load SIC data and resample to AMSR2 size
    with rioxarray.open_rasterio(sic_path) as da:
        raw = da.values.astype(np.float32)                # (bands, H_sic, W_sic)

    sic_native = raw[0]                                   # (H_sic, W_sic)
    invalid    = (sic_native == 254) | (sic_native == 255)

    # ── Bicubic downsampling to AMSR2 resolution ─────────────────────────────
    # Follows Liu et al. (2023, Remote Sensing, doi:10.3390/rs15225401) and
    # MFM-Net (Gao et al., arXiv:2406.01240, 2024) — standard in AMSR2 SR.
    # Invalid pixels set to nan before interpolating so they do not contribute
    # to the bicubic kernel. Mask re-derived from nan in output.
    # Clipped to [0, 100] to correct bicubic overshoot at sharp boundaries.
    sic_filled = sic_native.copy().astype(np.float32)
    sic_filled[invalid] = np.nan

    sic_t    = torch.from_numpy(sic_filled)[None, None]
    sic_lr_t = F.interpolate(sic_t, size=(amsr2_h, amsr2_w), mode='bicubic', align_corners=False)
    sic_lr   = sic_lr_t.numpy()[0, 0]
    mask_lr  = np.isnan(sic_lr)
    sic_lr   = np.clip(sic_lr, 0.0, 100.0)

    return (
        amsr2,                                            # (14, H, W) float32
        sic_lr[np.newaxis].astype(np.float32),            # (1,  H, W) float32
        mask_lr[np.newaxis],                              # (1,  H, W) bool
    )


def write_split(df_split, split_name, cache_dir, channel_names):
    split_dir = os.path.join(cache_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    store = zarr.DirectoryStore(split_dir)
    root  = zarr.open_group(store, mode='a')
    root.attrs['channel_names'] = channel_names

    done       = set(root['amsr2'].keys()) if 'amsr2' in root else set()
    compressor = zarr.Blosc(cname='lz4', clevel=3)
    errors     = 0

    for i, row in enumerate(tqdm(df_split.itertuples(), total=len(df_split), desc=split_name)):
        key = str(i)
        if key in done:
            continue
        try:
            amsr2, sic, mask = process_pair(row.amsr2_path, row.sic_path)
            root.require_group('amsr2')[key] = zarr.array(amsr2, chunks=amsr2.shape, compressor=compressor)
            root.require_group('sic')[key]   = zarr.array(sic,   chunks=sic.shape,   compressor=compressor)
            root.require_group('mask')[key]  = zarr.array(mask,  chunks=mask.shape,  compressor=compressor)
        except Exception as e:
            print(f'Error at index {row.Index} ({row.amsr2_path}): {e}')
            errors += 1

    ok = len(df_split) - len(done) - errors
    print(f'{split_name} done. Processed: {ok}, Skipped: {len(done)}, Errors: {errors}.')


if __name__ == '__main__':
    os.makedirs(CACHE_DIR, exist_ok=True)

    file_list = build_filelist(TRAINING_INDEX_CSV, DATA_DIRS)
    print(f'Files to process: {len(file_list)}')

    # add split column to CSV
    add_split_column(TRAINING_INDEX_CSV)

    train_files, val_files = train_test_split(
        file_list, test_size=VAL_SPLIT_RATIO, random_state=SEED, shuffle=True,
    )
    train_files = train_files.reset_index(drop=True)
    val_files   = val_files.reset_index(drop=True)
    print(f'Train: {len(train_files)}  Val: {len(val_files)}')

    channel_names = get_channel_names(file_list.iloc[0]['amsr2_path'])

    write_split(train_files, 'train', CACHE_DIR, channel_names)
    write_split(val_files,   'val',   CACHE_DIR, channel_names)

    print(f'Done → {CACHE_DIR}/train  and  {CACHE_DIR}/val')