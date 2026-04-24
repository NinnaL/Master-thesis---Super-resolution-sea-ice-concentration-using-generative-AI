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

MIXED_ICE_COLS = ['0-10', '10-20', '20-30', '30-40', '40-50',
                  '50-60', '60-70', '70-80', '80-90', '90-100']
BIN_COLS = ['val_0'] + MIXED_ICE_COLS + ['val_100']


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

def _compute_row_stats(row):
    """Compute derived columns for a metadata row to match training index format."""
    total = sum(row[c] for c in BIN_COLS)
    if total == 0:
        total = np.nan
 
    frac_mixed      = sum(row[c] for c in MIXED_ICE_COLS) / total
    frac_pure       = (row['val_0'] + row['val_100']) / total
    frac_open_water = row['val_0'] / total
    frac_ice100     = row['val_100'] / total
 
    ice_class = pd.cut(
        [frac_mixed], bins=[0, 0.3, 0.6, 1.01],
        labels=['low_mix', 'mid_mix', 'high_mix']
    )[0]
 
    ts    = pd.to_datetime(str(row['timestamp']), format='%Y%m%dT%H%M%S', errors='coerce')
    year  = ts.year  if pd.notna(ts) else np.nan
    month = ts.month if pd.notna(ts) else np.nan
 
    stratum = f'{str(int(month)).zfill(2)}_{ice_class}' if pd.notna(month) else np.nan
 
    return {
        'year':           year,
        'month':          month,
        'frac_mixed':     frac_mixed,
        'frac_pure':      frac_pure,
        'frac_open_water':frac_open_water,
        'frac_ice100':    frac_ice100,
        'ice_class':      ice_class,
        'stratum':        stratum,
        'split':          'train',
    }

def replace_bad_indices(bad_indices, csv_path, meta_path, data_dirs, cache_dir,
                        split_name, val_ratio=VAL_SPLIT_RATIO, seed=SEED):
    """
    Replace bad samples in the zarr cache with new samples drawn from the
    full metadata CSV, excluding files already in the training index.
 
    For each bad index:
      1. Samples a replacement row from the metadata not already in the CSV
      2. Computes all derived columns to match the training index format
      3. Processes the replacement file pair
      4. Deletes the bad zarr arrays and writes the replacement
      5. Updates the training index CSV with the replacement row
    """
    # Load training index
    training_df = pd.read_csv(csv_path)
 
    # Load metadata and exclude already-used files
    meta_df    = pd.read_csv(meta_path, low_memory=False)
    meta_df.columns = meta_df.columns.str.strip()
    meta_df    = meta_df[meta_df['error'].isna()].copy()   # remove corrupt rows
 
    used_amsr2 = set(training_df['amsr2_file'])
    used_sic   = set(training_df['sic_file'])
    candidates = meta_df[
        ~meta_df['amsr2_file'].isin(used_amsr2) &
        ~meta_df['sic_file'].isin(used_sic)
    ].sample(frac=1, random_state=seed).reset_index(drop=True)
 
    if len(candidates) < len(bad_indices):
        raise ValueError(
            f'Not enough replacement candidates: need {len(bad_indices)}, have {len(candidates)}'
        )
 
    # Open zarr store
    split_dir  = os.path.join(cache_dir, split_name)
    root       = zarr.open_group(zarr.DirectoryStore(split_dir), mode='a')
    compressor = zarr.Blosc(cname='lz4', clevel=3)
 
    errors = 0
    for i, bad_idx in enumerate(tqdm(bad_indices, desc='replacing bad indices')):
        key         = str(bad_idx)
        replacement = candidates.iloc[i]
 
        # Build file paths
        ts = pd.to_datetime(str(replacement['timestamp']), format='%Y%m%dT%H%M%S', errors='coerce')
        y  = str(ts.year).zfill(4)
        m  = str(ts.month).zfill(2)
        d  = str(ts.day).zfill(2)
        amsr2_path = os.path.join(data_dirs[0], y, m, d, replacement['amsr2_file'])
        sic_path   = os.path.join(data_dirs[1], y, m, d, replacement['sic_file'])
 
        try:
            amsr2, sic, mask = process_pair(amsr2_path, sic_path)
 
            # Delete existing bad arrays
            for group in ['amsr2', 'sic', 'mask']:
                if key in root[group]:
                    del root[group][key]
 
            # Write replacement arrays
            root.require_group('amsr2')[key] = zarr.array(amsr2, chunks=amsr2.shape, compressor=compressor)
            root.require_group('sic')[key]   = zarr.array(sic,   chunks=sic.shape,   compressor=compressor)
            root.require_group('mask')[key]  = zarr.array(mask,  chunks=mask.shape,  compressor=compressor)
 
            # Build full replacement row matching training index columns
            new_row = replacement.to_dict()
            new_row.update(_compute_row_stats(replacement))
 
            # Align to training_df columns and assign
            new_series = pd.Series(new_row)
            common_cols = training_df.columns.intersection(new_series.index)
            training_df.iloc[
                bad_idx,
                training_df.columns.get_indexer(common_cols)
            ] = new_series[common_cols].values
 
            print(f'  Replaced index {bad_idx} with {replacement["amsr2_file"]}')
 
        except Exception as e:
            print(f'  Error replacing index {bad_idx}: {e}')
            errors += 1
 
    # Save updated CSV
    training_df.to_csv(csv_path, index=False)
    print(f'\nTraining index updated → {csv_path}')
    print(f'Done. Replaced: {len(bad_indices) - errors}, Errors: {errors}.')
 
 
def reprocess_indices(indices, df_split, split_name, cache_dir, channel_names):
    """
    Reprocess and overwrite specific indices in the zarr cache.
    Use this to fix bad samples identified by the dataset scan.
    """
    split_dir  = os.path.join(cache_dir, split_name)
    store      = zarr.DirectoryStore(split_dir)
    root       = zarr.open_group(store, mode='a')
    compressor = zarr.Blosc(cname='lz4', clevel=3)
    errors     = 0
 
    for i in tqdm(indices, desc=f'reprocessing {split_name}'):
        key = str(i)
        row = df_split.iloc[i]
        try:
            amsr2, sic, mask = process_pair(row.amsr2_path, row.sic_path)
 
            # Delete existing arrays before overwriting
            for group in ['amsr2', 'sic', 'mask']:
                if key in root[group]:
                    del root[group][key]
 
            root.require_group('amsr2')[key] = zarr.array(amsr2, chunks=amsr2.shape, compressor=compressor)
            root.require_group('sic')[key]   = zarr.array(sic,   chunks=sic.shape,   compressor=compressor)
            root.require_group('mask')[key]  = zarr.array(mask,  chunks=mask.shape,  compressor=compressor)
        except Exception as e:
            print(f'Error at index {i}: {e}')
            errors += 1
 
    print(f'Done. Reprocessed: {len(indices) - errors}, Errors: {errors}.')

if __name__ == '__main__':
    os.makedirs(CACHE_DIR, exist_ok=True)

    file_list = build_filelist(TRAINING_INDEX_CSV, DATA_DIRS)
    print(f'Files to process: {len(file_list)}')

    # # add split column to CSV
    # add_split_column(TRAINING_INDEX_CSV)

    train_files, val_files = train_test_split(
        file_list, test_size=VAL_SPLIT_RATIO, random_state=SEED, shuffle=True,
    )
    train_files = train_files.reset_index(drop=True)
    val_files   = val_files.reset_index(drop=True)
    print(f'Train: {len(train_files)}  Val: {len(val_files)}')

    channel_names = get_channel_names(file_list.iloc[0]['amsr2_path'])

    #  ── Process files ──────────────────────────────
    # # write_split(train_files, 'train', CACHE_DIR, channel_names)
    # write_split(val_files,   'val',   CACHE_DIR, channel_names)

    # print(f'Done → {CACHE_DIR}/train  and  {CACHE_DIR}/val')

    # ── Optional: reprocess specific bad indices ──────────────────────────────
    # Uncomment and set bad_indices after running the dataset scan

    bad_indices = [4591, 5135, 7841, 16063, 28045, 28885, 33777, 37406, 38530]  # e.g. [42, 137, 891]
    if bad_indices:
        replace_bad_indices(
            bad_indices = bad_indices,
            csv_path    = TRAINING_INDEX_CSV,
            meta_path   = '/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI/Data/meta/sic_amsr2_metadata_stats_all_years.csv',
            data_dirs   = DATA_DIRS,
            cache_dir   = CACHE_DIR,
            split_name  = 'train',
        )