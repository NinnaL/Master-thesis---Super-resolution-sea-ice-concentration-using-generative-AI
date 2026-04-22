"""
zarr_maintenance.py
-------------------
Two utilities for managing the zarr cache:

    Option A — delete a split entirely (free space + inodes)
    Option B — repack a split to single-chunk-per-sample (fix inode overhead)

Usage:
    python zarr_maintenance.py --action delete --split train
    python zarr_maintenance.py --action repack --split train
    python zarr_maintenance.py --action repack --split val
"""

import os
import shutil
import argparse
import zarr
import numpy as np
from tqdm import tqdm


CACHE_DIR  = "/dmidata/projects/asip-cms/ninna_msc/zarr_cache"
COMPRESSOR = zarr.Blosc(cname='lz4', clevel=3)


# ---------------------------------------------------------------------------
# Option A: delete a split
# ---------------------------------------------------------------------------

def delete_split(split):
    split_dir = os.path.join(CACHE_DIR, split)
    if not os.path.exists(split_dir):
        print(f"{split_dir} does not exist — nothing to delete.")
        return

    answer = input(f"Delete {split_dir}? This cannot be undone. [yes/no]: ")
    if answer.strip().lower() != 'yes':
        print("Aborted.")
        return

    shutil.rmtree(split_dir)
    print(f"Deleted {split_dir}.")


# ---------------------------------------------------------------------------
# Option B: repack to single chunk per sample
# ---------------------------------------------------------------------------

def repack_split(split):
    split_dir  = os.path.join(CACHE_DIR, split)
    tmp_dir    = split_dir + '_repacking'

    if not os.path.exists(split_dir):
        print(f"{split_dir} does not exist.")
        return
    if os.path.exists(tmp_dir):
        print(f"Temp dir {tmp_dir} already exists — previous repack may have failed.")
        print("Delete it manually and re-run if you want to proceed.")
        return

    src  = zarr.open_group(zarr.DirectoryStore(split_dir), mode='r')
    dst  = zarr.open_group(zarr.DirectoryStore(tmp_dir),   mode='w')

    # Copy metadata
    dst.attrs.update(dict(src.attrs))

    indices = sorted(src['amsr2'].keys(), key=int)
    print(f"Repacking {len(indices)} samples in '{split}' split...")

    errors = 0
    for key in tqdm(indices, desc=f'  repacking {split}'):
        try:
            amsr2 = src['amsr2'][key][:]
            sic   = src['sic'][key][:]
            mask  = src['mask'][key][:]

            # Store each sample as a single chunk — one file per array
            dst.require_group('amsr2')[key] = zarr.array(amsr2, chunks=amsr2.shape, compressor=COMPRESSOR)
            dst.require_group('sic')[key]   = zarr.array(sic,   chunks=sic.shape,   compressor=COMPRESSOR)
            dst.require_group('mask')[key]  = zarr.array(mask,  chunks=mask.shape,  compressor=COMPRESSOR)

        except Exception as e:
            print(f"\n  Error at key {key}: {e}")
            errors += 1

    if errors:
        print(f"\n{errors} errors during repack. Temp dir kept at {tmp_dir} for inspection.")
        return

    # Swap old → new atomically
    old_dir = split_dir + '_old'
    os.rename(split_dir, old_dir)
    os.rename(tmp_dir,   split_dir)
    shutil.rmtree(old_dir)

    print(f"Repack complete. {split_dir} now uses single-chunk-per-sample storage.")
    print(f"Inode count reduced from ~{len(indices) * 16} to ~{len(indices) * 3}.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['delete', 'repack'], required=True)
    parser.add_argument('--split',  choices=['train', 'val'],     required=True)
    args = parser.parse_args()

    if args.action == 'delete':
        delete_split(args.split)
    elif args.action == 'repack':
        repack_split(args.split)