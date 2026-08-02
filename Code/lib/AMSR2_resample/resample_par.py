"""
Resamples AMSR2 L1B swaths onto a fixed 2 km polar-stereographic grid
for a date range, in parallel. Skips files already processed.

Usage: python resample_interval.py <start_date> <end_date> [--workers N]
Example: python resample_interval.py 2020-01-01 2020-01-31
"""

import os
import sys
import time
import argparse
import logging
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import numpy as np
import xarray as xr

from AMSR2Resampler import AMSR2Resampler
from grid_config import target_area

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_ROOT   = Path('/dmidata/projects/asip-cms/amsr2')          # parent of year folders
OUTPUT_ROOT = Path('/dmidata/projects/asip-cms/ninna_msc/AMSR2')  # mirrors year/month/day
MAX_WORKERS = 4
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

LOG = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(levelname)s: %(asctime)s: %(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)


def process_one_file(amsr2_file_str, data_root_str, output_root_str, target_grid):
    """Runs in a worker process: resample one AMSR2 file (onto a small
    sub-area covering just its footprint), save under the mirrored
    year/month/day output structure. Skips work entirely if the output
    file already exists."""
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # safety for 'spawn' start method

    amsr2_file  = Path(amsr2_file_str)
    data_root   = Path(data_root_str)
    output_root = Path(output_root_str)

    rel_dir = amsr2_file.relative_to(data_root).parent  # preserves year/month/day
    out_dir = output_root / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base = amsr2_file.stem
    final_path = out_dir / f"{base}_resampled.nc"

    if final_path.exists():
        return ('already_done', amsr2_file.name, 0.0)

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.time()
            resampler = AMSR2Resampler(
                amsr2_file=str(amsr2_file),
                output_dir=str(out_dir),
                target_grid=target_grid,
                hemisphere='N'
            )
            resampler.save_resampled_ds()

            elapsed = time.time() - t0
            return ('done', amsr2_file.name, elapsed)

        except ValueError as e:
            return ('skipped', amsr2_file.name, str(e))

        except OSError as e:
            last_exc = e
            if getattr(e, 'errno', None) == 121 or 'Remote I/O error' in str(e):
                time.sleep(RETRY_DELAY)
                continue
            return ('failed', amsr2_file.name, str(e))

        except Exception as e:
            return ('failed', amsr2_file.name, str(e))

    return ('failed', amsr2_file.name, f"Retries exhausted: {last_exc}")


def daterange(start_date, end_date):
    """Yields every date from start_date to end_date, inclusive."""
    n_days = (end_date - start_date).days
    for i in range(n_days + 1):
        yield start_date + timedelta(days=i)


def gather_files_for_interval(data_root, start_date, end_date):
    """Finds all .h5 files under data_root/YYYY/MM/DD for each date in the interval."""
    files = []
    for d in daterange(start_date, end_date):
        day_dir = data_root / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.day:02d}"
        if day_dir.exists():
            files.extend(sorted(day_dir.glob('*.h5')))
    return files


def main():
    parser = argparse.ArgumentParser(description="Resample AMSR2 swaths over a date interval onto the fixed target grid.")
    parser.add_argument('start_date', type=str, help="Start date, format YYYY-MM-DD")
    parser.add_argument('end_date', type=str, help="End date, format YYYY-MM-DD (inclusive)")
    parser.add_argument('--workers', type=int, default=MAX_WORKERS, help="Number of parallel worker processes")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date   = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    if end_date < start_date:
        print("Error: end_date is before start_date")
        sys.exit(1)

    amsr2_files = gather_files_for_interval(DATA_ROOT, start_date, end_date)
    print(f"Found {len(amsr2_files)} files between {start_date} and {end_date}")

    n_done, n_already, n_skipped, n_failed = 0, 0, 0, 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_one_file, str(f), str(DATA_ROOT), str(OUTPUT_ROOT), target_area
            ): f
            for f in amsr2_files
        }

        for future in as_completed(futures):
            status, name, info = future.result()
            if status == 'done':
                n_done += 1
                print(f"  Done: {name} ({info:.2f}s)")
            elif status == 'already_done':
                n_already += 1
                print(f"  Already processed: {name}")
            elif status == 'skipped':
                n_skipped += 1
                print(f"  Skipped (corrupted): {name} — {info}")
            else:
                n_failed += 1
                LOG.error(f"  Failed: {name} — {info}")

    print(f"\nFinished {start_date} to {end_date}. "
          f"done={n_done}, already_done={n_already}, skipped={n_skipped}, failed={n_failed}")


if __name__ == "__main__":
    main()