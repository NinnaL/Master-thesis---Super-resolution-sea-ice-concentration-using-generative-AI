import os
import glob
import re
import random
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from rasterio.enums import Resampling
import xarray as xr
import rioxarray


DEFAULT_TRAINING_INDEX_CSV = "/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI/training_index.csv"
DEFAULT_DATA_DIRS = ['/dmidata/projects/asip-cms/tests/new_input_ncs/AMSR2','/dmidata/projects/asip-cms/reproc']


class SICTrainingDataset(Dataset):
    """PyTorch dataset backed by rows in training_index.csv."""

    def __init__(
        self,
        training_index_csv=DEFAULT_TRAINING_INDEX_CSV,
        data_dirs=DEFAULT_DATA_DIRS,
        image_size=128,
        date_pattern=r"\d{8}[Tt]\d{6}",
    ):
        self.training_index_csv = training_index_csv
        self.data_dirs = [data_dirs] if isinstance(data_dirs, str) else list(data_dirs)
        self.image_size = image_size
        self.date_pattern = date_pattern

        self.df = pd.read_csv(training_index_csv, low_memory=False)
        self.df.columns = self.df.columns.str.strip()

        required_cols = {"amsr2_file", "sic_file"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in {training_index_csv}: {sorted(missing)}"
            )

        self.file_lookup = self._build_file_lookup(self.data_dirs)
        self.samples = self._build_samples(self.df, self.file_lookup)

        if not self.samples:
            raise RuntimeError(
                "No valid (AMSR2, SIC) samples found from training_index.csv. "
                "Verify --data-dirs points to folders containing the listed files."
            )

    @staticmethod
    def _build_file_lookup(data_dirs):
        lookup = {}
        for data_dir in data_dirs:
            for file_path in glob.iglob(os.path.join(data_dir, "**", "*"), recursive=True):
                if not os.path.isfile(file_path):
                    continue
                basename = os.path.basename(file_path)
                if basename.endswith(".nc") or basename.endswith("_SIC.tiff"):
                    lookup[basename] = file_path
        return lookup

    @staticmethod
    def _build_samples(df, file_lookup):
        samples = []
        for _, row in df.iterrows():
            amsr2_name = str(row["amsr2_file"]).strip()
            sic_name = str(row["sic_file"]).strip()
            amsr2_path = file_lookup.get(amsr2_name)
            sic_path = file_lookup.get(sic_name)
            if not amsr2_path or not sic_path:
                continue

            if "timestamp" in row:
                date_str = str(row["timestamp"]).strip()
            else:
                date_str = ""

            samples.append((amsr2_path, sic_path, date_str))
        return samples

    @staticmethod
    def _squeeze_to_2d(array):
        arr = np.asarray(array)
        if arr.ndim == 2:
            return arr
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            return arr
        return None

    def _build_input_tensor(self, frequencies):
        channels = []
        for var_name in sorted(frequencies.keys()):
            arr_2d = self._squeeze_to_2d(frequencies[var_name])
            if arr_2d is None:
                continue

            arr_2d = np.nan_to_num(arr_2d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            vmin = float(arr_2d.min())
            vmax = float(arr_2d.max())
            if vmax > vmin:
                arr_2d = (arr_2d - vmin) / (vmax - vmin)
            channels.append(arr_2d)

            if len(channels) == 3:
                break

        if len(channels) < 3:
            return None

        x_np = np.stack(channels, axis=0)
        x_t = torch.from_numpy(x_np).unsqueeze(0)
        x_t = F.interpolate(
            x_t,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return x_t.squeeze(0)

    def _build_target_tensor(self, sic_band):
        y_2d = self._squeeze_to_2d(sic_band)
        if y_2d is None:
            return None

        y_2d = np.nan_to_num(y_2d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        y_2d = np.clip(y_2d, 0.0, 100.0) / 100.0

        y_t = torch.from_numpy(y_2d).unsqueeze(0).unsqueeze(0)
        y_t = F.interpolate(
            y_t,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return y_t.squeeze(0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        amsr2_path, sic_path, date_str = self.samples[index]
        pair = {"date": date_str, "amsr2_files": [amsr2_path], "sic_files": [sic_path]}
        loaded = _load_pair((pair, self.date_pattern))

        input_entry = loaded["input"][0]
        label_entry = loaded["label"][0]
        if input_entry.get("error") or label_entry.get("error"):
            return None

        x = self._build_input_tensor(input_entry["frequencies"])  # [3, H, W]
        y = self._build_target_tensor(label_entry["band"].get("band_1"))  # [1, H, W]
        if x is None or y is None:
            return None

        return x, y


def collate_skip_none(batch):
    valid = [item for item in batch if item is not None]
    if not valid:
        return None
    xs, ys = zip(*valid)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def create_train_val_dataloaders(
    training_index_csv=DEFAULT_TRAINING_INDEX_CSV,
    data_dirs=DEFAULT_DATA_DIRS,
    batch_size=4,
    val_split=0.1,
    image_size=128,
    num_workers=0,
    shuffle=True,
):
    dataset = SICTrainingDataset(
        training_index_csv=training_index_csv,
        data_dirs=data_dirs,
        image_size=image_size,
    )

    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_skip_none,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_skip_none,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, dataset

def _load_amsr2(file_path, date_pattern):
    match = re.search(date_pattern, os.path.basename(file_path))
    try:
        with xr.open_dataset(file_path) as ds:
            return {
                'time': match.group(0) if match else None,
                'frequencies': {var: ds[var].values for var in ds.data_vars},
                'num_swaths': len(ds.attrs.get('AMSR2_swaths', [])),
                'file_path': file_path,
                'type': 'amsr2',
                'error': None,
            }
    except OSError as e:
        return {
            'time': match.group(0) if match else None,
            'num_swaths': 999,
            'file_path': file_path,
            'type': 'amsr2',
            'error': str(e),
        }

def _load_sic(file_path, date_pattern):
    match = re.search(date_pattern, os.path.basename(file_path))
    try:
        with rioxarray.open_rasterio(file_path) as ds:
            ds = ds.rio.reproject("EPSG:3411", resampling=Resampling.bilinear)
            ref = ds['spatial_ref'].attrs
            band = {f'band_{i + 1}': ds[i].where(ds.values[i] != 254, np.nan).values for i in range(ds.rio.count)}
            epsg = ds.rio.crs.to_epsg()
            bbox = ds.rio.bounds()
        return {
            'time': match.group(0) if match else None,
            'band': band,
            'epsg': epsg,
            'bbox': bbox,
            'standard_parallel': ref.get('standard_parallel'),
            'standard_vertical_longitude_from_pole': ref.get('standard_vertical_longitude_from_pole'),
            'semi_major_axis': ref.get('semi_major_axis'),
            'semi_minor_axis': ref.get('semi_minor_axis'),
            'file_path': file_path,
            'type': 'sic',
            'error': None,
        }
    except Exception as e:
        return {
            'time': match.group(0) if match else None,
            'file_path': file_path,
            'type': 'sic',
            'error': str(e),
        }

def _load_pair(args: tuple) -> dict:
    pair, date_pattern = args
    inputs = [_load_amsr2(fp, date_pattern) for fp in pair['amsr2_files']]
    labels = [_load_sic(fp, date_pattern)   for fp in pair['sic_files']]
    return {'date': pair['date'], 'input': inputs, 'label': labels}

class SICDataLoader:
    def __init__(self, data_dirs, batch_size=32, shuffle=True, date_pattern=r'\d{8}[Tt]\d{6}', years=None, max_workers=8, prefetch_batches=2):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.date_pattern_str = date_pattern
        self.date_pattern = re.compile(date_pattern)
        self.max_workers = max_workers
        self.prefetch_batches = prefetch_batches

        if years is None:
            self._years = None
        elif isinstance(years, int):
            self._years = frozenset({str(years)})
        else:
            self._years = frozenset(str(y) for y in years)

        self.data_dirs = [data_dirs] if isinstance(data_dirs, str) else list(data_dirs)
        self.date_groups = defaultdict(lambda: {'amsr2': [], 'sic': []})

        self._scan_files()

        self.matched_pairs = [
            {'date': date_str, 'amsr2_files': files['amsr2'], 'sic_files': files['sic']}
            for date_str, files in self.date_groups.items()
            if files['amsr2'] and files['sic']
        ]

        self.num_samples = len(self.matched_pairs)
        self.indices = list(range(self.num_samples))
        if self.shuffle:
            random.shuffle(self.indices)

        self._current_index = 0
        self._prefetch_cache = {}

        self._pool = ProcessPoolExecutor(max_workers=self.max_workers)

    def _year_patterns(self, is_amsr2):
        suffix = '*/*/*.nc' if is_amsr2 else '*/*/*_SIC.tiff'
        if self._years is None:
            return [suffix]
        return [f'{y}/{suffix}' for y in self._years]

    def _scan_files(self):
        for data_dir in self.data_dirs:
            is_amsr2 = 'AMSR2' in data_dir
            for pattern in self._year_patterns(is_amsr2):
                for file_path in glob.iglob(os.path.join(data_dir, pattern)):
                    classified = self._classify_file(file_path)
                    if classified:
                        date_str, file_type, path = classified
                        self.date_groups[date_str][file_type].append(path)

    def _classify_file(self, file_path):
        match = self.date_pattern.search(os.path.basename(file_path))
        if not match:
            return None
        date_str = match.group(0)
        name = os.path.basename(file_path)
        if name.startswith('AMSR2'):
            return date_str, 'amsr2', file_path
        if name.startswith('S1'):
            return date_str, 'sic', file_path
        return None

    def _load_batch(self, batch_indices):
        pairs = [self.matched_pairs[i] for i in batch_indices]
        args = [(pair, self.date_pattern_str) for pair in pairs]
        return list(self._pool.map(_load_pair, args))

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        self._current_index = 0
        for val in self._prefetch_cache.values():
            if isinstance(val, Future):
                val.cancel()
        self._prefetch_cache.clear()
        if self.shuffle:
            random.shuffle(self.indices)
        self._submit_prefetch()
        return self

    def __next__(self):
        if self._current_index >= self.num_samples:
            raise StopIteration

        batch_num = self._current_index // self.batch_size
        batch_indices = self.indices[self._current_index: self._current_index + self.batch_size]
        self._current_index += self.batch_size

        cached = self._prefetch_cache.pop(batch_num, None)
        batch_data = (cached.result() if isinstance(cached, Future) else cached) \
                        if cached is not None else self._load_batch(batch_indices)

        self._submit_prefetch()
        return batch_data

    def _submit_prefetch(self):
        for offset in range(1, self.prefetch_batches + 1):
            future_batch_num = (self._current_index // self.batch_size) + offset - 1
            start = self._current_index + (offset - 1) * self.batch_size
            if start >= self.num_samples or future_batch_num in self._prefetch_cache:
                continue
            indices_slice = self.indices[start: start + self.batch_size]
            pairs = [self.matched_pairs[i] for i in indices_slice]
            args = [(pair, self.date_pattern_str) for pair in pairs]
            self._prefetch_cache[future_batch_num] = self._pool.submit(_load_pair, args[0])

    def shutdown(self):
        self._pool.shutdown(wait=False)

    def __del__(self):
        try:
            self._pool.shutdown(wait=False)
        except Exception:
            pass

    def get_date_groups(self):
        return dict(self.date_groups)

    def get_matched_pairs_info(self):
        return [
            {
                'date': p['date'],
                'num_amsr2_files': len(p['amsr2_files']),
                'num_sic_files': len(p['sic_files']),
                'amsr2_files': p['amsr2_files'],
                'sic_files': p['sic_files'],
            }
            for p in self.matched_pairs
        ]

    def summary(self):
        years_label = ', '.join(sorted(self._years)) if self._years else 'all'
        print(f"SICDataLoader summary")
        print(f"  Years          : {years_label}")
        print(f"  Directories    : {len(self.data_dirs)}")
        print(f"  Matched pairs  : {self.num_samples}")
        print(f"  Batches        : {len(self)}")
        print(f"  Batch size     : {self.batch_size}")
        print(f"  Workers        : {self.max_workers}")
        print(f"  Shuffle        : {self.shuffle}")

def repair_all_yearly_csvs(csv_root, data_dirs, year_range=range(2014, 2025), date_pattern=r'(\d{8})[T]\d{6}'):
    """
    Check and repair num_swaths == 999 errors in all yearly CSV files (in-place).
    
    Args:
        csv_root: Root directory containing yearly CSV files (e.g., /path/to/Master-thesis-...).
        data_dirs: List of data directories (AMSR2, SIC paths) for repair operations.
        year_range: Range of years to process (default 2014-2024).
        date_pattern: Regex pattern for extracting dates from filenames.
    
    Returns:
        Dictionary with repair summary per year.
    """
    from pathlib import Path
    
    csv_root = Path(csv_root)
    
    print('\n=== Repairing Yearly CSVs ===\n')
    
    repair_summary = {}
    total_errors_found = 0
    total_repaired = 0
    
    for year in year_range:
        csv_path = csv_root / f'sic_amsr2_metadata_stats_{year}.csv'
        
        if not csv_path.exists():
            print(f'Year {year}: File does not exist')
            repair_summary[year] = {'exists': False, 'repaired': 0, 'remaining': 0, 'total': 0}
            continue
        
        # Load and clean column names
        df_year = pd.read_csv(csv_path, low_memory=False)
        df_year.columns = df_year.columns.str.strip()
        
        if 'num_swaths' not in df_year.columns:
            print(f'Year {year}: No num_swaths column')
            repair_summary[year] = {'exists': True, 'repaired': 0, 'remaining': 0, 'total': len(df_year)}
            continue
        
        # Check for errors
        bad_count = int((df_year['num_swaths'] == 999).sum())
        total_errors_found += bad_count
        
        if bad_count > 0:
            print(f'Year {year}: Found {bad_count} rows with num_swaths == 999')
            try:
                repaired, remaining, total = repair_csv_errors(str(csv_path), data_dirs, date_pattern)
                print(f'  → Repaired {repaired}, {remaining} still bad, {total} total rows')
                repair_summary[year] = {'exists': True, 'repaired': repaired, 'remaining': remaining, 'total': total}
                total_repaired += repaired
            except Exception as e:
                print(f'  → Error during repair: {e}')
                repair_summary[year] = {'exists': True, 'repaired': 0, 'remaining': bad_count, 'total': len(df_year)}
        else:
            print(f'Year {year}: ✓ Clean ({len(df_year)} rows, no errors)')
            repair_summary[year] = {'exists': True, 'repaired': 0, 'remaining': 0, 'total': len(df_year)}
    
    print(f'\n=== Summary ===')
    print(f'Total errors found: {total_errors_found}')
    print(f'Total repaired: {total_repaired}')
    
    return repair_summary

BINS = [0, 0.001, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99.999, 100]
BIN_LABELS = ['val_0', '0-10', '10-20', '20-30', '30-40', '40-50',
              '50-60', '60-70', '70-80', '80-90', '90-100', 'val_100']
 
 
def _extract_statistics_worker(args):
    import time
    matched_pair, date_pattern = args
    RETRY_COUNT = 3
    RETRY_SLEEP_SEC = 0.4
 
    sic_entries = [
        (os.path.basename(p), _load_sic(p, date_pattern))
        for p in matched_pair['sic_files']
    ]
    if not sic_entries:
        return []
 
    sic_basename, sic_info = sic_entries[0]

    if sic_info['error']:
        # If SIC load failed, return error records for all AMSR2 files with complete schema.
        return [{
            'timestamp':  matched_pair['date'],
            'num_swaths': 999,
            'bbox':       None,
            **{label: None for label in BIN_LABELS},
            'sic_file':   sic_basename,
            'amsr2_file': os.path.basename(p),
            'error':      sic_info['error'],
        } for p in matched_pair['amsr2_files']]

    valid_data = sic_info['band']['band_1']
    valid_data = valid_data[~np.isnan(valid_data)]
    counts, _ = np.histogram(valid_data, bins=BINS)
    bin_dict = dict(zip(BIN_LABELS, counts.tolist()))
 
    records = []
    for amsr2_path in matched_pair['amsr2_files']:
        # Retry AMSR2 loads up to 3 times to recover from transient IO failures (num_swaths=999).
        amsr2_info = None
        for attempt in range(RETRY_COUNT):
            amsr2_info = _load_amsr2(amsr2_path, date_pattern)
            if amsr2_info.get('num_swaths') != 999:
                break
            if attempt < RETRY_COUNT - 1:
                time.sleep(RETRY_SLEEP_SEC)
        
        records.append({
            'timestamp':  matched_pair['date'],
            'num_swaths': amsr2_info.get('num_swaths'),
            'bbox':       sic_info['bbox'],
            **bin_dict,
            'amsr2_file': os.path.basename(amsr2_path),
            'sic_file':   sic_basename,
            'error': amsr2_info.get('error'),
        })
    return records

def repair_csv_errors(csv_path, data_dirs, date_pattern=r'(\d{8})[T]\d{6}'):
    """
    Repair rows with num_swaths == 999 in a CSV file by recomputing them.
    
    Args:
        csv_path: Path to the yearly CSV file to repair.
        data_dirs: List of data directories (AMSR2, SIC paths).
        date_pattern: Regex pattern to extract dates from filenames.
    
    Returns:
        Tuple (repaired_count, remaining_bad_count, total_rows)
    """
    import os
    from pathlib import Path
    
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f'CSV file not found: {csv_path}')
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # Clean up whitespace in column names
    
    if 'num_swaths' not in df.columns:
        raise ValueError(f'Column "num_swaths" not found in {csv_path}')
    
    bad_mask = df['num_swaths'].eq(999)
    bad_count = int(bad_mask.sum())
    
    if bad_count == 0:
        print(f'[repair] No rows with num_swaths == 999 in {csv_path.name}')
        return 0, 0, len(df)
    
    print(f'[repair] Found {bad_count} error rows in {csv_path.name}; recomputing...')
    
    # Extract year from CSV filename (e.g., 'sic_amsr2_metadata_stats_2015.csv' -> 2015)
    try:
        year = int(str(csv_path.stem).split('_')[-1])
    except (ValueError, IndexError):
        raise ValueError(f'Could not extract year from CSV filename: {csv_path.name}')
    
    # Load matched pairs for this year
    sic_loader = SICDataLoader(
        data_dirs=data_dirs,
        shuffle=False,
        date_pattern=date_pattern,
        years=[year],
    )
    matched_pairs = sic_loader.get_matched_pairs_info()
    pairs_by_date = {p['date']: p for p in matched_pairs}
    
    # Build AMSR2 filename -> pair lookup
    pair_by_amsr2_file = {}
    for pair in matched_pairs:
        for amsr2_path in pair['amsr2_files']:
            pair_by_amsr2_file[os.path.basename(amsr2_path)] = pair
    
    # Cache to avoid recomputing same pair multiple times
    pair_records_cache = {}
    repaired_rows = 0
    stat_columns = [
        'num_swaths', 'bbox', 'error',
        'val_0', '0-10', '10-20', '20-30', '30-40', '40-50',
        '50-60', '60-70', '70-80', '80-90', '90-100', 'val_100',
    ]
    
    for idx in df.index[bad_mask]:
        row = df.loc[idx]
        amsr2_file = os.path.basename(str(row.get('amsr2_file', '')).strip())
        ts = str(row.get('timestamp', '')).strip()
        
        # Find matching pair
        pair = None
        if amsr2_file and amsr2_file in pair_by_amsr2_file:
            pair = pair_by_amsr2_file[amsr2_file]
        elif ts in pairs_by_date:
            pair = pairs_by_date[ts]
        elif ts and len(ts) == 8:
            # Fallback: try YYYYMMDD match
            candidates = [p for d, p in pairs_by_date.items() if d.startswith(ts)]
            if len(candidates) == 1:
                pair = candidates[0]
        
        if pair is None:
            continue
        
        # Recompute this pair's records
        pair_key = pair['date']
        if pair_key not in pair_records_cache:
            pair_records_cache[pair_key] = _extract_statistics_worker((pair, date_pattern))
        
        records = pair_records_cache[pair_key]
        
        # Find the matching record
        rec = None
        for r in records:
            if r.get('amsr2_file') == amsr2_file:
                rec = r
                break
        
        if rec is None:
            continue
        
        # Update the bad row with repaired data
        for col in stat_columns:
            if col in rec:
                df.at[idx, col] = rec[col]
        repaired_rows += 1
    
    sic_loader.shutdown()
    
    # Save the repaired CSV
    df.to_csv(csv_path, index=False)
    remaining_bad = int(df['num_swaths'].eq(999).sum())
    
    print(f'[repair] Repaired {repaired_rows} rows; remaining num_swaths==999: {remaining_bad}')
    return repaired_rows, remaining_bad, len(df)

def build_yearly_stats(years, data_dirs, output_prefix = 'sic_amsr2_metadata_stats', max_workers = 8,):
    """Process all years; writes per-year CSVs and returns a combined DataFrame."""
    all_data = []
    date_pattern = r'(\d{8})[T]\d{6}'
 
    for year in years:
        print(f'\nWorking on year {year}')
 
        sic_loader = SICDataLoader(
            data_dirs=data_dirs,
            shuffle=False,
            date_pattern=date_pattern,
            years=[year],
            max_workers=max_workers,
        )
 
        matched_pairs = sic_loader.get_matched_pairs_info()
        print(f'  Matched pairs found: {len(matched_pairs)}')
 
        year_results = []
        args = [(pair, date_pattern) for pair in matched_pairs]
 
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_extract_statistics_worker, a): a for a in args}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f'  {year}', unit='pair'):
                try:
                    result = future.result()
                    if result:
                        year_results.extend(result)
                except Exception as exc:
                    print(f'  Warning: pair failed — {exc}')
 
        sic_loader.shutdown()
 
        year_df  = pd.DataFrame(year_results)
        csv_path = f'{output_prefix}_{year}.csv'
        year_df.to_csv(csv_path, index=False)
        print(f'  Saved {len(year_df)} records → {csv_path}')
    #     all_data.extend(year_results)
 
    # combined_df = pd.DataFrame(all_data)
    # print(f'\nTotal records across all years: {len(combined_df)}')
    return combined_df

# if __name__ == '__main__':
#     DATA_DIRS = [
#         '/dmidata/projects/asip-cms/tests/new_input_ncs/AMSR2',
#         '/dmidata/projects/asip-cms/reproc',
#     ]
#     YEARS = [2018, 2019, 2023, 2024]
 
#     df = build_yearly_stats(years=YEARS, data_dirs=DATA_DIRS)
#     # df.to_csv('sic_amsr2_metadata_stats_all_years.csv', index=False)
#     print('Done.')

# df_good    = df[df['val_0'] != 999]
# df_corrupt = df[df['val_0'] == 999]

if __name__ == '__main__':
    from pathlib import Path
    
    DATA_DIRS = [
        '/dmidata/projects/asip-cms/tests/new_input_ncs/AMSR2',
        '/dmidata/projects/asip-cms/reproc',
    ]
    CSV_ROOT = Path('/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI')
    
    # Call the function
    summary = repair_all_yearly_csvs(
        csv_root=CSV_ROOT,
        data_dirs=DATA_DIRS,
        year_range=range(2014, 2025)
    )
    
    # View the repair results
    for year, stats in summary.items():
        print(f'{year}: repaired={stats["repaired"]}, remaining={stats["remaining"]}, total={stats["total"]}')