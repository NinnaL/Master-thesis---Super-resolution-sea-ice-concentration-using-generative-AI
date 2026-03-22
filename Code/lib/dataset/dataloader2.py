import os
import glob
import re
import random
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from rasterio.enums import Resampling
import xarray as xr
import rioxarray


class SICDataLoader:
    def __init__(self, data_dirs, batch_size=32, shuffle=True, date_pattern=r'\d{8}[Tt]\d{6}', years=None, max_workers=8, prefetch_batches=2):
        self.batch_size = batch_size
        self.shuffle = shuffle
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

        self._load_files_parallel()

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

    def _year_patterns(self, is_amsr2):
        suffix = '*/*/*.nc' if is_amsr2 else '*/*/*_SIC.tiff'
        if self._years is None:
            return [f'*/{suffix}']
        return [f'{y}/{suffix}' for y in self._years]

    def _glob_directory(self, data_dir):
        is_amsr2 = 'AMSR2' in data_dir
        paths = []
        for pattern in self._year_patterns(is_amsr2):
            paths.extend(glob.glob(os.path.join(data_dir, pattern)))
        return paths

    def _load_files_parallel(self):
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(self.data_dirs) or 1)) as ex:
            futures = {ex.submit(self._glob_directory, d): d for d in self.data_dirs}
            all_paths = []
            for future in as_completed(futures):
                all_paths.extend(future.result())

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(self._classify_file, p): p for p in all_paths}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    date_str, file_type, file_path = result
                    self.date_groups[date_str][file_type].append(file_path)

    def _classify_file(self, file_path):
        date_str = self._extract_date(file_path)
        if not date_str:
            return None
        file_type = self._get_file_type(file_path)
        if not file_type:
            return None
        return date_str, file_type, file_path

    def _extract_date(self, file_path):
        match = self.date_pattern.search(os.path.basename(file_path))
        return match.group(0) if match else None

    def _get_file_type(self, file_path):
        name = os.path.basename(file_path)
        if name.startswith('AMSR2'):
            return 'amsr2'
        if name.startswith('S1'):
            return 'sic'
        return None

    def _load_amsr2(self, file_path):
        ds = xr.open_dataset(file_path)
        return {
            'time': self._extract_date(file_path),
            'frequencies': {var: ds[var].values for var in ds.data_vars},
            'num_swaths': len(ds.attrs.get('AMSR2_swaths', [])),
            'dataset': ds,
            'file_path': file_path,
            'type': 'amsr2',
        }

    def _load_sic(self, file_path):
        ds = rioxarray.open_rasterio(file_path).rio.reproject("EPSG:3411", resampling=Resampling.bilinear)
        ref = ds['spatial_ref'].attrs
        band = {
            f'band_{i + 1}': ds[i].where(ds.values[i] != 254, np.nan).values
            for i in range(ds.rio.count)
        }
        return {
            'time': self._extract_date(file_path),
            'band': band,
            'epsg': ds.rio.crs.to_epsg(),
            'bbox': ds.rio.bounds(),
            'standard_parallel': ref.get('standard_parallel'),
            'standard_vertical_longitude_from_pole': ref.get('standard_vertical_longitude_from_pole'),
            'semi_major_axis': ref.get('semi_major_axis'),
            'semi_minor_axis': ref.get('semi_minor_axis'),
            'dataset': ds,
            'file_path': file_path,
            'type': 'sic',
        }

    def _load_pair(self, pair):
        all_files = [('amsr2', f) for f in pair['amsr2_files']] + [('sic', f) for f in pair['sic_files']]
        inputs, labels = [], []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            future_map = {
                ex.submit(self._load_amsr2 if ftype == 'amsr2' else self._load_sic, fp): ftype
                for ftype, fp in all_files
            }
            for future in as_completed(future_map):
                ftype = future_map[future]
                data = future.result()
                (inputs if ftype == 'amsr2' else labels).append(data)
        return {'date': pair['date'], 'input': inputs, 'label': labels}

    def _load_batch(self, batch_indices):
        pairs = [self.matched_pairs[i] for i in batch_indices]
        batch_data = [None] * len(pairs)
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            future_map = {ex.submit(self._load_pair, pair): idx for idx, pair in enumerate(pairs)}
            for future in as_completed(future_map):
                batch_data[future_map[future]] = future.result()
        return batch_data

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        self._current_index = 0
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

        if batch_num in self._prefetch_cache:
            batch_data = self._prefetch_cache.pop(batch_num)
        else:
            batch_data = self._load_batch(batch_indices)

        self._submit_prefetch()
        return batch_data

    def _submit_prefetch(self):
        for offset in range(1, self.prefetch_batches + 1):
            future_batch_num = (self._current_index // self.batch_size) + offset - 1
            start = self._current_index + (offset - 1) * self.batch_size
            if start >= self.num_samples or future_batch_num in self._prefetch_cache:
                continue
            indices_slice = self.indices[start: start + self.batch_size]
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(self._load_batch, indices_slice)
                self._prefetch_cache[future_batch_num] = future.result()

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


sic_loader = SICDataLoader(
    data_dirs=['/dmidata/projects/asip-cms/tests/new_input_ncs/AMSR2','/dmidata/projects/asip-cms/reproc'],
    shuffle=False,
    date_pattern=r'(\d{8})[T]\d{6}',
    years = [2014, 2015, 2016, 2017, 2018, 2019, 2023, 2024]
)

BINS = [0, 0.001, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99.999, 100]
BIN_LABELS = ['val_0', '0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100', 'val_100']

def extract_matched_statistics(matched_pair, loader=sic_loader):
    date_str = matched_pair['date']
    records = []

    # Load SIC-file
    sic_data = [(os.path.basename(p), loader._load_sic(p)) for p in matched_pair['sic_files']]

    for sic_basename, sic_info in sic_data:
        band1 = sic_info['band']['band_1']
        valid_data = band1[~np.isnan(band1)]
        counts, _ = np.histogram(valid_data, bins=BINS)

    records = []
    for amsr2_path in matched_pair['amsr2_files']:
        amsr2_info = loader._load_amsr2(amsr2_path)
        amsr2_basename = os.path.basename(amsr2_path)
        num_swaths = amsr2_info['num_swaths']

        records.append({
            'timestamp':  date_str,
            'num_swaths': num_swaths,
            'bbox':       sic_info['bbox'],
            **dict(zip(BIN_LABELS, counts.tolist())),
            'amsr2_file': amsr2_basename,
            'sic_file':   sic_basename
        })

    return records

# Make list of all matched pairs for future sampling

matched_pairs = sic_loader.get_matched_pairs_info()
final_results = []

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = executor.map(extract_matched_statistics, matched_pairs, chunksize=100)
    
    # Progress bar
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Pairs"):
        result = future.result()
        if result:
            final_results.extend(result)

# Convert to DataFrame in order to save as CSV
df = pd.DataFrame(final_results)
print(f"Processed {len(df)} total records.")
df.to_csv("sic_amsr2_metadata_stats2.csv", index=False)
print("Saved statistics to sic_amsr2_metadata_stats2.csv")