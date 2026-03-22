import os
import glob
import re
import numpy as np
from collections import defaultdict
from rasterio.enums import Resampling
import xarray as xr
import rioxarray

class SICDataLoader:
    def __init__(self, data_dirs, batch_size=32, shuffle=True, date_pattern=r'\d{8}[Tt]\d{6}', year=None):
        """
        DataLoader that matches AMSR2 multi-frequency files with S1B ground truth files based on date.
        
        Args:
            data_dirs: Single directory or list of directories containing .nc files.
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle the data.
            date_pattern: Regex pattern to extract date from filename (default: YYYYMMDD[T]HHMMSS).
            year: Optional year filter (e.g., 2020). If set, only that year is loaded.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.date_pattern = date_pattern
        self.year = str(year) if year is not None else '*'
        
        self.data_dirs = [data_dirs] if isinstance(data_dirs, str) else data_dirs
        self.date_groups = defaultdict(lambda: {'amsr2': [], 'sic': []})
        
        self._load_files()
        
        self.matched_pairs = [
            {'date': date_str, 'amsr2_files': files['amsr2'], 'sic_files': files['sic']}
            for date_str, files in self.date_groups.items() if files['amsr2'] and files['sic']
        ]
        
        self.num_samples = len(self.matched_pairs)
        self.indices = list(range(self.num_samples))
        if self.shuffle:
            self._shuffle_indices()

    def _load_files(self):
        """Load files from specified directories."""
        for data_dir in self.data_dirs:
            file_paths = glob.glob(os.path.join(data_dir, f'{self.year}/*/*.nc' if 'AMSR2' in data_dir else f'{self.year}/*/*_SIC.tiff'))
            for file_path in file_paths:
                date_str = self._extract_date(file_path)
                if date_str and self._matches_year(date_str):
                    file_type = self._get_file_type(file_path)
                    if file_type:
                        self.date_groups[date_str][file_type].append(file_path)

    def _extract_date(self, file_path):
        """Extract date (YYYYMMDDTHHMMSS) from the first occurrence in filename."""
        match = re.search(self.date_pattern, os.path.basename(file_path))
        return match.group(0) if match else None

    def _matches_year(self, date_str):
        """Check if the date matches the specified year filter."""
        return self.year == '*' or date_str.startswith(self.year)

    def _get_file_type(self, file_path):
        """Determine if file is AMSR2 or S1B based on filename."""
        filename = os.path.basename(file_path)
        if filename.startswith('AMSR2'):
            return 'amsr2'
        elif filename.startswith(('S1B', 'S1A')):
            return 'sic'
        return None
    
    def _load_amsr2(self, file_path):
        """Load AMSR2 file with multiple frequencies."""
        ds = xr.open_dataset(file_path)
        return {
            'time': self._extract_date(file_path),
            'frequencies': {var_name: ds[var_name].values for var_name in ds.data_vars},
            'num_swaths': len(ds.attrs.get('AMSR2_swaths')),
            'dataset': ds,
            'file_path': file_path,
            'type': 'amsr2'
        }
    
    def _load_sic(self, file_path):
        """Load SIC file (.tiff)."""
        ds = rioxarray.open_rasterio(file_path).rio.reproject("EPSG:3411", resampling=Resampling.bilinear)
        
        # Get EPSG and bounding box
        epsg = ds.rio.crs.to_epsg()
        minx, miny, maxx, maxy = ds.rio.bounds()
        
        # Get spatial reference attributes
        standard_parallel = ds['spatial_ref'].attrs.get('standard_parallel')
        standard_vertical = ds['spatial_ref'].attrs.get('standard_vertical_longitude_from_pole')
        semi_major_axis = ds['spatial_ref'].attrs.get('semi_major_axis')
        semi_minor_axis = ds['spatial_ref'].attrs.get('semi_minor_axis')
        
        # Extract SIC bands
        band = {f'band_{i+1}': ds[i].where(ds.values[i] != 254, np.nan).values[i] for i in range(ds.rio.count)}
        
        return {
            'time': self._extract_date(file_path),
            'band': band,
            'epsg': epsg,
            'bbox': (minx, miny, maxx, maxy),
            'standard_parallel': standard_parallel,
            'standard_vertical_longitude_from_pole': standard_vertical,
            'semi_major_axis': semi_major_axis,
            'semi_minor_axis': semi_minor_axis,
            'dataset': ds,
            'file_path': file_path,
            'type': 'sic'
        }

    def _shuffle_indices(self):
        """Shuffle the indices."""
        import random
        random.shuffle(self.indices)

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        self.current_index = 0
        if self.shuffle:
            self._shuffle_indices()
        return self

    def _get_item_(self):
        """Load the next batch of data suitable for training."""
        if self.current_index >= self.num_samples:
            raise StopIteration

        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        batch_data = [
            {
                'date': pair['date'],
                'input': [self._load_amsr2(amsr2_file) for amsr2_file in pair['amsr2_files']],
                'label': [self._load_sic(sic_file) for sic_file in pair['sic_files']]
            }
            for idx in batch_indices
            for pair in [self.matched_pairs[idx]]
        ]

        self.current_index += self.batch_size
        return batch_data

    def get_date_groups(self):
        """Return dictionary of dates and their associated files."""
        return dict(self.date_groups)

    def get_matched_pairs_info(self):
        """Return information about matched pairs."""
        return [
            {'date': pair['date'], 'num_amsr2_files': len(pair['amsr2_files']), 'num_sic_files': len(pair['sic_files']), 'amsr2_files': pair['amsr2_files'], 'sic_files': pair['sic_files']}
            for pair in self.matched_pairs
        ]