import os
import glob
import xarray as xr
import rioxarray
import re
from datetime import datetime
from collections import defaultdict

class SICDataLoader:
    def __init__(self, data_dirs, batch_size=32, shuffle=True, date_pattern=r'\d{8}[Tt]\d{6}', year=None):
        """
        DataLoader that matches AMSR2 multi-frequency files with S1B ground truth files based on date.
        
        Files are matched by date (YYYYMMDD) extracted from filenames:
        - AMSR2_S1A_EW_GRDM_1SDH_20200202t032611_.... (input with multiple frequencies)
        - S1B_EW_GRDM_1SDH_20200202t185117_.... (ground truth)
        
        Args:
            data_dirs: Single directory or list of directories containing .nc files
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            date_pattern: Regex pattern to extract date from filename (default: YYYYMMDD[T]HHMMSS)
            year: Optional year filter (e.g., 2020). If set, only that year is loaded.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.date_pattern = date_pattern
        self.year = str(year) if year is not None else '*'  # Match all years if no filter is set
        
        # Handle single directory or multiple directories
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        
        # Group files by date
        self.date_groups = defaultdict(lambda: {'amsr2': [], 'sic': []})
        
        for data_dir in data_dirs:
            if 'AMSR2' in data_dir:
                file_paths = glob.glob(os.path.join(data_dir, f'{self.year}/**/', '*.nc'), recursive=True)
            else:
                file_paths = glob.glob(os.path.join(data_dir, f'{self.year}/**/', '*_SIC.tiff'), recursive=True)
            for file_path in file_paths:
                date_str = self._extract_date(file_path)
                file_type = self._get_file_type(file_path)
                
                if date_str and file_type and self._matches_year(date_str):
                    self.date_groups[date_str][file_type].append(file_path)
        
        # Create list of matched pairs (only dates with both AMSR2 and S1B)
        self.matched_pairs = []
        for date_str, files in self.date_groups.items():
            if files['amsr2'] and files['sic']:
                self.matched_pairs.append({
                    'date': date_str,
                    'amsr2_files': files['amsr2'],
                    'sic_files': files['sic']
                })
        
        self.num_samples = len(self.matched_pairs)
        self.indices = list(range(self.num_samples))
        
        if self.shuffle:
            self._shuffle_indices()
    
    def _extract_date(self, file_path):
        """Extract date (YYYYMMDDTHHMMSS) from filename."""
        filename = os.path.basename(file_path)
        match = re.search(self.date_pattern, filename)
        if match:
            return match.group(0)
        return None

    def _matches_year(self, date_str):
        """Return True if sample belongs to selected year or no year filter is set."""
        if self.year == '*':
            return True
        if len(date_str) < 4:
            return False
        return date_str[:4] == self.year
    
    def _get_file_type(self, file_path):
        """Determine if file is AMSR2 or S1B based on filename."""
        filename = os.path.basename(file_path)
        if filename.startswith('AMSR2') :
            return 'amsr2'
        elif filename.startswith('S1B') or filename.startswith('S1A'):
            return 'sic'
        return None
    
    def _load_data(self, file_path):
        """Load lat, lon, and data from .nc file."""
        ds = xr.open_dataset(file_path)
        
        # Extract lat, lon - common variable names
        lat = None
        lon = None
        
        for lat_name in ['lat', 'latitude', 'y', 'lat_bnds']:
            if lat_name in ds.variables:
                lat = ds[lat_name].values
                break
        
        for lon_name in ['lon', 'longitude', 'x', 'lon_bnds']:
            if lon_name in ds.variables:
                lon = ds[lon_name].values
                break
        
        # Get data variables (exclude coordinates)
        data_vars = {}
        for var_name in ds.data_vars:
            data_vars[var_name] = ds[var_name].values
        
        return {
            'lat': lat,
            'lon': lon,
            'data': data_vars,
            'dataset': ds,
            'file_path': file_path
        }
    
    def _load_amsr2_frequencies(self, file_path):
        """Load AMSR2 file with multiple frequencies."""
        ds = xr.open_dataset(file_path)
        print('AMSR2')
        print(ds)
        
        # Extract coordinates
        lat = None
        lon = None
        
        for lat_name in ['lat', 'latitude', 'y', 'lat_bnds']:
            if lat_name in ds.variables:
                lat = ds[lat_name].values
                break
        
        for lon_name in ['lon', 'longitude', 'x', 'lon_bnds']:
            if lon_name in ds.variables:
                lon = ds[lon_name].values
                break
        
        # Extract all frequency channels
        frequencies = {}
        for var_name in ds.data_vars:
            frequencies[var_name] = ds[var_name].values
        
        return {
            'lat': lat,
            'lon': lon,
            'frequencies': frequencies,
            'dataset': ds,
            'file_path': file_path,
            'type': 'amsr2'
        }
    
    def _load_sic_ground_truth(self, file_path):
        """Load SIC ground truth file (.tiff or .nc)."""
        # Check if file is a TIFF file
        if file_path.lower().endswith(('.tiff', '.tif')):
            ds = rioxarray.open_rasterio(file_path)
            print('SIC')
            print(ds)
            
            # Get coordinates from raster
            lat = ds.y.values if 'y' in ds.coords else None
            lon = ds.x.values if 'x' in ds.coords else None
            
            # Extract ground truth data (all bands)
            ground_truth = {}
            if len(ds.shape) == 3:  # (band, y, x)
                for i in range(ds.shape[0]):
                    ground_truth[f'band_{i+1}'] = ds.values[i, :, :]
            else:  # (y, x)
                ground_truth['data'] = ds.values
        
        return {
            'lat': lat,
            'lon': lon,
            'ground_truth': ground_truth,
            'dataset': ds,
            'file_path': file_path,
            'type': 'sic'
        }

    def _shuffle_indices(self):
        import random
        random.shuffle(self.indices)

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        self.current_index = 0
        if self.shuffle:
            self._shuffle_indices()
        return self

    def __next__(self):
        if self.current_index >= self.num_samples:
            raise StopIteration
        
        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        batch_data = []
        
        for idx in batch_indices:
            pair = self.matched_pairs[idx]
            
            # Load all AMSR2 files (multiple frequencies)
            amsr2_data = []
            for amsr2_file in pair['amsr2_files']:
                loaded = self._load_amsr2_frequencies(amsr2_file)
                amsr2_data.append(loaded)
            
            # Load all SIC files (ground truth)
            sic_data = []
            for sic_file in pair['sic_files']:
                loaded = self._load_sic_ground_truth(sic_file)
                sic_data.append(loaded)
            
            batch_data.append({
                'date': pair['date'],
                'input': amsr2_data,
                'ground_truth': sic_data
            })
        
        self.current_index += self.batch_size
        return batch_data
    
    def get_date_groups(self):
        """Return dictionary of dates and their associated files."""
        return dict(self.date_groups)
    
    def get_matched_pairs_info(self):
        """Return information about matched pairs."""
        info = []
        for pair in self.matched_pairs:
            info.append({
                'date': pair['date'],
                'num_amsr2_files': len(pair['amsr2_files']),
                'num_sic_files': len(pair['sic_files']),
                'amsr2_files': pair['amsr2_files'],
                'sic_files': pair['sic_files']
            })
        return info