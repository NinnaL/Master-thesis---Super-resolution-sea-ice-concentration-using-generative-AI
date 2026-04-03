import os
import torch
import torch.nn.functional as F
import numpy as np
import xarray as xr
import pandas as pd
import rioxarray
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from rasterio.enums import Resampling

DATA_DIRS = ['/dmidata/projects/asip-cms/tests/new_input_ncs/AMSR2', '/dmidata/projects/asip-cms/reproc']
TRAINING_INDEX_CSV = "/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI/training_index2.csv"

TARGET_SIZE = (192, 192)

class AMSR2Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dirs, csv_path, split=None, split_ratio=(0.9, 0.1), seed = 42, shuffle=True, transform=None, target_size=TARGET_SIZE):
        self.amsr2_dir = data_dirs[0]
        self.sic_dir = data_dirs[1]
        self.csv_path = csv_path
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.shuffle = shuffle
        self.transform = transform
        self.target_size = target_size

        # Names of the columns in the CSV file
        self.amsr_col = 'amsr2_file'
        self.label_col = 'sic_file'

        # Load CSV file and build file list
        df = pd.read_csv(self.csv_path)
        time = df['timestamp']
        year, month, day = time.str.slice(0, 4), time.str.slice(5, 7), time.str.slice(8, 10)

        all_files = [
            [os.path.join(self.amsr2_dir, y, m, d, amsr2),
             os.path.join(self.sic_dir, y, m, d, sic)]
            for y, m, d, amsr2, sic in zip(year, month, day, df[self.amsr_col], df[self.label_col])
        ]

        if split is None:
            self.file_list = all_files
        else:
            train_files, val_files = train_test_split(all_files, test_size=split_ratio[1], random_state=seed, shuffle=shuffle)
            self.file_list = train_files if split == 'train' else val_files
    
    def __len__(self):
        return len(self.file_list)

    def _resize(self, tensor, mode='bilinear'):
        """Resize a (C, H, W) tensor to self.target_size using bilinear interpolation if necessary."""
        if self.target_size is None or tuple(tensor.shape[-2:]) == self.target_size:
            return tensor
        return F.interpolate(
            tensor.unsqueeze(0), # (1, C, H, W) 
            size=self.target_size, 
            mode=mode, 
            align_corners=False if mode in ['bilinear'] else None
        ).squeeze(0)             # (C, H, W)

    def __getitem__(self, idx):
        amsr2_file, sic_file = self.file_list[idx]
        
        # Load AMSR2 frequencies (excluding the last one, which is )
        with xr.open_dataset(amsr2_file) as ds:
            # data = {var: ds[var].values for var in ds.data_vars}
            data = ds.to_array().values.astype(np.float32) 
        amsr2_tensor =self._resize(torch.from_numpy(data[:-1]))

        # Load SIC label, and mask out no-data values (254 and 255)
        with rioxarray.open_rasterio(sic_file) as da:
            da = da.rio.reproject('EPSG:3411', resampling=Resampling.bilinear)
            data = da.values.astype(np.float32) # (2, H, W) two bands
        
        invalid_mask = (data >= 253.5).any(axis=0, keepdims=True)  # True for no-data values (254 and 255) (1, H, W)
        data = np.where(invalid_mask, np.nan, data[[0]])  # (1, H, W)
        sic_tensor = self._resize(torch.from_numpy(data))

        mask_tensor = self._resize(torch.from_numpy(invalid_mask.astype(np.float32)), mode='nearest').bool()

        if self.transform:
            amsr2_tensor = self.transform(amsr2_tensor)

        return amsr2_tensor, sic_tensor, mask_tensor 



# if __name__ == "__main__":
#     train_dataset = AMSR2Dataset(DATA_DIRS, TRAINING_INDEX_CSV, split="train")
#     val_dataset   = AMSR2Dataset(DATA_DIRS, TRAINING_INDEX_CSV, split="val")
 
#     train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=0)
#     val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, num_workers=0)
 
#     amsr2_batch, label_batch, mask_batch = next(iter(train_loader))
#     print(f"AMSR2 batch shape : {amsr2_batch.shape}")  # (B, n_freq-1, H, W)
#     print(f"Label batch shape : {label_batch.shape}")  # (B, 1, H, W)
#     print(f"Mask batch shape  : {mask_batch.shape}")    # (B, 1, H, W)
#     print(f"Invalid pixels    : {mask_batch.float().mean():.1%}")
