import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import xarray as xr
import pandas as pd
import rioxarray
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

DATA_DIRS = ['/dmidata/projects/asip-cms/tests/new_input_ncs/AMSR2', '/dmidata/projects/asip-cms/reproc']
TRAINING_INDEX_CSV = "/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI/training_index3.csv"

SCALE_FACTOR = 25  # SIC is 25x higher resolution than AMSR2

class AMSR2Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dirs, csv_path, split=None, split_ratio=(0.9, 0.1), seed = 42, shuffle=True, transform=None):
        self.amsr2_dir = data_dirs[0]
        self.sic_dir = data_dirs[1]
        self.csv_path = csv_path
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.shuffle = shuffle
        self.transform = transform

        # Names of the columns in the CSV file
        self.amsr_col = 'amsr2_file'
        self.label_col = 'sic_file'

        # Load CSV file and build file list
        df = pd.read_csv(self.csv_path)
        timestamp = df['timestamp']
        year, month, day = timestamp.str.slice(0, 4), timestamp.str.slice(5, 7), timestamp.str.slice(8, 10)

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

    def __getitem__(self, idx):
        amsr2_file, sic_file = self.file_list[idx]

        # Load AMSR2 frequencies (excluding the last one, which is swath_segmentation)
        with xr.open_dataset(amsr2_file, engine='h5netcdf') as ds:
            data = ds.to_array().values.astype(np.float32) # (n_freq, H, W) 
            data = data[:-1] # exclude swath_segmentation, (n_freq-1, H, W)
        amsr2_tensor = torch.from_numpy(data) # (2, H, W) two 89.9 GHz channels

        amsr2_h, amsr2_w = amsr2_tensor.shape[-2:]
        target_h, target_w = amsr2_h * SCALE_FACTOR, amsr2_w * SCALE_FACTOR

        # Load SIC label, and mask out no-data values (254 and 255)
        with rioxarray.open_rasterio(sic_file) as da:
            raw = da.values.astype(np.float32) # (2, H, W) 
        
        sic = raw[[0]] # (1, H, W)
        invalid_mask = (sic[0] == 254) | (sic[0] == 255)  # (H, W) bool
        invalid_mask = invalid_mask[np.newaxis] # (1, H, W)

        valid_vals = sic[0][~invalid_mask[0]] 
        fill_value = float(valid_vals.mean()) if valid_vals.size > 0 else 0.0
        sic_filled = np.where(invalid_mask, fill_value, sic)  # (1, H, W) no nan

        # Resize SIC to exactly (target_h, target_w) using area averaging (mode = 'area' for downsampling)
        sic_tensor = F.interpolate(
                            torch.from_numpy(sic_filled).unsqueeze(0), # (1, 1, H, W)
                            size=(target_h, target_w), 
                            mode='area').squeeze(0) # (1, target_h, target_w)

        # Resize mask with nearest to ensure boolean
        mask_tensor = F.interpolate(
                            torch.from_numpy(invalid_mask.astype(np.float32)).unsqueeze(0), # (1, 1, H, W)
                            size=(target_h, target_w), 
                            mode='nearest').squeeze(0).bool() # (1, target_h, target_w)
        
        # Add nan in sic where invalid, so loss correctly ignores them
        sic_tensor = sic_tensor.masked_fill(mask_tensor, float('nan')) # (1, H, W) with nan in invalid pixels

        if self.transform:
            amsr2_tensor = self.transform(amsr2_tensor)

        return amsr2_tensor, sic_tensor, mask_tensor 

def collate_variable_size(batch):
    """
    Pads all samples in a batch to the largest (H, W) found in that batch.
    - AMSR2 padded with 0
    - SIC padded with NaN
    - Mask padded with True (invalid) so padded pixels are excluded from loss
    """
    amsr2_list, sic_list, mask_list = zip(*batch)
    
    # AMSR2 padding
    max_h = max(x.shape[-2] for x in amsr2_list)
    max_w = max(x.shape[-1] for x in amsr2_list)
    
    # SIC padding
    max_sh = max(x.shape[-2] for x in sic_list)
    max_sw = max(x.shape[-1] for x in sic_list)


    def pad(tensor, fill, out_h, out_w):
        c, h, w = tensor.shape
        out = torch.full((c, out_h, out_w), fill, dtype=tensor.dtype)
        out[:, :h, :w] = tensor
        return out
 
    amsr2_batch = torch.stack([pad(x, 0.0, max_h, max_w)          for x in amsr2_list])
    sic_batch   = torch.stack([pad(x, float('nan'), max_sh, max_sw) for x in sic_list])
    mask_batch  = torch.stack([pad(x.float(), 1.0, max_sh, max_sw).bool() for x in mask_list])
 
    return amsr2_batch, sic_batch, mask_batch

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
