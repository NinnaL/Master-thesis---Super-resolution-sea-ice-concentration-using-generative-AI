import os
import torch
import zarr
from torch.utils.data import DataLoader


class AMSR2Dataset(torch.utils.data.Dataset):
    """
    Reads preprocessed AMSR2 and SIC data from zarr cache.
    Split is determined by the subdirectory — no leakage between train and val.

    Returns a 3-tuple per sample:
        amsr2 : (14, H, W)         float32
        sic   : (1,  H_sic, W_sic) float32
        mask  : (1,  H_sic, W_sic) bool     True where invalid (254 or 255)
    """

    def __init__(self, cache_dir, split, transform=None):
        assert split in ['train', 'val'], "split must be 'train' or 'val'"

        self.transform = transform
        split_dir = os.path.join(cache_dir, split)

        self.store   = zarr.open_group(zarr.DirectoryStore(split_dir), mode='r')
        self.indices = sorted(self.store['amsr2'].keys(), key=int)

    @property
    def channel_names(self):
        """AMSR2 variable names in channel order."""
        return self.store.attrs['channel_names']

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        key = self.indices[idx]

        amsr2 = self.store['amsr2'][key][:]   # (14, H, W)        float32
        sic   = self.store['sic'][key][:]     # (1, H_sic, W_sic) float32
        mask  = self.store['mask'][key][:]    # (1, H_sic, W_sic) bool

        amsr2_tensor = torch.from_numpy(amsr2)
        sic_tensor   = torch.from_numpy(sic)
        mask_tensor  = torch.from_numpy(mask)

        if self.transform:
            amsr2_tensor = self.transform(amsr2_tensor)

        return amsr2_tensor, sic_tensor, mask_tensor


def collate_pad_to_max(batch):
    """
    Pads all samples in a batch to the largest (H, W) in that batch.
        - AMSR2 padded with 0     (neutral for brightness temperatures)
        - SIC   padded with 255   (matches raw fill convention, excluded by mask)
        - mask  padded with True  (marks padded region as invalid)
    """
    amsr2_list, sic_list, mask_list = zip(*batch)

    max_h  = max(x.shape[-2] for x in amsr2_list)
    max_w  = max(x.shape[-1] for x in amsr2_list)
    max_sh = max(x.shape[-2] for x in sic_list)
    max_sw = max(x.shape[-1] for x in sic_list)

    def pad(tensor, fill, out_h, out_w):
        c, h, w = tensor.shape
        out = torch.full((c, out_h, out_w), fill, dtype=tensor.dtype)
        out[:, :h, :w] = tensor
        return out

    def pad_mask(tensor, out_h, out_w):
        c, h, w = tensor.shape
        out = torch.ones((c, out_h, out_w), dtype=torch.bool)  # True = invalid
        out[:, :h, :w] = tensor
        return out

    amsr2_batch = torch.stack([pad(x, 0.0,   max_h,  max_w)  for x in amsr2_list])
    sic_batch   = torch.stack([pad(x, 255.0, max_sh, max_sw) for x in sic_list])
    mask_batch  = torch.stack([pad_mask(x,   max_sh, max_sw) for x in mask_list])

    return amsr2_batch, sic_batch, mask_batch


def collate_crop_to_min(batch):
    """
    Crops all samples in a batch to the smallest (H, W) in that batch.
    Eliminates padding entirely — memory is bounded by the smallest sample,
    not the largest. Border pixels from larger swaths are discarded.

        - AMSR2 cropped to (min_h, min_w)
        - SIC   cropped to (min_sh, min_sw)
        - mask  cropped to (min_sh, min_sw)
    """
    amsr2_list, sic_list, mask_list = zip(*batch)

    min_h  = min(x.shape[-2] for x in amsr2_list)
    min_w  = min(x.shape[-1] for x in amsr2_list)
    min_sh = min(x.shape[-2] for x in sic_list)
    min_sw = min(x.shape[-1] for x in sic_list)

    amsr2_batch = torch.stack([x[:, :min_h,  :min_w]  for x in amsr2_list])
    sic_batch   = torch.stack([x[:, :min_sh, :min_sw] for x in sic_list])
    mask_batch  = torch.stack([x[:, :min_sh, :min_sw] for x in mask_list])

    return amsr2_batch, sic_batch, mask_batch