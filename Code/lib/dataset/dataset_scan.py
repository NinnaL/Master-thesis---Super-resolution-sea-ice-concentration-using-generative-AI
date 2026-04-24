import zarr
import numpy as np

store = zarr.open_group('/dmidata/projects/asip-cms/ninna_msc/zarr_cache/val', mode='r')
keys  = sorted(store['amsr2'].keys(), key=int)
bad   = []

for i, key in enumerate(keys):
    amsr2 = store['amsr2'][key][:]
    sic   = store['sic'][key][:]
    mask  = store['mask'][key][:]
    valid = ~mask

    reasons = []

    # AMSR2 checks
    if np.isnan(amsr2).any():
        reasons.append(f'amsr2_nan={np.isnan(amsr2).sum()}')
    if np.isinf(amsr2).any():
        reasons.append(f'amsr2_inf={np.isinf(amsr2).sum()}')
    if (amsr2 == 0).all():
        reasons.append('amsr2_all_zero')

    # SIC checks
    if valid.any():
        valid_sic = sic[valid]
        if np.isnan(valid_sic).any():
            reasons.append(f'sic_nan={np.isnan(valid_sic).sum()}')
        if np.isinf(valid_sic).any():
            reasons.append(f'sic_inf={np.isinf(valid_sic).sum()}')
        if valid_sic.max() > 100.0:
            reasons.append(f'sic_above_100={valid_sic.max():.2f}')
        if valid_sic.min() < 0.0:
            reasons.append(f'sic_below_0={valid_sic.min():.2f}')
    else:
        reasons.append('all_masked')

    # Shape check
    if amsr2.shape[-2:] != sic.shape[-2:]:
        reasons.append(f'shape_mismatch amsr2={amsr2.shape} sic={sic.shape}')

    if reasons:
        bad.append((key, reasons))

    if i % 1000 == 0:
        print(f'{i}/{len(keys)} checked — {len(bad)} bad so far')

print(f'\nTotal bad samples: {len(bad)}')
for key, reasons in bad:
    print(f'  key={key}  issues={reasons}')