import zarr
import numpy as np

store = zarr.open_group('/dmidata/projects/asip-cms/ninna_msc/zarr_cache/train', mode='r')
keys  = sorted(store['amsr2'].keys(), key=int)
bad   = []

for i, key in enumerate(keys):
    amsr2 = store['amsr2'][key][:]
    sic   = store['sic'][key][:]
    mask  = store['mask'][key][:]

    amsr2_bad = np.isnan(amsr2).any() or np.isinf(amsr2).any()
    sic_bad   = np.isnan(sic[~mask]).any() if (~mask).any() else False

    if amsr2_bad or sic_bad:
        bad.append((key, 'amsr2' if amsr2_bad else 'sic'))

    if i % 1000 == 0:
        print(f'{i}/{len(keys)} checked — {len(bad)} bad so far')

print(f'\nTotal bad samples: {len(bad)}')
for key, source in bad:
    print(f'  key={key}  source={source}')