"""
Runs FusionNetASPP inference on all AMSR2 files across specified years, 
resamples the grid of the corresponding SAR sentinel1 zip file to georefrence the prediction into the 2 km polar grid,
saves the timestamp, grid and prediction as a netCDF file.

Author: Ninna Juul Ligaard, MSc thesis, DMI/DTU, 2026

Usage: 
    python/run.py
"""

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import sys
import glob
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
import geopandas as gpd
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.control import GroundControlPoint as RioGCP
from rasterio.transform import from_gcps
import io
import rioxarray
from tqdm import tqdm

CODE_DIR    = '/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI/Code'
CKPT_DIR    = '/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI/outputs/training'
AMSR2_DIR   = '/dmidata/projects/asip-cms/tests/new_input_ncs/AMSR2'
SAR_BASE    = '/dmidata/projects/asip-cms/sentinel1'
BASE_OUTPUT = '/dmidata/projects/asip-cms/ninna_msc/output'
LAND_SHP    = '/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI/Code/lib/predict/arctic_shp/op_str_maps_circum_polar_40_EPSG3411.shp' 

sys.path.append(CODE_DIR)
from lib.model.FusionNetASPP import FusionNetASPP
 
### Config ### 
MODEL_NAME      = 'fusionnetaspp'
POSTFIX         = '4'
YEARS           = [2022]
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

### Load model ###
CKPT_PATH = os.path.join(CKPT_DIR, MODEL_NAME, f'best_model_{POSTFIX}.pth')
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model = FusionNetASPP(in_channels=ckpt['in_channels'], features=ckpt['features']).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'Loaded model: {MODEL_NAME} | epoch: {ckpt["epoch"]} | val_rmse: {ckpt["val_rmse"]:.2f}%')

### Landmask ###
gdf_land = gpd.read_file(LAND_SHP)
if gdf_land.crs is None or gdf_land.crs.to_epsg() != 3411:
    gdf_land = gdf_land.set_crs('EPSG:3411', allow_override=True)
land_geoms = [(geom, 1) for geom in gdf_land.geometry if geom is not None]
print(f'Land shapefile loaded: {len(land_geoms)} polygons  CRS: {gdf_land.crs}')

### File list ###
amsr2_files = []
for year in YEARS:
    amsr2_files.extend(
        sorted(glob.glob(os.path.join(AMSR2_DIR, str(year), '*', '*', '*.nc')))
    )
print(f'Found {len(amsr2_files)} AMSR2 files across years: {YEARS}')


### HELPER FUNCTIONS ###
### GCP reader ###
def read_gcp_from_zip(zip_path):
    """Reads gcps from a Sentinel-1 zip file."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        ann_file = [f for f in z.namelist() if f.endswith('.xml') and 'calibration' not in f and 'rfi' not in f and 'hh' in f][0]
        with z.open(ann_file) as f:
            tree = ET.parse(f)
    
    gcps = tree.getroot().findall('.//geolocationGridPoint')
    lines = np.array([int(g.find('line').text)      for g in gcps])
    pixels = np.array([int(g.find('pixel').text)    for g in gcps])
    lats = np.array([float(g.find('latitude').text)  for g in gcps])
    lons = np.array([float(g.find('longitude').text) for g in gcps])
    return lines, pixels, lats, lons

### Mask extractor ###
# Land
def get_land_mask_for_scene(land_geoms, lats, lons, lines, pixels, grid_h, grid_w):
    """
    Rasterize land shapefile onto the exact AMSR2 scene grid using the
    affine transform derived from SAR GCPs via rasterio.from_gcps.
    Correctly handles scene rotation and skew.
    """
    # Transform GCP lon/lat to EPSG:3411
    transformer    = Transformer.from_crs('EPSG:4326', 'EPSG:3411', always_xy=True)
    x_3411, y_3411 = transformer.transform(lons, lats)

    # Scale SAR pixel coordinates to AMSR2 grid size
    sar_line_max  = lines.max()
    sar_pixel_max = pixels.max()
    rows_amsr2    = lines  / sar_line_max  * (grid_h - 1)
    cols_amsr2    = pixels / sar_pixel_max * (grid_w - 1)

    # Build rasterio GCPs — maps AMSR2 pixel (row, col) → EPSG:3411 (x, y)
    gcps = [
        RioGCP(row=r, col=c, x=x, y=y)
        for r, c, x, y in zip(rows_amsr2, cols_amsr2, x_3411, y_3411)
    ]

    # Fit affine transform from GCPs — accounts for rotation and skew
    transform = from_gcps(gcps)

    land_mask = rasterize(
        land_geoms,
        out_shape=(grid_h, grid_w),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    return land_mask

# Invalid
def get_sar_invalid_mask(zip_path, amsr2_h, amsr2_w):
    """
    Load SAR HH and HV measurement tiffs, derive invalid mask (0 DN in
    either channel), then downsample to AMSR2 2km grid using nearest neighbour.
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        tiffs = sorted([f for f in z.namelist()
                        if 'measurement' in f and f.endswith('.tiff')])
        with z.open(tiffs[0]) as f1:
            hh_bytes = io.BytesIO(f1.read())
        with z.open(tiffs[1]) as f2:
            hv_bytes = io.BytesIO(f2.read())
 
    with rioxarray.open_rasterio(hh_bytes) as da:
        hh = da.values[0].astype(np.float32)
    with rioxarray.open_rasterio(hv_bytes) as da:
        hv = da.values[0].astype(np.float32)
 
    invalid_sar = ((hh == 0) | (hv == 0)).astype(np.float32)
    mask_t  = torch.from_numpy(invalid_sar)[None, None]
    resized = F.interpolate(mask_t, size=(amsr2_h, amsr2_w),
                            mode='nearest').numpy()[0, 0]
    return (resized > 0.5).astype(np.uint8)

### Save function ###
def save_prediction_nc(out_file, pred,
                        lines, pixels, lats, lons,
                        amsr2_base, scene_id, ckpt,
                        model_name, postfix):
    """Save SIC prediction and masks as a NetCDF file with sparse GCP
    coordinates and model metadata in global attributes."""
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    ds_out = xr.Dataset(
        {
            'SIC_pred': xr.DataArray(
                pred,
                dims=['y', 'x'],
                attrs={
                    'long_name':     'Sea Ice Concentration',
                    'units':         '%',
                    'valid_range':   [0.0, 100.0],
                    'flag_values':   [254, 255],
                    'flag_meanings': 'land invalid',
                    'comment':       '254=land (from shape file), 255=invalid (from SAR)',
                }
            ),
        },
        attrs={
            'model':        f'{model_name} postfix={postfix}',
            'model_epoch':  int(ckpt['epoch']),
            'val_rmse':     float(ckpt['val_rmse']),
            'val_mae':      float(ckpt['val_mae']),
            'source_amsr2': amsr2_base,
            'source_sar':   scene_id + '.zip',
            'crs':          'EPSG:3411',
            'grid_spacing': '~2km  (resampled onto Sentinel-1 GCP grid)',
            'gcp_lines':    lines.tolist(),
            'gcp_pixels':   pixels.tolist(),
            'gcp_lats':     lats.tolist(),
            'gcp_lons':     lons.tolist(),
        }
    )
    ds_out.to_netcdf(out_file)

### Inference loop — parallel I/O, sequential GPU ###
import multiprocessing as mp
from queue import Empty
# After loading the shapefile — serialise geometries for multiprocessing
import pickle
land_geoms_wkb = [(geom.wkb, val) for geom, val in land_geoms]

def cpu_worker(amsr2_path, sar_path, land_geoms_wkb, amsr2_h, amsr2_w):
    """CPU-bound work: read SAR, compute masks — runs in separate process."""
    from shapely.wkb import loads as wkb_loads
    # Deserialise geometries in the worker process
    land_geoms = [(wkb_loads(wkb), val) for wkb, val in land_geoms_wkb]

    lines, pixels, lats, lons = read_gcp_from_zip(sar_path)
    invalid_mask = get_sar_invalid_mask(sar_path, amsr2_h, amsr2_w)
    land_mask    = get_land_mask_for_scene(
        land_geoms, lats, lons, lines, pixels, amsr2_h, amsr2_w)
    return lines, pixels, lats, lons, invalid_mask, land_mask


### Inference loop — prefetch CPU work while GPU runs ###
from concurrent.futures import ProcessPoolExecutor, as_completed
import collections

errors  = 0
skipped = 0

# Split into batches — prefetch next batch's CPU work while GPU processes current
PREFETCH = 8  # number of scenes to prefetch CPU work for

todo = [(p, os.path.join(SAR_BASE,
         *p.split('/')[p.split('/').index('AMSR2')+1:p.split('/').index('AMSR2')+4],
         os.path.basename(p).replace('AMSR2_','').replace('.nc','') + '.zip'))
        for p in amsr2_files
        if not os.path.exists(os.path.join(
            BASE_OUTPUT,
            *p.split('/')[p.split('/').index('AMSR2')+1:p.split('/').index('AMSR2')+4],
            os.path.basename(p).replace('AMSR2_','').replace('.nc','') + '_prediction.nc'))]

skipped = len(amsr2_files) - len(todo)
print(f'To process: {len(todo)}  Already done: {skipped}')

with ProcessPoolExecutor(max_workers=PREFETCH) as pool:
    # Submit first batch
    pending = collections.deque()
    it = iter(todo)

    def submit_next():
        try:
            amsr2_path, sar_path = next(it)
            if not os.path.exists(sar_path):
                return False, amsr2_path
            # Load AMSR2 shape first to know grid size
            with xr.open_dataset(amsr2_path) as ds:
                ch_names = [v for v in ds.data_vars if 'swath' not in v.lower()]
                shape    = ds[ch_names[0]].shape
            h, w = shape
            fut = pool.submit(cpu_worker, amsr2_path, sar_path, land_geoms_wkb, h, w)
            pending.append((amsr2_path, sar_path, fut))
            return True, None
        except StopIteration:
            return None, None

    # Prime the queue
    for _ in range(PREFETCH):
        ok, bad = submit_next()
        if ok is None:
            break
        if ok is False:
            tqdm.write(f'Warning: SAR not found {bad}')
            errors += 1

    pbar = tqdm(total=len(todo), desc='Processing')
    while pending:
        amsr2_path, sar_path, fut = pending.popleft()

        # Submit next while waiting
        ok, bad = submit_next()
        if ok is False:
            errors += 1

        try:
            lines, pixels, lats, lons, invalid_mask, land_mask = fut.result()

            # Load AMSR2 and run GPU inference — sequential
            with xr.open_dataset(amsr2_path) as ds:
                ch_names = [v for v in ds.data_vars if 'swath' not in v.lower()]
                amsr2_np = ds[ch_names].to_array().values.astype(np.float32)
            amsr2_h, amsr2_w = amsr2_np.shape[-2], amsr2_np.shape[-1]

            amsr2_input = np.nan_to_num(amsr2_np, nan=150.0)
            amsr2_t     = torch.from_numpy(amsr2_input)[None].to(device)
            with torch.no_grad():
                pred = model(amsr2_t, target_size=(amsr2_h, amsr2_w))
            pred_np = np.clip(pred[0, 0].cpu().numpy(), 0, 100)

            pred_np[:, :4]  = 255
            pred_np[:, -4:] = 255
            pred_np[invalid_mask == 1] = 255
            pred_np[land_mask    == 1] = 254

            parts    = amsr2_path.split('/')
            date_idx = parts.index('AMSR2') + 1
            y, m, d  = parts[date_idx], parts[date_idx+1], parts[date_idx+2]
            amsr2_base = os.path.basename(amsr2_path)
            scene_id   = amsr2_base.replace('AMSR2_', '').replace('.nc', '')
            out_dir    = os.path.join(BASE_OUTPUT, y, m, d)
            out_file   = os.path.join(out_dir, scene_id + '_prediction.nc')

            save_prediction_nc(out_file, pred_np,
                               lines, pixels, lats, lons,
                               amsr2_base, scene_id, ckpt,
                               MODEL_NAME, POSTFIX)
        except Exception as e:
            tqdm.write(f'Error: {os.path.basename(amsr2_path)}: {e}')
            errors += 1

        pbar.update(1)
    pbar.close()

print(f'\nDone.')
print(f'  Processed : {len(todo) - errors}')
print(f'  Skipped   : {skipped}')
print(f'  Errors    : {errors}')

