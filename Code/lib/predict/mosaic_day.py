"""
mosaic_day.py
-------------
Takes all predicted SIC NetCDF files for a given day, mosaics them onto
a common 2km EPSG:3411 grid using pyresample with the newest data on top,
plots the result and saves it as a NetCDF file.

KD-tree resampling indices computed once per scene (only 1 variable to
resample vs 14 for AMSR2 mosaic) — fast and clean.

Author: Ninna Juul Ligaard, MSc thesis, DMI/DTU, 2026

Usage:
    python Code/mosaic_day.py
"""

import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
from pyproj import Transformer
from scipy.interpolate import RectBivariateSpline
from pyresample import geometry, kd_tree
from pyresample.geometry import AreaDefinition
from datetime import datetime
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATE        = '2020/01/01'
INPUT_BASE  = '/dmidata/projects/asip-cms/ninna_msc/output'
OUTPUT_BASE = '/dmidata/projects/asip-cms/ninna_msc/output_mosaic'
y, m, d     = DATE.split('/')

REF_RES  = 2000
REF_EXT  = 7_000_000
REF_SIZE = int(2 * REF_EXT / REF_RES)

# ── Find and sort prediction files — oldest first, newest overwrites ──────────
pred_files = sorted(glob.glob(
    os.path.join(INPUT_BASE, y, m, d, '*_prediction.nc')))
print(f'Found {len(pred_files)} prediction files for {DATE}')

if len(pred_files) == 0:
    raise FileNotFoundError(f'No prediction files in {INPUT_BASE}/{y}/{m}/{d}')

def extract_timestamp(path):
    return os.path.basename(path).split('_')[4]  # e.g. '20200101T025017'

pred_files = sorted(pred_files, key=extract_timestamp)
print(f'Time range: {extract_timestamp(pred_files[0])} → {extract_timestamp(pred_files[-1])}')
pred_files = pred_files[:4]

# ── Target area ───────────────────────────────────────────────────────────────
target_area = AreaDefinition(
    area_id     = 'arctic_2km',
    description = 'Arctic 2km EPSG:3411',
    proj_id     = 'EPSG:3411',
    projection  = {'proj': 'stere', 'lat_0': 90, 'lat_ts': 70,
                   'lon_0': -45, 'x_0': 0, 'y_0': 0,
                   'a': 6378273, 'b': 6356889.449},
    width       = REF_SIZE,
    height      = REF_SIZE,
    area_extent = (-REF_EXT, -REF_EXT, REF_EXT, REF_EXT),
)

# ── Build mosaic — newest overwrites oldest ───────────────────────────────────
print('Building SIC mosaic...')
mosaic = np.full((REF_SIZE, REF_SIZE), np.nan, dtype=np.float32)

errors = 0
for pred_path in tqdm(pred_files, desc='Mosaicking'):
    try:
        ds   = xr.open_dataset(pred_path)
        sic  = ds['SIC_pred'].values.astype(np.float32)
        lats = np.array(ds.attrs['gcp_lats'])
        lons = np.array(ds.attrs['gcp_lons'])
        lines  = np.array(ds.attrs['gcp_lines'])
        pixels = np.array(ds.attrs['gcp_pixels'])
        ds.close()

        grid_h, grid_w = sic.shape

        # Interpolate sparse GCPs to full scene lon/lat grid
        n_lines  = len(np.unique(lines))
        n_pixels = len(np.unique(pixels))
        lon_interp = RectBivariateSpline(
            lines.reshape(n_lines, n_pixels)[:, 0],
            pixels.reshape(n_lines, n_pixels)[0, :],
            lons.reshape(n_lines, n_pixels))
        lat_interp = RectBivariateSpline(
            lines.reshape(n_lines, n_pixels)[:, 0],
            pixels.reshape(n_lines, n_pixels)[0, :],
            lats.reshape(n_lines, n_pixels))

        row_coords = np.linspace(0, lines.max(), grid_h)
        col_coords = np.linspace(0, pixels.max(), grid_w)
        lon_grid   = lon_interp(row_coords, col_coords)
        lat_grid   = lat_interp(row_coords, col_coords)

        # Mask sentinel values
        sic_valid = sic.copy()
        sic_valid[sic >= 254] = np.nan

        source_swath = geometry.SwathDefinition(lons=lon_grid, lats=lat_grid)

        # Compute KD-tree indices once — only 1 variable so apply directly
        valid_input_index, valid_output_index, index_array, distance_array = \
            kd_tree.get_neighbour_info(
                source_swath,
                target_area,
                radius_of_influence=3000,
                neighbours=30,
                nprocs=1,
            )

        resampled = kd_tree.get_sample_from_neighbour_info(
            'custom',
            target_area.shape,
            sic_valid.ravel(),
            valid_input_index,
            valid_output_index,
            index_array,
            distance_array=distance_array,
            weight_funcs=lambda r: np.exp(-r**2 / (2 * 1500**2)),
            fill_value=np.nan,
        )

        # Newest overwrites — simply replace valid pixels
        valid_new = ~np.isnan(resampled)
        mosaic[valid_new] = resampled[valid_new]

    except Exception as e:
        tqdm.write(f'  Error {os.path.basename(pred_path)}: {e}')
        errors += 1

n_used = len(pred_files) - errors
print(f'Mosaic complete — {n_used} scenes used  {errors} errors')

# ── Crop to data extent ───────────────────────────────────────────────────────
valid_mosaic   = ~np.isnan(mosaic)
rows_with_data = np.where(valid_mosaic.any(axis=1))[0]
cols_with_data = np.where(valid_mosaic.any(axis=0))[0]

if len(rows_with_data) == 0:
    raise ValueError('No valid data in mosaic')

r0, r1 = rows_with_data[0], rows_with_data[-1] + 1
c0, c1 = cols_with_data[0], cols_with_data[-1] + 1
mosaic_crop = mosaic[r0:r1, c0:c1]
print(f'Cropped mosaic: {mosaic_crop.shape[0]} × {mosaic_crop.shape[1]}')

# ── Back-transform to lon/lat ─────────────────────────────────────────────────
transformer_back = Transformer.from_crs('EPSG:3411', 'EPSG:4326', always_xy=True)
col_coords_crop  = (np.arange(c0, c1) * REF_RES - REF_EXT + REF_RES / 2)
row_coords_crop  = (REF_EXT - np.arange(r0, r1) * REF_RES - REF_RES / 2)
xx, yy           = np.meshgrid(col_coords_crop, row_coords_crop)
lon_out, lat_out = transformer_back.transform(xx, yy)

# ── Save NetCDF ───────────────────────────────────────────────────────────────
os.makedirs(os.path.join(OUTPUT_BASE, y, m, d), exist_ok=True)
out_nc = os.path.join(OUTPUT_BASE, y, m, d, f'SIC_mosaic_{y}{m}{d}.nc')

xr.Dataset(
    {
        'SIC_pred': xr.DataArray(
            mosaic_crop, dims=['y', 'x'],
            attrs={
                'long_name':   'Sea Ice Concentration mosaic',
                'units':       '%',
                'valid_range': [0.0, 100.0],
                'comment':     'Newest scene on top — oldest to newest order',
            }
        ),
        'lon': xr.DataArray(lon_out, dims=['y', 'x'],
                            attrs={'long_name': 'longitude', 'units': 'degrees_east'}),
        'lat': xr.DataArray(lat_out, dims=['y', 'x'],
                            attrs={'long_name': 'latitude',  'units': 'degrees_north'}),
    },
    attrs={
        'date':          DATE,
        'n_scenes':      n_used,
        'crs':           'EPSG:3411',
        'grid_spacing':  f'{REF_RES}m',
        'creation_date': datetime.utcnow().isoformat(),
    }
).to_netcdf(out_nc)
print(f'Saved NetCDF → {out_nc}')

# ── Plot ──────────────────────────────────────────────────────────────────────
cmap = plt.cm.Blues_r.copy()
cmap.set_bad('none')

proj = ccrs.NorthPolarStereo(central_longitude=0, true_scale_latitude=70)
fig, ax = plt.subplots(1, 1, figsize=(12, 12), subplot_kw={'projection': proj})
ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
ax.set_facecolor('black')

ax.pcolormesh(lon_out, lat_out, mosaic_crop,
              transform=ccrs.PlateCarree(),
              cmap=cmap, vmin=0, vmax=100,
              shading='auto', zorder=4, alpha=1)

land = NaturalEarthFeature('physical', 'land', '10m',
                            facecolor='#c8c8a0', edgecolor='gray',
                            linewidth=0.5)
ax.add_feature(land, zorder=5)

gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray',
                  alpha=0.5, linestyle='--')
gl.top_labels   = False
gl.right_labels = False

sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=100))
sm.set_array([])
plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.05, label='SIC (%)', shrink=0.7)
ax.set_title(f'FusionNetASPP SIC mosaic — {DATE}  ({n_used} scenes)', fontsize=12)
plt.tight_layout()

out_png = os.path.join(OUTPUT_BASE, y, m, d, f'SIC_mosaic2_{y}{m}{d}.png')
plt.savefig(out_png, dpi=150, bbox_inches='tight', pad_inches=0.1)
plt.show()
print(f'Saved plot → {out_png}')