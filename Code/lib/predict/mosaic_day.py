"""
mosaic_day.py
-------------
Takes all retrieved SIC NetCDF files for a given day, mosaics them onto
a common 25km EPSG:3411 grid using pyresample and average overlapping pixels,
plots the result and saves it as a NetCDF file.

KD-tree resampling indices computed once per scene (only 1 variable to
resample vs 14 for AMSR2 mosaic) — fast and clean.

Author: Ninna Juul Ligaard, MSc thesis, DMI/DTU, 2026

Usage:
    python Code/lib/predict/mosaic_day.py <start_date> <end_date>
    e.g.: python Code/lib/predict/mosaic_day.py 2020/01/01 2020/01/05
"""

import os
import sys
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
from datetime import datetime, timedelta
from tqdm import tqdm
import geopandas as gpd
from NorthPolStere import NorthPolStere

import warnings
warnings.filterwarnings('ignore', message='Possible more than 30 neighbours', category=UserWarning)

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_BASE  = '/home/nili/ninna_msc_output'
OUTPUT_BASE = '/dmidata/projects/asip-cms/ninna_msc/output_mosaic'
shp = gpd.read_file('/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI/Code/lib/predict/arctic_shp/op_str_maps_circum_polar_40_EPSG3411.shp')

REF_RES  = 25000
REF_EXT  = 7_000_000
REF_SIZE = int(2 * REF_EXT / REF_RES)

# ── Target grid ───────────────────────────────────────────────────────────────
target_area = AreaDefinition(
    area_id     = 'arctic_25km',
    description = 'Arctic 25km EPSG:3411',
    proj_id     = 'EPSG:3411',
    projection  = {'proj': 'stere', 'lat_0': 90, 'lat_ts': 70,
                   'lon_0': -45, 'x_0': 0, 'y_0': 0,
                   'a': 6378273, 'b': 6356889.449},
    width       = REF_SIZE,
    height      = REF_SIZE,
    area_extent = (-REF_EXT, -REF_EXT, REF_EXT, REF_EXT),
)


def process(DATE):
    y, m, d     = DATE.split('/')

    # ── Find and sort prediction files ──────────
    pred_files = sorted(glob.glob(
        os.path.join(INPUT_BASE, y, m, d, '*_prediction.nc')))
    # print(f'Found {len(pred_files)} predicted files for {DATE}')

    if len(pred_files) == 0:
        raise FileNotFoundError(f'No prediction files in {INPUT_BASE}/{y}/{m}')

    def extract_timestamp(path):
        return os.path.basename(path).split('_')[4]  # e.g. '20200101T025017'

    pred_files = sorted(pred_files, key=extract_timestamp)
    # print(f'Time range: {extract_timestamp(pred_files[0])} → {extract_timestamp(pred_files[-1])}')
    # pred_files = pred_files[:4] # for testing — IGNORE



    # ── Build mosaic — newest overwrites oldest ───────────────────────────────────
    # print('Building SIC mosaic...')
    mosaic_sum   = np.zeros((REF_SIZE, REF_SIZE), dtype=np.float32)
    mosaic_count = np.zeros((REF_SIZE, REF_SIZE), dtype=np.float32)

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
                    radius_of_influence=30000,
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
                weight_funcs=lambda r: np.exp(-r**2 / (2 * 12500**2)), #sigma 12.5 km
                fill_value=np.nan,
            )

            # Newest overwrites — simply replace valid pixels
            valid_new = ~np.isnan(resampled)
            mosaic_sum[valid_new] += resampled[valid_new]
            mosaic_count[valid_new] += 1

        except Exception as e:
            tqdm.write(f'  Error {os.path.basename(pred_path)}: {e}')
            errors += 1

    # ── Average overlapping pixels ────────────────────────────────────────────────
    mosaic = np.full((REF_SIZE, REF_SIZE), np.nan, dtype=np.float32)
    valid  = mosaic_count > 0
    mosaic[valid] = mosaic_sum[valid] / mosaic_count[valid]

    n_used = len(pred_files) - errors
    # print(f'Mosaic complete — {n_used} scenes used  {errors} errors')
    # print(f'Overlap: max {mosaic_count.max():.0f} scenes per pixel  '
    #     f'mean {mosaic_count[valid].mean():.2f}')

    # ── Crop to data extent ───────────────────────────────────────────────────────
    valid_mosaic   = ~np.isnan(mosaic)
    rows_with_data = np.where(valid_mosaic.any(axis=1))[0]
    cols_with_data = np.where(valid_mosaic.any(axis=0))[0]

    if len(rows_with_data) == 0:
        raise ValueError('No valid data in mosaic')

    r0, r1 = rows_with_data[0], rows_with_data[-1] + 1
    c0, c1 = cols_with_data[0], cols_with_data[-1] + 1
    mosaic_crop = mosaic[r0:r1, c0:c1]
    # print(f'Cropped mosaic: {mosaic_crop.shape[0]} × {mosaic_crop.shape[1]}')

    # ── Back-transform to lon/lat ─────────────────────────────────────────────────
    transformer_back = Transformer.from_crs('EPSG:3411', 'EPSG:4326', always_xy=True)
    col_coords_crop  = (np.arange(c0, c1) * REF_RES - REF_EXT + REF_RES / 2)
    row_coords_crop  = (REF_EXT - np.arange(r0, r1) * REF_RES - REF_RES / 2)
    xx, yy           = np.meshgrid(col_coords_crop, row_coords_crop)
    lon_out, lat_out = transformer_back.transform(xx, yy)

    # ── Save NetCDF ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.join(OUTPUT_BASE, y, m), exist_ok=True)
    out_nc = os.path.join(OUTPUT_BASE, y, m, f'SIC_mosaic_{y}{m}{d}.nc')

    xr.Dataset(
        {
            'SIC': xr.DataArray(
                mosaic_crop, dims=['y', 'x'],
                attrs={
                    'long_name':   'Sea Ice Concentration mosaic',
                    'units':       '%',
                    'valid_range': [0.0, 100.0],
                    'comment':     'Overlapping pixels are averaged. NaN = no data.',
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
def plot_mosaic():
    cmap = plt.cm.Blues_r.copy()
    cmap.set_bad('none')

    proj = NorthPolStere()
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), subplot_kw={'projection': proj})
    ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
    ax.set_facecolor('black')

    ax.pcolormesh(lon_out, lat_out, mosaic_crop,
                transform=ccrs.PlateCarree(),
                cmap=cmap, vmin=0, vmax=100,
                shading='auto', zorder=4, alpha=1)

    shp.plot(ax=ax, facecolor='#c8c8a0', edgecolor='gray',
            linewidth=0.1, zorder=5)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray',
                    alpha=0.5, linestyle='--')
    gl.top_labels   = False
    gl.right_labels = False

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.05, label='SIC (%)', shrink=0.7)
    ax.set_title(f'FusionNetASPP SIC mosaic — {DATE}  ({n_used} scenes)', fontsize=12)
    plt.tight_layout()

    out_png = os.path.join(OUTPUT_BASE, y, m, f'SIC_mosaic2_{y}{m}{d}.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    print(f'Saved plot → {out_png}')

def date_range(start_date, end_date):
    current_date = start_date
    while current_date <= end_date:
        yield current_date.strftime('%Y/%m/%d')
        current_date += timedelta(days=1)

def main():
    if len(sys.argv) != 3:
            print('Usage: python mosaic_day.py <start_date> <end_date>')
            print('  Date format: YYYY/MM/DD  e.g. 2020/01/01 2020/01/05')
            sys.exit(1)

    try:
        start = datetime.strptime(sys.argv[1], '%Y/%m/%d')
        end   = datetime.strptime(sys.argv[2], '%Y/%m/%d')
    except ValueError:
        print('Error: dates must be in YYYY/MM/DD format.')
        sys.exit(1)

    if end < start:
        print('Error: end_date must be >= start_date.')
        sys.exit(1)

    dates = list(date_range(start, end))
    # print(f'Processing {len(dates)} day(s): {dates[0]} → {dates[-1]}')

    for date in dates:
        process(date)


if __name__ == '__main__':
    main()
