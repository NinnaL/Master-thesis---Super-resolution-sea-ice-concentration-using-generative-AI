"""
mosaic_day.py
-------------
Takes all retrieved SIC NetCDF files for a given day, mosaics them onto
the same 25km common grid used for ASIP/Landsat/OSI-SAF validation
(derived from the ASIP L3 500m grid, subsampled by 50x), using
pyresample nearest-neighbour resampling. Overlapping pixels are averaged.

GCP interpolation is done in projected EPSG:3411 x/y space (not lon/lat)
to avoid distortion near the pole, then converted back to lon/lat for
pyresample's SwathDefinition.

Saves the result as a NetCDF file.

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
from pyproj import Transformer
from scipy.interpolate import RectBivariateSpline
import pyresample
from pyresample import geometry
from datetime import datetime, timedelta
from tqdm import tqdm
import geopandas as gpd
from NorthPolStere import NorthPolStere

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_BASE  = '/home/nili/ninna_msc_output'
OUTPUT_BASE = '/dmidata/projects/asip-cms/ninna_msc/output_mosaic'
shp = gpd.read_file('/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI/Code/lib/predict/arctic_shp/op_str_maps_circum_polar_40_EPSG3411.shp')

transformer      = Transformer.from_crs('EPSG:4326', 'EPSG:3411', always_xy=True)
transformer_back = Transformer.from_crs('EPSG:3411', 'EPSG:4326', always_xy=True)

# ── Common 25km grid — same as ASIP/Landsat/OSI-SAF validation pipeline ──────
ymin, ymax = 7500, 20000
xmin, xmax = 1500, 13500
ASIP_L3_FILE = '/dmidata/projects/asip-cms/reproc/mosaics/level3_0500m_v1/dmi_asip_seaice_mosaic_arc_l3_20200101.nc'

ds_ref  = xr.open_dataset(ASIP_L3_FILE)
lat_ref = ds_ref['lat'][ymin:ymax, xmin:xmax].values
lon_ref = ds_ref['lon'][ymin:ymax, xmin:xmax].values
ds_ref.close()

# Subsample 500m → 25km (factor 50), exactly matching the ASIP pipeline
new_lat = lat_ref[::50, ::50]
new_lon = lon_ref[::50, ::50]
common_swath_def = pyresample.geometry.SwathDefinition(lons=new_lon, lats=new_lat)
# print(f'Common grid shape: {new_lat.shape}')

RADIUS_OF_INF = 25_000  # same scale as the 25km grid spacing


def process(DATE):
    y, m, d = DATE.split('/')

    pred_files = sorted(glob.glob(
        os.path.join(INPUT_BASE, y, m, d, '*_prediction.nc')))

    if len(pred_files) == 0:
        raise FileNotFoundError(f'No prediction files in {INPUT_BASE}/{y}/{m}')

    def extract_timestamp(path):
        return os.path.basename(path).split('_')[4]

    pred_files = sorted(pred_files, key=extract_timestamp)

    # ── Build mosaic — overlapping pixels averaged ────────────────────────────
    mosaic_sum   = np.zeros(new_lat.shape, dtype=np.float32)
    mosaic_count = np.zeros(new_lat.shape, dtype=np.float32)

    errors = 0
    for pred_path in tqdm(pred_files, desc=f'Mosaicking {DATE}'):
        try:
            ds     = xr.open_dataset(pred_path)
            sic    = ds['SIC_pred'].values.astype(np.float32)
            lats   = np.array(ds.attrs['gcp_lats'])
            lons   = np.array(ds.attrs['gcp_lons'])
            lines  = np.array(ds.attrs['gcp_lines'])
            pixels = np.array(ds.attrs['gcp_pixels'])
            ds.close()

            grid_h, grid_w = sic.shape

            n_lines  = len(np.unique(lines))
            n_pixels = len(np.unique(pixels))

            # ── Project GCP lon/lat → EPSG:3411 x/y BEFORE interpolating ───────
            # Avoids pole-singularity distortion from interpolating raw lon/lat
            gcp_x, gcp_y = transformer.transform(
                lons.reshape(n_lines, n_pixels),
                lats.reshape(n_lines, n_pixels)
            )

            x_interp = RectBivariateSpline(
                lines.reshape(n_lines, n_pixels)[:, 0],
                pixels.reshape(n_lines, n_pixels)[0, :],
                gcp_x)
            y_interp = RectBivariateSpline(
                lines.reshape(n_lines, n_pixels)[:, 0],
                pixels.reshape(n_lines, n_pixels)[0, :],
                gcp_y)

            row_coords = np.linspace(0, lines.max(), grid_h)
            col_coords = np.linspace(0, pixels.max(), grid_w)
            x_scene    = x_interp(row_coords, col_coords)
            y_scene    = y_interp(row_coords, col_coords)

            # Convert the now-correct projected coords back to lon/lat
            # for pyresample's SwathDefinition
            lon_scene, lat_scene = transformer_back.transform(x_scene, y_scene)

            # Mask sentinel values
            sic_valid = sic.copy()
            sic_valid[sic >= 200] = np.nan

            source_swath = geometry.SwathDefinition(lons=lon_scene, lats=lat_scene)

            # resampled = pyresample.kd_tree.resample_nearest(
            #     source_swath,
            #     sic_valid,
            #     common_swath_def,
            #     radius_of_influence=RADIUS_OF_INF,
            #     fill_value=np.nan,
            # )
            resampled = pyresample.kd_tree.resample_gauss(source_swath, sic_valid, common_swath_def, radius_of_influence=RADIUS_OF_INF, sigmas=12_500, fill_value=np.nan)

            # Accumulate for averaging
            valid_new = ~np.isnan(resampled)
            mosaic_sum[valid_new]   += resampled[valid_new]
            mosaic_count[valid_new] += 1.0

        except Exception as e:
            tqdm.write(f'  Error {os.path.basename(pred_path)}: {e}')
            errors += 1

    # ── Average overlapping pixels ────────────────────────────────────────────
    mosaic = np.full(new_lat.shape, np.nan, dtype=np.float32)
    valid  = mosaic_count > 0
    mosaic[valid] = mosaic_sum[valid] / mosaic_count[valid]

    n_used = len(pred_files) - errors
    # print(f'{DATE}: {n_used}/{len(pred_files)} scenes  '
    #       f'valid pixels: {valid.sum()}  max overlap: {int(mosaic_count.max())}')

    if not valid.any():
        print(f'No valid data for {DATE} — skipping save')
        return

    # ── Crop to data extent ───────────────────────────────────────────────────
    rows_with_data = np.where(valid.any(axis=1))[0]
    cols_with_data = np.where(valid.any(axis=0))[0]

    r0, r1 = rows_with_data[0], rows_with_data[-1] + 1
    c0, c1 = cols_with_data[0], cols_with_data[-1] + 1
    mosaic_crop = mosaic[r0:r1, c0:c1]
    lon_out     = new_lon[r0:r1, c0:c1]
    lat_out     = new_lat[r0:r1, c0:c1]

    # ── Save NetCDF ───────────────────────────────────────────────────────────
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
                    'comment':     'Overlapping pixels averaged. On ASIP common 25km grid.',
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
            'grid':          'ASIP L3 500m subsampled 50x to ~25km',
            'creation_date': datetime.utcnow().isoformat(),
            'crop_r0': int(r0), 'crop_c0': int(c0),
            'crop_r1': int(r1), 'crop_c1': int(c1),
        }
    ).to_netcdf(out_nc)
    print(f'Saved → {out_nc}')

    return lon_out, lat_out, mosaic_crop, n_used


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_mosaic(DATE, lon_out, lat_out, mosaic_crop, n_used, y, m, d):
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

    out_png = os.path.join(OUTPUT_BASE, y, m, f'SIC_mosaic_{y}{m}{d}.png')
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

    for date in dates:
        process(date)


if __name__ == '__main__':
    main()