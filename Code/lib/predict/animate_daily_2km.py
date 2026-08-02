"""
animate_daily_2km.py
---------------------
Plots each day's FusionNetASPP 2km predictions (all scenes for that day,
GCP-interpolated in projected EPSG:3411 space) as a PNG frame, then
assembles all frames into a GIF/MP4 animation.

Frames are cached to disk so the job can be resumed if interrupted —
re-running skips days that already have a frame PNG.

Author: Ninna Juul Ligaard, MSc thesis, DMI/DTU, 2026

Usage:
    python animate_daily_2km.py <start_date> <end_date>
    e.g.: python animate_daily_2km.py 2020/01/01 2022/12/31
"""

import os
import sys
import glob
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — required for batch frame rendering
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from scipy.interpolate import RectBivariateSpline
from pyproj import Transformer
import geopandas as gpd
from NorthPolStere import NorthPolStere
from datetime import datetime, timedelta
from tqdm import tqdm
import imageio.v2 as imageio

# ── Config ────────────────────────────────────────────────────────────────────
SIC_BASE   = '/home/nili/ninna_msc_output'
FRAME_DIR  = '/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI/anim_frames'
OUT_DIR    = '/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI'
SHP_PATH   = '/dmidata/users/nili/Master/Master-thesis---Super-resolution-sea-ice-concentration-using-generative-AI/Code/lib/predict/arctic_shp/op_str_maps_circum_polar_40_EPSG3411.shp'
FPS        = 8
DPI        = 100

shp         = gpd.read_file(SHP_PATH)
transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3411', always_xy=True)
cmap = plt.cm.Blues_r.copy()
cmap.set_bad('none')


def date_range(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def plot_day(DATE_dt):
    """Plot one day's worth of 2km scenes and save as a PNG frame."""
    DATE     = DATE_dt.strftime('%Y/%m/%d')
    y, m, d  = DATE.split('/')
    out_png  = os.path.join(FRAME_DIR, f'frame_{y}{m}{d}.png')

    if os.path.exists(out_png):
        return out_png  # already rendered — skip

    sic_files = sorted(glob.glob(os.path.join(SIC_BASE, y, m, d, '*.nc')))

    fig = plt.figure(figsize=(12, 12))
    proj = NorthPolStere()
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
    ax.set_facecolor('black')

    n_plotted = 0
    for sic_path in sic_files:
        try:
            ds     = xr.open_dataset(sic_path)
            sic    = ds['SIC_pred'].values.astype(np.float32)
            lats   = np.array(ds.attrs['gcp_lats'])
            lons   = np.array(ds.attrs['gcp_lons'])
            lines  = np.array(ds.attrs['gcp_lines'])
            pixels = np.array(ds.attrs['gcp_pixels'])
            ds.close()

            grid_h, grid_w = sic.shape
            n_lines  = len(np.unique(lines))
            n_pixels = len(np.unique(pixels))

            # Project GCP lon/lat → x/y in EPSG:3411 BEFORE interpolating
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

            sic_valid = sic.copy()
            sic_valid[sic >= 254] = np.nan

            ax.pcolormesh(x_scene, y_scene, sic_valid,
                          transform=NorthPolStere(),
                          cmap=cmap, vmin=0, vmax=100,
                          shading='auto', zorder=4)
            n_plotted += 1

        except Exception as e:
            tqdm.write(f'  Error {os.path.basename(sic_path)}: {e}')

    shp.plot(ax=ax, facecolor='#c8c8a0', edgecolor='gray',
             linewidth=0.1, zorder=5)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray',
                      alpha=0.5, linestyle='--')
    gl.top_labels   = False
    gl.right_labels = False

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.05, label='SIC (%)', shrink=0.7)
    ax.set_title(f'{DATE}', fontsize=12)
    fig.tight_layout()

    fig.savefig(out_png, dpi=DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    return out_png


def main():
    if len(sys.argv) != 3:
        print('Usage: python animate_daily_2km.py <start_date> <end_date>')
        print('  Date format: YYYY/MM/DD  e.g. 2020/01/01 2022/12/31')
        sys.exit(1)

    start = datetime.strptime(sys.argv[1], '%Y/%m/%d')
    end   = datetime.strptime(sys.argv[2], '%Y/%m/%d')

    os.makedirs(FRAME_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    dates = list(date_range(start, end))
    print(f'Rendering {len(dates)} daily frames ({start.date()} → {end.date()})...')

    frame_paths = []
    for d in tqdm(dates, desc='Frames'):
        try:
            path = plot_day(d)
            frame_paths.append(path)
        except Exception as e:
            tqdm.write(f'Error on {d.date()}: {e}')

    print(f'\nRendered {len(frame_paths)} frames. Assembling animation...')

    # ── Assemble GIF ──────────────────────────────────────────────────────────
    out_gif = os.path.join(
        OUT_DIR, f'SIC_2km_animation_{start.strftime("%Y%m%d")}_{end.strftime("%Y%m%d")}.gif')
    with imageio.get_writer(out_gif, mode='I', fps=FPS) as writer:
        for path in tqdm(frame_paths, desc='GIF'):
            writer.append_data(imageio.imread(path))
    print(f'Saved GIF → {out_gif}')

    # ── Assemble MP4 (smaller file size, requires ffmpeg) ────────────────────
    try:
        out_mp4 = out_gif.replace('.gif', '.mp4')
        with imageio.get_writer(out_mp4, fps=FPS, codec='libx264',
                                quality=8) as writer:
            for path in tqdm(frame_paths, desc='MP4'):
                writer.append_data(imageio.imread(path))
        print(f'Saved MP4 → {out_mp4}')
    except Exception as e:
        print(f'MP4 export skipped (ffmpeg not available?): {e}')


if __name__ == '__main__':
    main()