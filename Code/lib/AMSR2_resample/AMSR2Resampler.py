import os
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
import logging
import h5py
import satpy
import numpy as np
import xarray as xr
import pyproj
from pyresample import kd_tree, utils
from pyresample.geometry import SwathDefinition, AreaDefinition
from datetime import datetime

LOG = logging.getLogger(__name__)


class AMSR2Resampler:
    """
    Resamples a single AMSR2 L1b .h5 swath directly onto a fixed target_grid
    (a pyresample AreaDefinition), with no timestamp matching, intersection
    checks, or multi-swath blending — one file in, one resampled file out.
    """

    BEAM_WIDTHS = {
        'btemp_6.9h':  [35, 62], 'btemp_6.9v':  [35, 62],
        'btemp_7.3h':  [35, 62], 'btemp_7.3v':  [35, 62],
        'btemp_10.7h': [24, 42], 'btemp_10.7v': [24, 42],
        'btemp_18.7h': [24, 42], 'btemp_18.7v': [24, 42],
        'btemp_23.8h': [15, 26], 'btemp_23.8v': [15, 26],
        'btemp_36.5h': [7, 12],  'btemp_36.5v': [7, 12],
        'btemp_89.0ah': [3, 5], 'btemp_89.0bh': [3, 5],
        'btemp_89.0av': [3, 5], 'btemp_89.0bv': [3, 5],
    }
    CHS = list(BEAM_WIDTHS.keys())
    COMBINED_89_BEAM_WIDTHS = {'btemp_89.0h': [3, 5], 'btemp_89.0v': [3, 5]}
    COMBINED_89_CHS = list(COMBINED_89_BEAM_WIDTHS.keys())

    SUBAREA_MARGIN = 150 # km margin around the AMSR2 swath footprint to avoid edge effects during resampling

    def __init__(self, amsr2_file, output_dir, target_grid, hemisphere=None):
        self.amsr2_file = str(amsr2_file)
        self.output_dir = str(output_dir)
        self.target_grid = target_grid
        self.lat_limits = (
            [60, 90] if hemisphere == 'N' else
            [-90, -60] if hemisphere == 'S' else
            None
        )

        self._check_corrupted()
        self.ds_combined = None

    def _check_corrupted(self):
        """
        Occasionally AMSR2 L1b .h5 files contain invalid data (2**16-1 sentinel).
        Raises ValueError if invalid pixels are found north/south of the hemisphere limit.
        """
        with h5py.File(self.amsr2_file, 'r', locking=False) as f:
            bt89 = f['Brightness Temperature (89.0GHz-A,H)'][:]
            lats89 = f['Latitude of Observation Point for 89A'][:]

        if self.lat_limits is not None:
            mask = (lats89 > self.lat_limits[0]) if self.lat_limits[0] >= 0 else (lats89 < self.lat_limits[1])
        else:
            mask = np.ones_like(lats89, dtype=bool)

        if (bt89[mask] == 2**16 - 1).any():
            raise ValueError(f"Corrupted AMSR2 file detected: {self.amsr2_file}")

    def _get_combined_89(self, ds):
        """Combine the 89.0GHz-A and 89.0GHz-B sub-beams into single btemp_89.0h/v channels."""
        out = {}
        for pol in ['v', 'h']:
            name   = f'btemp_89.0{pol}'
            name_a = f'btemp_89.0a{pol}'
            name_b = f'btemp_89.0b{pol}'

            lons_a, lats_a = ds[name_a].area.get_lonlats()
            lons_b, lats_b = ds[name_b].area.get_lonlats()

            shape = np.array(ds[name_a].values.shape) * np.array([1, 2])
            bt89 = np.empty(shape, dtype=ds[name_a].values.dtype)
            lats = np.empty_like(bt89)
            lons = np.empty_like(bt89)

            bt89[:, 0::2] = ds[name_a].values
            bt89[:, 1::2] = ds[name_b].values
            lons[:, 0::2] = lons_a.compute()
            lons[:, 1::2] = lons_b.compute()
            lats[:, 0::2] = lats_a.compute()
            lats[:, 1::2] = lats_b.compute()

            attrs = dict(ds[name_a].attrs)
            attrs['name'] = name
            attrs['area'] = SwathDefinition(
                lons=xr.DataArray(lons, dims=['y', 'x']),
                lats=xr.DataArray(lats, dims=['y', 'x']),
            )
            out[name] = xr.DataArray(bt89, attrs=attrs)

        return xr.Dataset(out)

    def _get_target_subarea(self, scn, reference_channel='btemp_36.5v', margin=None):
        """
        Compute a small AreaDefinition covering just the swath's footprint
        within target_grid, instead of resampling onto the full grid. 
 
        Returns (sub_area, row0, col0) where row0/col0 are the sub-area's
        offset within the full target_grid (needed to place results back).
        """
        if margin is None:
            margin = self.SUBAREA_MARGIN

        lons, lats = scn[reference_channel].area.get_lonlats()
        lons, lats = np.asarray(lons), np.asarray(lats)

        proj = pyproj.Proj(self.target_grid.proj_dict)
        x,y = proj(lons, lats)

        margin_m = margin * 1000  # km -> m
        x_min, x_max = np.nanmin(x) - margin_m, np.nanmax(x) + margin_m
        y_min, y_max = np.nanmin(y) - margin_m, np.nanmax(y) + margin_m

        ext = self.target_grid.area_extent #[x_min, y_min, x_max, y_max]
        px = (ext[2]-ext[0]) / self.target_grid.width
        py = (ext[3]-ext[1]) / self.target_grid.height

        col0 = max(int((x_min - ext[0]) / px), 0)
        col1 = min(int(np.ceil((x_max - ext[0]) / px)), self.target_grid.width)
        row0 = max(int((ext[3]-y_max) / py), 0)
        row1 = min(int(np.ceil((ext[3]-y_min) / py)), self.target_grid.height)

        # guard against a degenerate (zero-size) sub-area, e.g. swath barely
        if col1 <= col0 or row1 <= row0:
            col1, row1 = col0 + 1, row0 + 1
 
        sub_extent = [
            ext[0] + col0 * px,
            ext[3] - row1 * py,
            ext[0] + col1 * px,
            ext[3] - row0 * py,
        ]
        sub_area = AreaDefinition(
            'sub', 'sub', 'sub', self.target_grid.proj_dict,
            col1 - col0, row1 - row0, sub_extent
        )
        return sub_area, row0, col0

    def _resample_channel(self, ds_ch, beam_widths, target_grid, neighbours=30, nprocs=1,
                           fill_value=np.nan, reduce_data=True):
        """Resample one AMSR2 channel onto target_grid. kd_tree resamples in ECEF
        xyz space internally, so dateline-crossing swaths are handled correctly
        without any manual lon/lat -> x/y conversion."""
        beam_width = float(1000 * np.array(beam_widths[ds_ch.attrs['name']]).mean())
        sigma = utils.fwhm2sigma(beam_width)

        res = kd_tree.resample_gauss(
            ds_ch.area,
            ds_ch.values.ravel(),
            target_grid,
            radius_of_influence=2 * beam_width,
            sigmas=sigma,
            neighbours=neighbours,
            nprocs=nprocs,
            fill_value=fill_value,
            reduce_data=reduce_data,
        )
        res = xr.DataArray(res, dims=['y', 'x'])
        res.name = ds_ch.attrs['name']
        return res

    def resample(self):
        """Runs the resampling and stores the result in self.ds_combined."""
        scn = satpy.Scene(reader=['amsr2_l1b'], filenames=[self.amsr2_file])
        scn.load(self.CHS)

        sub_area, row0, col0 = self._get_target_subarea(scn)

        resampled = xr.merge(
            self._resample_channel(scn[ch], self.BEAM_WIDTHS, target_grid=sub_area) for ch in self.CHS
        )

        scn_89 = self._get_combined_89(scn)
        resampled_89 = xr.merge(
            self._resample_channel(scn_89[ch], self.COMBINED_89_BEAM_WIDTHS, target_grid=sub_area)
            for ch in self.COMBINED_89_CHS
        )

        ds_combined = xr.merge([resampled, resampled_89])
        ds_combined = ds_combined.drop_vars(
            ['btemp_89.0ah', 'btemp_89.0av', 'btemp_89.0bh', 'btemp_89.0bv'],
            errors='ignore'
        )

        ds_combined.attrs.update({
            'instrument_name': 'AMSR-2',
            'platform_name': 'GCOM-W',
            'institution': 'DMI',
            'creation_date': datetime.now().strftime("%Y-%m-%d"),
            'description': 'AMSR-2 L1b brightness temperatures resampled onto a fixed north-polar-stereographic grid.',
            'source_file': os.path.basename(self.amsr2_file),
            'crop_y0': int(row0),
            'crop_x0': int(col0),
            'full_shape': [int(self.target_grid.height), int(self.target_grid.width)],
        })

        self.ds_combined = ds_combined
        return ds_combined

    def save_resampled_ds(self):
        """Runs resample() if needed, then writes the result to output_dir."""
        if self.ds_combined is None:
            self.resample()

        os.makedirs(self.output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.amsr2_file))[0]
        final_path = os.path.join(self.output_dir, f"{base}_resampled.nc")
        tmp_path   = os.path.join(self.output_dir, f".{base}_resampled.nc.tmp")

        encoding = {
            v: {"least_significant_digit": 2, "zlib": True, "complevel": 6}
            for v in self.ds_combined.data_vars if 'btemp' in v
        }
        self.ds_combined.to_netcdf(path=tmp_path, encoding=encoding)
        self.ds_combined.close()

        os.replace(tmp_path, final_path)  # atomic on the same filesystem
 
        LOG.info(f"Saved: {final_path}")
        return final_path

        




        