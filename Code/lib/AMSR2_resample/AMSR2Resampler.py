import warnings

from requests import get
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import time
import h5py
import logging
import satpy
import numpy as np
import xarray as xr
import numpy as np
from osgeo import ogr, osr
from pyresample import kd_tree, utils
from pyresample.geometry import SwathDefinition
from datetime import datetime, timedelta
import pandas as pd

os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

class AMSR2Resampler():
    """
    Class to resample AMSR2 data to a specified grid. The class takes in a list of AMSR2 files, an output directory, a target grid definition, and an optional hemisphere parameter (default is 'N' for Northern Hemisphere). The resampling process involves reading the AMSR2 data, defining the target grid, and using the pyresample library to perform the resampling. The resampled data is then saved to the specified output directory.""
    
    Elements denoted TWU has been copied from https://gitlab.dmi.dk/remote-sensing/twu/asip/asip_opr/-/blob/dev/l2_prod/amsr2_matchup/ResampleAMSR2.py?ref_type=heads 
    """
    def __init__(self, amsr2_file, output_dir, target_grid, hemisphere = 'N'):

        self.amsr2_file = amsr2_file
        self.beam_widths = {'btemp_6.9h': [35, 62],
                            'btemp_6.9v': [35, 62],
                            'btemp_7.3h': [35, 62],
                            'btemp_7.3v': [35, 62],
                            'btemp_10.7h': [24, 42],
                            'btemp_10.7v': [24, 42],
                            'btemp_18.7h': [24, 42],
                            'btemp_18.7v': [24, 42],
                            'btemp_23.8h': [15, 26],
                            'btemp_23.8v': [15, 26],
                            'btemp_36.5h': [7, 12],
                            'btemp_36.5v': [7, 12],
                            'btemp_89.0ah': [3, 5],
                            'btemp_89.0bh': [3, 5],
                            'btemp_89.0av': [3, 5],
                            'btemp_89.0bv': [3, 5]}
        self.chs = list(self.beam_widths.keys())
        self.combined_89_beam_widths = {'btemp_89.0h': [3, 5],
                                        'btemp_89.0v': [3, 5]}
        self.combined_89_chs = list(self.combined_89_beam_widths.keys())

        self.lat_limits = [40, 90] if hemisphere == 'N' else ([-90, -40] if hemisphere == 'S' else None)
        self.ESPG = 3411 if hemisphere == 'N' else (3412 if hemisphere == 'S' else 6933) # Arctic Polar Stereographic, Antarctic Polar Stereographic, EASE-Grid 2.0 Global

        self.amsr2_file = self.filter_corrupted_h5s(amsr2_file)

        self.output_dir = output_dir
        self.target_grid = target_grid

    def filter_corrupted_h5s(self, amsr2_file):
        """
        Occasionally AMSR2 L1b .h5-files will contain invalid data. 
        Assuming invalid pixels are given the value 2¹⁶, all AMSR2 swaths with invalid data north of 40 degrees North are discarded.  
        From TWU
        """

        with h5py.File(amsr2_file, locking=False) as f:
            if not (f['Brightness Temperature (89.0GHz-A,H)'][f['Latitude of Observation Point for 89A'][:] > 40] == 2**16 - 1).any():
                return amsr2_file
            else:
                print(f'Discarded: {amsr2_file}')

        return None

    def _mask_to_lat_limits(self, ds_ch):
        lons, lats = ds_ch.area.get_lonlats()
        lats_np = lats.compute() if hasattr(lats, 'compute') else lats
        mask = (lats_np < self.lat_limits[0]) | (lats_np > self.lat_limits[1])
        masked = ds_ch.values.copy()
        masked[mask] = np.nan
        return masked 
    
    def get_combined_89(self, ds):
        """
        Takes a complete AMSR2 satpy dataset and returns a new ds, where the 89.0Pa and 89.0Pb have been combined.
        Taken directly from John Lavelle's old AMSR2 matchup scripts. 
        From TWU
        """
        def get():
            for pol in ['v', 'h']:
                name = 'btemp_89.0{p}'.format(p=pol)
                name_a = 'btemp_89.0a{p}'.format(p=pol)
                name_b = 'btemp_89.0b{p}'.format(p=pol)

                lons_a, lats_a = ds[name_a].area.get_lonlats()
                lons_b, lats_b = ds[name_b].area.get_lonlats()

                da89size = np.array(ds[name_a].values.shape) * np.array([1, 2])
                bt89 = np.empty(da89size, dtype=ds[name_a].values.dtype)
                lats = np.empty_like(bt89)
                lons = np.empty_like(bt89)

                bt89[:, 0::2] = ds[name_a].values
                bt89[:, 1::2] = ds[name_b].values
                lons[:, 0::2] = lons_a.compute()
                lons[:, 1::2] = lons_b.compute()
                lats[:, 0::2] = lats_a.compute()
                lats[:, 1::2] = lats_b.compute()

                lons = xr.DataArray(lons, dims=['y', 'x'])
                lats = xr.DataArray(lats, dims=['y', 'x'])

                attrs = ds[name_a].attrs
                attrs['shape'] = bt89.shape
                attrs['name'] = name
                attrs['area'] = SwathDefinition(lons=lons, lats=lats)

                yield name, xr.DataArray(bt89, attrs=attrs)#, dims=('y', 'x'))

        return xr.Dataset({name: da for name, da in get()})

    def resample_amsr2(self, ds_ch, beam_widths, neighbours=30, nprocs=1, fill_value=None, reduce_data=False):
        """
        Resamples a single-channeled AMSR2 satpy dataset to the grid defined by target_swath_def.
        From TWU
        """
        beam_width = float(1000 * np.array(beam_widths[ds_ch.attrs['name']]).mean())
        sigma = utils.fwhm2sigma(beam_width)
        res = kd_tree.resample_gauss(
            ds_ch.area, 
            self._mask_to_lat_limits(ds_ch).ravel(), 
            self.target_grid, 
            radius_of_influence=2*beam_width, 
            sigmas=sigma,
            neighbours=neighbours, 
            nprocs=nprocs, 
            fill_value=fill_value if fill_value is not None else np.nan, 
            reduce_data=reduce_data)

        res = xr.DataArray(res)
        res.name = ds_ch.attrs['name']

        return res
    
    def set_attrs(self, ds):
        """
        Sets the attributes of an xarray dataset.
        """
        ds.attrs['instrument_name'] = "AMSR-2"
        ds.attrs['platform_name'] = "GCOM-W"
        ds.attrs['institution'] = "DMI"
        ds.attrs['creation_date'] = datetime.now().strftime("%Y-%m-%d")
        ds.attrs['contact'] = "nili@dmi.dk"
        ds.attrs['description'] = "AMSR-2 Level 1b brightness temperatures resampled onto an upsampled ~2 km grid. "
        ds.attrs['AMSR2_swaths'] = os.path.basename(self.amsr2_file)

        return ds
    
    def get_encodings(self, ds):
        """
        Sets the encodings of an xarray dataset prior to the netCDF export.
        """
        encoding_Tb = {"least_significant_digit": 2, "zlib": True, "complevel": 6}
        encodings = {v: encoding_Tb for v in ds.data_vars if 'btemp' in v}

        return encodings
    
    def save_resampled_ds(self):
        """
        Saves the resampled xarray dataset to a netCDF file.
        """
        
        basename = os.path.basename(self.amsr2_file).replace('.h5', '_resampled.nc')
        output_path = os.path.join(self.output_dir, basename)

        # Load AMSR2 swath with satpy
        amsr2_scn = satpy.Scene(reader='amsr2_l1b', filenames=[self.amsr2_file])
        amsr2_scn.load(self.chs)
        amsr2_resampled = xr.merge(self.resample_amsr2(amsr2_scn[ch], self.beam_widths) for ch in self.chs)

        amsr2_89_scn = self.get_combined_89(amsr2_scn)
        amsr2_89_resampled = xr.merge(self.resample_amsr2(amsr2_89_scn[ch], self.combined_89_beam_widths) for ch in self.combined_89_chs)
        ds_combined = xr.merge([amsr2_resampled, amsr2_89_resampled])

        ds_combined = self.set_attrs(ds_combined)

        ds_combined = ds_combined.drop(['btemp_89.0ah', 'btemp_89.0av', 'btemp_89.0bh', 'btemp_89.0bv'])

        ds_combined.to_netcdf(path=output_path, encoding=self.get_encodings(ds_combined))
        ds_combined.close()




        




        