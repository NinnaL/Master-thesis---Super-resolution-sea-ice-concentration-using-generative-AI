import os
import glob
import zarr
import dateutil
import shutil
import h5py
import xarray as xr
import numpy as np
import pyresample
import multiprocessing
from datetime import datetime, timedelta
import pyproj
from pyproj import Transformer
from pyresample.geometry import AreaDefinition

import warnings
warnings.filterwarnings('ignore', message='Possible more than 30 neighbours', category=UserWarning)
warnings.filterwarnings('ignore', message='Possible more than 8 neighbours', category=UserWarning)
warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

### CONFIG ###
OUTPUT_DIR = '/dmidata/projects/asip-cms/ninna_msc/validation'
INPUT_DIR = '/dmidata/projects/asip-cms/ninna_msc/output_mosaic'

LANDSAT_DIR = '/dmidata/projects/asip-cms/code/my_dataset_paper/landsat_sic'
ASIP_L3_DIR = '/dmidata/projects/asip-cms/reproc/mosaics/level3_0500m_v1'
OSI_SAF_DIR = '/dmidata/projects/asip-cms/SIC/OSISAF_458_CDR/v3p0'
OSI_SAF_ICDR_DIR = '/dmidata/projects/asip-cms/SIC/OSISAF_458_ICDR'
NSIDC_DIR = '/dmidata/projects/asip-cms/nsidc_sic_cdr'
NT2_DIR = '/dmidata/projects/asip-cms/code/my_dataset_paper/nt2'
MPD2_DIR = '/dmidata/projects/asip-cms/code/my_dataset_paper/mpd2'

# Set values
RADIUS_OF_INF = 30000  # 30 km radius of influence for resampling on a 25 km grid
SIGMA = pyresample.utils.fwhm2sigma(RADIUS_OF_INF)  # Convert FWHM to sigma for Gaussian resampling
FILL_VALUE = np.nan  # Use NaN for areas with no data

# Spatial subsetting to improve performance
xmin, xmax, ymin, ymax = 1500, 13500, 7500, 20000 

def transform_points(x, y, fromEPSG, toEPSG):
    # Function to transform coordinates from one EPSG to another using pyproj
    transformer = Transformer.from_crs(pyproj.CRS(f'EPSG:{fromEPSG}'), pyproj.CRS(f'EPSG:{toEPSG}'), always_xy=True)
    x, y = transformer.transform(x, y)

    return x, y

# Generate lat/lon grid for MPD2 data (which is in polstere projection) [source for grid definition: NSIDC website on polstere proj]
x = np.linspace(-3850000, 3750000, 608)
y = np.linspace(-5350000, 5850000, 896)
X, Y = np.meshgrid(x, y)
mpd2_lon, mpd2_lat = transform_points(X, Y, fromEPSG=3413, toEPSG=4326)

# Get common 25 km polstere grid definition from one ASIP product
asip_l3_file = '/dmidata/projects/asip-cms/reproc/mosaics/level3_0500m_v1/dmi_asip_seaice_mosaic_arc_l3_20200101.nc'
ds = xr.open_dataset(asip_l3_file)
lat = ds['lat'][ymin:ymax, xmin:xmax].values
lon = ds['lon'][ymin:ymax, xmin:xmax].values
ds.close()
l3_swath_def = pyresample.geometry.SwathDefinition(lons=lon, lats=lat)
new_lat = lat[::50, ::50] # 500m*50 = 25km
new_lon = lon[::50, ::50]
common_swath_def = pyresample.geometry.SwathDefinition(lons=new_lon, lats=new_lat)



def get_dt(nc):
    # Get datetime from filename
    return dateutil.parser.parse(os.path.basename(nc)[11:-3])

def process(nc, l3_swath_def, common_swath_def):
    # Function that loadsthe ASIP L3, relevant OSI-458, NSIDC, NT2, Landsat, MPD2 data, resamples them to the common grid and saves everything to a Zarr file for each date. 
    # This is the function that will be called in parallel for each file.
    DATE = get_dt(nc)
    print(f"Processing {DATE}")

    # Fetching ASIP L3 data
    if DATE.year in [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]:
        asip_path = f'{ASIP_L3_DIR}/dmi_asip_seaice_mosaic_arc_l3_{DATE.strftime("%Y%m%d")}.nc'
        if os.path.exists(asip_path):
            ds = xr.open_dataset(asip_path)
            asip_sic = ds['sic'][0, ymin:ymax, xmin:xmax].values
            ds.close()

            asip_sic = pyresample.kd_tree.resample_gauss(l3_swath_def, asip_sic, common_swath_def, radius_of_influence=RADIUS_OF_INF, sigmas=SIGMA, fill_value=FILL_VALUE)
        else:
            print(f'No ASIP L3 data for date {DATE}.')
    
    # Fetching and resampling OSI-458 CDR data
    if DATE.year in [2014, 2015, 2016, 2017, 2018, 2019, 2020]: 
        osisaf_path = f'{OSI_SAF_DIR}/{DATE.year}/{DATE.month:02}/ice_conc_nh_ease2-250_cdr-v3p0-amsr_{DATE.strftime("%Y%m%d")}1200.nc'
    elif DATE.year in [2021, 2022, 2023, 2024]:
        osisaf_path = f'{OSI_SAF_ICDR_DIR}/{DATE.month:02}/ice_conc_nh_ease2-250_icdr-amsr2_{DATE.strftime("%Y%m%d")}1200.nc'
        if not os.path.exists(osisaf_path):
            osisaf_path = f'{OSI_SAF_ICDR_DIR}/{DATE.month:02}/ice_conc_nh_ease2-250_dm1-amsr2_{DATE.strftime("%Y%m%d")}1200.nc'

    if os.path.exists(osisaf_path):
        ds = xr.open_dataset(osisaf_path)
        osisaf_sic = ds['ice_conc'].values[0]
        osisaf_sic_raw = ds['raw_ice_conc_values'].values[0]
        combined_osisaf_sic = np.where(np.logical_and(~np.isnan(osisaf_sic_raw), osisaf_sic_raw <= 100), osisaf_sic_raw, osisaf_sic)
        osisaf_lon = ds['lon'].values
        osisaf_lat = ds['lat'].values
        osisaf_swath_def = pyresample.geometry.SwathDefinition(lons=osisaf_lon, lats=osisaf_lat)
        osisaf_sic = pyresample.kd_tree.resample_nearest(osisaf_swath_def, osisaf_sic, common_swath_def, radius_of_influence=RADIUS_OF_INF, fill_value=FILL_VALUE)

        # osi saf raw
        osisaf_sic_raw = pyresample.kd_tree.resample_nearest(osisaf_swath_def, combined_osisaf_sic, common_swath_def, radius_of_influence=RADIUS_OF_INF, fill_value=FILL_VALUE)
    else:
        print(f'No OSI-458 data for date {DATE}.')
        osisaf_sic = np.full_like(asip_sic, np.nan)
        osisaf_sic_raw = np.full_like(asip_sic, np.nan)

    # Fecthing and resampling NSIDC data
    nsidc_path = f'{NSIDC_DIR}/sic_psn25_{DATE.strftime("%Y%m%d")}_F17_v05r00.nc'
    if os.path.exists(nsidc_path):
        ds = xr.open_dataset(nsidc_path)
        nsidc_sic = ds['cdr_seaice_conc'].values[0]*100
        x = ds['x'].values
        y = ds['y'].values
        x, y = np.meshgrid(x, y)
        x, y = transform_points(x, y, 3411, 4326)
        nsidc_swath_def = pyresample.geometry.SwathDefinition(lons=x, lats=y)
        nsidc_sic = pyresample.kd_tree.resample_nearest(nsidc_swath_def, nsidc_sic, common_swath_def, radius_of_influence=RADIUS_OF_INF, fill_value=FILL_VALUE)
    else:
        print(f'No NSIDC data for date {DATE}.')
        nsidc_sic = np.full_like(asip_sic, np.nan)

    # Fecthing and resampling NT2 data
    nt2_path = f'{NT2_DIR}/{DATE.year}/{DATE.month:02}/AMSR_U2_L3_SeaIce25km_B04_{DATE.strftime("%Y%m%d")}.he5'
    if os.path.exists(nt2_path):
        with h5py.File(nt2_path, "r") as f:
            nt2_lon = f['HDFEOS']['GRIDS']['NpPolarGrid25km']['lon'][:] 
            nt2_lat = f['HDFEOS']['GRIDS']['NpPolarGrid25km']['lat'][:] 
            nt2_sic = f['HDFEOS']['GRIDS']['NpPolarGrid25km']['Data Fields']['SI_25km_NH_ICECON_DAY'][:].astype('float32')
            nt2_sic[nt2_sic > 100] = np.nan
        nt2_swath_def = pyresample.geometry.SwathDefinition(lons=nt2_lon, lats=nt2_lat)
        nt2_sic = pyresample.kd_tree.resample_nearest(nt2_swath_def, nt2_sic, common_swath_def, radius_of_influence=RADIUS_OF_INF, fill_value=FILL_VALUE)
    else:
        print(f'No NT2 data for date {DATE}.')
        nt2_sic = np.full_like(asip_sic, np.nan)

    # Fetching and resampling Landsat data
    if DATE.year in [2020, 2021, 2022]:
        landsat_sic = []
        for region in landsat_ncs:
            ds = xr.open_dataset(region)
            time_idxs = np.where((ds['time'].values.astype('datetime64[D]') == np.datetime64(DATE, 'D')))
            landsat_sic.append(ds['sea_ice_concentration'][time_idxs].values)
        landsat_sic = [tmp for tmp in landsat_sic if not tmp.shape[0] == 0] # removing regions with no valid Landsat imagery

        if len(landsat_sic) > 0: 
            landsat_sic = np.vstack(landsat_sic)
        else: 
            print(f'No valid Landsat imagery for date {DATE}.')
            landsat_sic = np.full_like(asip_sic, np.nan)

        if not np.isnan(landsat_sic).all():
            landsat_sic = np.nanmean(landsat_sic, axis=0) # daily mean (IF more than one valid Landsat for each pixel, else its just the one)
            # resampling to OSI SAF grid
            l3_swath_def = pyresample.geometry.SwathDefinition(lons=lon, lats=lat)
            landsat_swath_def = pyresample.geometry.SwathDefinition(lons=ds['lon'].values, lats=ds['lat'].values)
            landsat_sic = pyresample.kd_tree.resample_gauss(landsat_swath_def, landsat_sic, common_swath_def, radius_of_influence=RADIUS_OF_INF, sigmas=SIGMA, fill_value=FILL_VALUE)
        else: 
            print(f'No valid Landsat imagery for date {DATE}.')
            landsat_sic = np.full_like(asip_sic, np.nan)
    else:
        print(f'No valid Landsat imagery for date {DATE}.')
        landsat_sic = np.full_like(asip_sic, np.nan)

    # Fetching and resampling MPD2 data
    if DATE.year in [2020, 2021, 2022]:
        mpd2_path = f'{MPD2_DIR}/{DATE.year}/{DATE.strftime("%Y%m%d")}_1daycomp_mpd2.nc'
        if os.path.exists(mpd2_path):
            ds = xr.open_dataset(mpd2_path)
            mpf = ds['mpf'].values
            mpd2_swath_def = pyresample.geometry.SwathDefinition(lons=mpd2_lon, lats=mpd2_lat)
            mpf = pyresample.kd_tree.resample_nearest(mpd2_swath_def, mpf, common_swath_def, radius_of_influence=15000, fill_value=FILL_VALUE)
        else:
            print(f'No MPD2 data for date {DATE}.')
            mpf = np.full_like(asip_sic, np.nan)
    else:
        mpf = np.full_like(asip_sic, np.nan)

    # Fetching and resampling NINNA data
    if DATE.year in [2020, 2021, 2022]:
        data_path = f'{INPUT_DIR}/{DATE.year}/{DATE.month:02d}/SIC_mosaic_{DATE.strftime("%Y%m%d")}.nc'
        if os.path.exists(data_path):
            ds = xr.open_dataset(data_path)
            ninna_sic = ds['SIC'].values
            ninna_lon = ds['lon'].values
            ninna_lat = ds['lat'].values
            ds.close()

            ninna_swath_def = pyresample.geometry.SwathDefinition(lons=ninna_lon, lats=ninna_lat)
            ninna_sic = pyresample.kd_tree.resample_nearest(ninna_swath_def, ninna_sic, common_swath_def, radius_of_influence=RADIUS_OF_INF, fill_value=FILL_VALUE)

        else:
            print(f'No NINNA data for date {DATE}.')
            ninna_sic = np.full_like(asip_sic, np.nan)
    else:
        ninna_sic = np.full_like(asip_sic, np.nan)

    if True:
        # Save to Zarr
        # Create root group
        zarr_path = f'{OUTPUT_DIR}/{DATE.strftime("%Y%m%d")}.zarr'
        if os.path.exists(zarr_path):
            shutil.rmtree(zarr_path)
        store = zarr.DirectoryStore(zarr_path)
        root = zarr.group(store=store)

        # Create groups and datasets
        datasets = root.create_group('datasets')

        # Store arrays with compression
        datasets.create_dataset('asip_sic', 
                            data=asip_sic,
                            chunks=(500, 500),
                            compressor=zarr.Blosc(cname='zstd'))
        
        #datasets.create_dataset('asip_unc', 
        #                    data=l3_unc,
        #                    chunks=(500, 500),
        #                    compressor=zarr.Blosc(cname='zstd'))

        datasets.create_dataset('mpf',
                            data=mpf, 
                            chunks=(500, 500),
                            compressor=zarr.Blosc(cname='zstd'))

        datasets.create_dataset('landsat',
                            data=landsat_sic, 
                            chunks=(500, 500),
                            compressor=zarr.Blosc(cname='zstd'))

        datasets.create_dataset('osisaf',
                            data=osisaf_sic,
                            chunks=(500, 500),
                            compressor=zarr.Blosc(cname='zstd'))

        datasets.create_dataset('osisaf_raw',
                                    data=osisaf_sic_raw,
                                    chunks=(500, 500),
                                    compressor=zarr.Blosc(cname='zstd'))
        
        datasets.create_dataset('nsidc',
                            data=nsidc_sic,
                            chunks=(500, 500),
                            compressor=zarr.Blosc(cname='zstd'))
        
        datasets.create_dataset('nt2',
                            data=nt2_sic,
                            chunks=(500, 500),
                            compressor=zarr.Blosc(cname='zstd'))
        
        datasets.create_dataset('NINNA',
                            data=ninna_sic,
                            chunks=(500, 500),
                            compressor=zarr.Blosc(cname='zstd'))
        
    #return asip_sic, landsat_sic, osisaf_sic
    
def multiprocess(l3_ncs, l3_swath_def, common_swath_def, n_processes=8):
    with multiprocessing.Pool(n_processes) as pool:
        pool.starmap(process, [(nc, l3_swath_def, common_swath_def) for nc in l3_ncs])



if __name__ == '__main__':
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2021, 12, 31)  

    all_ninna_sic_files = sorted(glob.glob(f'{INPUT_DIR}/*/*/SIC_mosaic_*.nc'))
    landsat_ncs = sorted(glob.glob(f'{LANDSAT_DIR}/*.nc'))
    
    ncs = [nc for nc in all_ninna_sic_files if start_date <= get_dt(nc) <= end_date and not os.path.exists(f'{OUTPUT_DIR}/{get_dt(nc).strftime("%Y%m%d")}.zarr')]

    print(len(ncs))

    multiprocess(ncs, l3_swath_def, common_swath_def, n_processes=24)
    

