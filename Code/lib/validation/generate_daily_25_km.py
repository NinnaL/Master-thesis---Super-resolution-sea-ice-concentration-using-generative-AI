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

os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

def transform_points(x, y, fromEPSG, toEPSG):
    transformer = Transformer.from_crs(pyproj.CRS(f'EPSG:{fromEPSG}'), pyproj.CRS(f'EPSG:{toEPSG}'), always_xy=True)
    x, y = transformer.transform(x, y)

    return x, y

# mpd2 grid
x = np.linspace(-3850000, 3750000, 608) # from NSIDC website on polstere proj
y = np.linspace(-5350000, 5850000, 896)
X, Y = np.meshgrid(x, y)
mpd2_lon, mpd2_lat = transform_points(X, Y, fromEPSG=3413, toEPSG=4326)

l3_ncs = sorted(glob.glob('/dmidata/projects/asip-cms/reproc/mosaics/level3_0500m_v1/*.nc'))
landsat_ncs = glob.glob('landsat_sic/*.nc')
xmin, xmax, ymin, ymax = 1500, 13500, 7500, 20000 # Setup spatial subsetting to improve performance

# Get common 25 km polstere grid definition from ASIP product
ds = xr.open_dataset(l3_ncs[0])
lat = ds['lat'][ymin:ymax, xmin:xmax].values
lon = ds['lon'][ymin:ymax, xmin:xmax].values
ds.close()
l3_swath_def = pyresample.geometry.SwathDefinition(lons=lon, lats=lat)
new_lat = lat[::50, ::50] # 500m*50 = 25km
new_lon = lon[::50, ::50]
common_swath_def = pyresample.geometry.SwathDefinition(lons=new_lon, lats=new_lat)


def get_nc_dt(nc):
    return dateutil.parser.parse(os.path.basename(nc)[30:-3])

def process(l3_nc, l3_swath_def, common_swath_def):
    l3_date = get_nc_dt(l3_nc)
    print(f"Processing {l3_date}")

    # Read L3 data
    ds = xr.open_dataset(l3_nc)
    l3_sic = ds['sic'][0, ymin:ymax, xmin:xmax].values
    ds.close()

    l3_sic = pyresample.kd_tree.resample_gauss(l3_swath_def, l3_sic, common_swath_def, radius_of_influence=30000, sigmas=pyresample.utils.fwhm2sigma(30000), fill_value=np.nan)
    #l3_unc = pyresample.kd_tree.resample_gauss(l3_swath_def, l3_unc, common_swath_def, radius_of_influence=30000, sigmas=pyresample.utils.fwhm2sigma(30000), fill_value=np.nan)
    
    # Fetching and resampling OSI-458 CDR data
    if l3_date.year in [2014, 2015, 2016, 2017, 2018, 2019, 2020]: 
        osisaf_path = f'/dmidata/projects/asip-cms/SIC/OSISAF_458_CDR/v3p0/{l3_date.year}/{l3_date.month:02}/ice_conc_nh_ease2-250_cdr-v3p0-amsr_{l3_date.strftime("%Y%m%d")}1200.nc'
    elif l3_date.year in [2021, 2022, 2023, 2024]:
        osisaf_path = f'/dmidata/projects/asip-cms/SIC/OSISAF_458_ICDR/{l3_date.month:02}/ice_conc_nh_ease2-250_icdr-amsr2_{l3_date.strftime("%Y%m%d")}1200.nc'
        if not os.path.exists(osisaf_path):
            osisaf_path = f'/dmidata/projects/asip-cms/SIC/OSISAF_458_ICDR/{l3_date.month:02}/ice_conc_nh_ease2-250_dm1-amsr2_{l3_date.strftime("%Y%m%d")}1200.nc'

    if os.path.exists(osisaf_path):
        ds = xr.open_dataset(osisaf_path)
        osisaf_sic = ds['ice_conc'].values[0]
        osisaf_sic_raw = ds['raw_ice_conc_values'].values[0]
        combined_osisaf_sic = np.where(np.logical_and(~np.isnan(osisaf_sic_raw), osisaf_sic_raw <= 100), osisaf_sic_raw, osisaf_sic)
        osisaf_lon = ds['lon'].values
        osisaf_lat = ds['lat'].values
        osisaf_swath_def = pyresample.geometry.SwathDefinition(lons=osisaf_lon, lats=osisaf_lat)
        osisaf_sic = pyresample.kd_tree.resample_nearest(osisaf_swath_def, osisaf_sic, common_swath_def, radius_of_influence=30000, fill_value=np.nan)

        # osi saf raw
        osisaf_sic_raw = pyresample.kd_tree.resample_nearest(osisaf_swath_def, combined_osisaf_sic, common_swath_def, radius_of_influence=30000, fill_value=np.nan)
    else:
        print(f'No OSI-458 data for date {l3_date}.')
        osisaf_sic = np.full_like(l3_sic, np.nan)
        osisaf_sic_raw = np.full_like(l3_sic, np.nan)

    # Fecthing and resampling NSIDC data
    nsidc_path = f'/dmidata/projects/asip-cms/nsidc_sic_cdr/sic_psn25_{l3_date.strftime("%Y%m%d")}_F17_v05r00.nc'
    if os.path.exists(nsidc_path):
        ds = xr.open_dataset(nsidc_path)
        nsidc_sic = ds['cdr_seaice_conc'].values[0]*100
        x = ds['x'].values
        y = ds['y'].values
        x, y = np.meshgrid(x, y)
        x, y = transform_points(x, y, 3411, 4326)
        nsidc_swath_def = pyresample.geometry.SwathDefinition(lons=x, lats=y)
        nsidc_sic = pyresample.kd_tree.resample_nearest(nsidc_swath_def, nsidc_sic, common_swath_def, radius_of_influence=30000, fill_value=np.nan)
    else:
        print(f'No NSIDC data for date {l3_date}.')
        nsidc_sic = np.full_like(l3_sic, np.nan)

    # Fecthing and resampling NT2 data
    nt2_path = f'/dmidata/projects/asip-cms/code/my_dataset_paper/nt2/{l3_date.year}/{l3_date.month:02}/AMSR_U2_L3_SeaIce25km_B04_{l3_date.strftime("%Y%m%d")}.he5'
    if os.path.exists(nt2_path):
        with h5py.File(nt2_path, "r") as f:
            nt2_lon = f['HDFEOS']['GRIDS']['NpPolarGrid25km']['lon'][:] 
            nt2_lat = f['HDFEOS']['GRIDS']['NpPolarGrid25km']['lat'][:] 
            nt2_sic = f['HDFEOS']['GRIDS']['NpPolarGrid25km']['Data Fields']['SI_25km_NH_ICECON_DAY'][:].astype('float32')
            nt2_sic[nt2_sic > 100] = np.nan
        nt2_swath_def = pyresample.geometry.SwathDefinition(lons=nt2_lon, lats=nt2_lat)
        nt2_sic = pyresample.kd_tree.resample_nearest(nt2_swath_def, nt2_sic, common_swath_def, radius_of_influence=30000, fill_value=np.nan)
    else:
        print(f'No NT2 data for date {l3_date}.')
        nt2_sic = np.full_like(l3_sic, np.nan)

    # Fetching and resampling Landsat data
    if l3_date.year in [2020, 2021, 2022]:
        landsat_sic = []
        for region in landsat_ncs:
            ds = xr.open_dataset(region)
            time_idxs = np.where((ds['time'].values.astype('datetime64[D]') == np.datetime64(l3_date, 'D')))
            landsat_sic.append(ds['sea_ice_concentration'][time_idxs].values)
        landsat_sic = [tmp for tmp in landsat_sic if not tmp.shape[0] == 0] # removing regions with no valid Landsat imagery

        if len(landsat_sic) > 0: 
            landsat_sic = np.vstack(landsat_sic)
        else: 
            print(f'No valid Landsat imagery for date {l3_date}.')
            landsat_sic = np.full_like(l3_sic, np.nan)

        if not np.isnan(landsat_sic).all():
            landsat_sic = np.nanmean(landsat_sic, axis=0) # daily mean (IF more than one valid Landsat for each pixel, else its just the one)
            # resampling to OSI SAF grid
            l3_swath_def = pyresample.geometry.SwathDefinition(lons=lon, lats=lat)
            landsat_swath_def = pyresample.geometry.SwathDefinition(lons=ds['lon'].values, lats=ds['lat'].values)
            landsat_sic = pyresample.kd_tree.resample_gauss(landsat_swath_def, landsat_sic, common_swath_def, radius_of_influence=30000, sigmas=pyresample.utils.fwhm2sigma(30000), fill_value=np.nan)
        else: 
            print(f'No valid Landsat imagery for date {l3_date}.')
            landsat_sic = np.full_like(l3_sic, np.nan)
    else:
        print(f'No valid Landsat imagery for date {l3_date}.')
        landsat_sic = np.full_like(l3_sic, np.nan)

    # Fetching and resampling MPD2 data
    if l3_date.year in [2020, 2021, 2022]:
        mpd2_path = f'mpd2/{l3_date.year}/{l3_date.strftime("%Y%m%d")}_1daycomp_mpd2.nc'
        if os.path.exists(mpd2_path):
            ds = xr.open_dataset(mpd2_path)
            mpf = ds['mpf'].values
            mpd2_swath_def = pyresample.geometry.SwathDefinition(lons=mpd2_lon, lats=mpd2_lat)
            mpf = pyresample.kd_tree.resample_nearest(mpd2_swath_def, mpf, common_swath_def, radius_of_influence=15000, fill_value=np.nan)
        else:
            print(f'No MPD2 data for date {l3_date}.')
            mpf = np.full_like(l3_sic, np.nan)
    else:
        mpf = np.full_like(l3_sic, np.nan)


    if True:
        # Save to Zarr
        # Create root group
        zarr_path = f'cci_data/{l3_date.strftime("%Y%m%d")}.zarr'
        if os.path.exists(zarr_path):
            shutil.rmtree(zarr_path)
        store = zarr.DirectoryStore(zarr_path)
        root = zarr.group(store=store)

        # Create groups and datasets
        datasets = root.create_group('datasets')

        # Store arrays with compression
        datasets.create_dataset('asip_sic', 
                            data=l3_sic,
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
        
    #return l3_sic, landsat_sic, osisaf_sic
    
def multiprocess(l3_ncs, l3_swath_def, common_swath_def, n_processes=8):
    with multiprocessing.Pool(n_processes) as pool:
        pool.starmap(process, [(nc, l3_swath_def, common_swath_def) for nc in l3_ncs])



if __name__ == '__main__':
    start_date = datetime(2014, 10, 1)
    end_date = datetime(2024, 9, 30)  
    
    ncs = [nc for nc in l3_ncs if start_date <= get_nc_dt(nc) <= end_date and not os.path.exists(f'cci_data/{get_nc_dt(nc).strftime("%Y%m%d")}.zarr')]

    print(len(ncs))

    multiprocess(ncs, l3_swath_def, common_swath_def, n_processes=24)
    

