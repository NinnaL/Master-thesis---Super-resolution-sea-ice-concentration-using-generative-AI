# grid_config.py
from pyresample import geometry

target_area = geometry.AreaDefinition(
    'nps_2km', 'North Polar Stereographic 2km', 'nps_2km',
    {'proj': 'stere', 'lat_0': 90, 'lon_0': -45, 'lat_ts': 70,
     'a': 6378137, 'b': 6356752.3142, 'units': 'm'},
    width=4510, height=4510,
    area_extent=[-4511000, -4511000, 4511000, 4511000]
)

lon_grid, lat_grid = target_area.get_lonlats()