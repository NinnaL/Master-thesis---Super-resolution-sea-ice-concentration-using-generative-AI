import cartopy

class NorthPolStere(cartopy.crs.Projection):
    def __init__(self):
        # see: http://www.spatialreference.org/ref/epsg/3413/
        proj4_params = {'proj': 'stere',
            'lat_0': 90.,
            'lon_0': -45,
            'lat_ts':70,
            'x_0': 0,
            'y_0': 0,
            'a':6378137,
            'b':298.257223563,
            'units': 'm',
            'datum':'WGS84',
            'no_defs': ''}

        super(NorthPolStere, self).__init__(proj4_params)

    @property
    def boundary(self):
        coords = ((self.x_limits[0], self.y_limits[0]),(self.x_limits[1], self.y_limits[0]),
                  (self.x_limits[1], self.y_limits[1]),(self.x_limits[0], self.y_limits[1]),
                  (self.x_limits[0], self.y_limits[0]))

        return cartopy.crs.sgeom.Polygon(coords).exterior

    @property
    def threshold(self):
        return 1e5

    @property
    def x_limits(self):
        return (-4000000,4000000)

    @property
    def y_limits(self):
        return (-5000000, 3000000)
    

class SouthPolStere(cartopy.crs.Projection):
    """
    A class for the Southern Hemisphere polar stereographic projection.

    This projection is based on EPSG:3976 (WGS 84 / NSIDC Sea Ice
    Polar Stereographic South).
    """
    def __init__(self):
        # See: https://epsg.io/3976
        proj4_params = {
            'proj': 'stere',
            'lat_0': -90.,
            'lon_0': 0,
            'lat_ts': -70,
            'x_0': 0,
            'y_0': 0,
            'a': 6378137,
            'rf': 298.257223563, # WGS84 inverse flattening
            'units': 'm',
            'datum': 'WGS84',
            'no_defs': ''
        }

        super(SouthPolStere, self).__init__(proj4_params)

    @property
    def boundary(self):
        """Returns the rectangular boundary of the projection."""
        coords = ((self.x_limits[0], self.y_limits[0]),
                  (self.x_limits[1], self.y_limits[0]),
                  (self.x_limits[1], self.y_limits[1]),
                  (self.x_limits[0], self.y_limits[1]),
                  (self.x_limits[0], self.y_limits[0]))

        return cartopy.crs.sgeom.Polygon(coords).exterior

    @property
    def threshold(self):
        """The resolution threshold for the projection (in meters)."""
        return 1e5

    @property
    def x_limits(self):
        """The x-axis limits for the projection (in meters)."""
        return (-4200000, 4200000)

    @property
    def y_limits(self):
        """The y-axis limits for the projection (in meters)."""
        return (-4200000, 4200000)