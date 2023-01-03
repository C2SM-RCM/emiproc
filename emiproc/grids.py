"""Emiproc Grids.

Classes handling different grids, namely the simulation grids and
grids used in different emissions inventories.
"""
from functools import cache, cached_property
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import geopandas as gpd
import pyproj
import math

from copy import deepcopy
from netCDF4 import Dataset
from shapely.geometry import Polygon, Point, box, LineString, MultiPolygon
from shapely.ops import split

WGS84 = 4326
WGS84_PROJECTED = 3857
LV95 = 2056  # EPSG:2056, swiss CRS
WGS84_NSIDC = 6933

# Radius of the earth
R_EARTH = 6371000  # m

class Grid:
    """Abstract base class for a grid.
    Derive your own grid implementation from this and make sure to provide
    an appropriate implementation of the required methods.
    As an example you can look at TNOGrid.
    """

    nx: int
    ny: int

    # The crs value as an integer
    crs: int = WGS84
    gdf: gpd.GeoDataFrame | None = None

    def __init__(self, name, projection):
        """
        Parameters
        ----------
        name : str
            name of the inventory
        projection : cartopy.crs.Projection
            Projection used for the inventory grid. Used to transform points to
            other coordinate systems.
        """
        self.name = name
        self.projection = projection

    def get_projection(self):
        """Returns a copy of the projection"""
        return self.projection

    def cell_corners(self, i, j):
        """Return the corners of the cell with indices (i,j).

        The points are ordered clockwise, starting in the top
        left:

        4.   1.
        ^    v
        3. < 2.

        Returns a tuple of arrays with shape (4,). The first
        tuple element are the x-coordinates of the corners,
        the second are the y-coordinates.

        The coordinates are in the projection of the grid, so
        to work with them you might have to transform them to
        the desired projection. For example, to be sure you're
        working with regular (lon, lat) coordinates:

        >>> corners = ccrs.PlateCarree().transform_points(
        ...    grid.get_projection(),
        ...    *grid.cell_corners(i,j)
        ...)

        The clunky return type is necessary because the corners
        are transformed after by cartopy.crs.CRS.transform_points.

        Parameters
        ----------
        i : int
        j : int

        Returns
        -------
        tuple(np.array(shape=(4,), dtype=float),
              np.array(shape=(4,), dtype=float))
            Arrays containing the x and y coordinates of the corners

        """
        raise NotImplementedError("Method not implemented")

    def lon_range(self):
        """Return an array containing all the longitudinal points on the grid.

        Returns
        -------
        np.array(shape=(nx,), dtype=float)
        """
        raise NotImplementedError("Method not implemented")

    def lat_range(self):
        """Return an array containing all the latitudinal points on the grid.

        Returns
        -------
        np.array(shape=(ny,), dtype=float)
        """
        raise NotImplementedError("Method not implemented")

    @cache
    def cells_as_polylist(self):
        return [
            Polygon(zip(*self.cell_corners(i, j)))
            for i in range(self.nx)
            for j in range(self.ny)
        ]


class LatLonNcGrid(Grid):
    """A regular grid with lat/lon values from a nc file.

    This is a copy of the tno grid basically, but reading a nc file.
    """

    def __init__(self, dataset_path, lat_name="clat", lon_name="clon", name="LatLon"):

        self.dataset_path = dataset_path

        ds = xr.load_dataset(dataset_path)

        self.lon_var = np.unique(ds[lon_name])
        self.lat_var = np.unique(ds[lat_name])

        self.nx = len(self.lon_var)
        self.ny = len(self.lat_var)

        # The lat/lon values are the cell-centers
        self.dx = (self.lon_var[-1] - self.lon_var[0]) / (self.nx - 1)
        self.dy = (self.lat_var[-1] - self.lat_var[0]) / (self.ny - 1)

        # Compute the cell corners
        x = self.lon_var
        y = self.lat_var
        dx2 = self.dx / 2
        dy2 = self.dy / 2

        self.cell_x = np.array([x + dx2, x + dx2, x - dx2, x - dx2])
        self.cell_y = np.array([y + dy2, y - dy2, y - dy2, y + dy2])

        super().__init__(name, ccrs.PlateCarree())

    def cell_corners(self, i, j):
        """Return the corners of the cell with indices (i,j).

        See also the docstring of Grid.cell_corners.

        Parameters
        ----------
        i : int
        j : int

        Returns
        -------
        tuple(np.array(shape=(4,), dtype=float),
              np.array(shape=(4,), dtype=float))
            Arrays containing the x and y coordinates of the corners
        """
        return self.cell_x[:, i], self.cell_y[:, j]

    def lon_range(self):
        """Return an array containing all the longitudinal points on the grid.

        Returns
        -------
        np.array(shape=(nx,), dtype=float)
        """
        return self.lon_var

    def lat_range(self):
        """Return an array containing all the latitudinal points on the grid.

        Returns
        -------
        np.array(shape=(ny,), dtype=float)
        """
        return self.lat_var


class TNOGrid(Grid):
    """Contains the grid from the TNO emission inventory
    This grid is defined as a standard lat/lon coordinate system.
    The gridpoints are at the center of the cell.
    """

    def __init__(self, dataset_path, name="TNO"):
        """Open the netcdf-dataset and read the relevant grid information.

        Parameters
        ----------
        dataset_path : str
        name : str, optional
        """
        self.dataset_path = dataset_path

        with Dataset(dataset_path) as dataset:
            self.lon_var = np.array(dataset["longitude"][:])
            self.lat_var = np.array(dataset["latitude"][:])

        self.nx = len(self.lon_var)
        self.ny = len(self.lat_var)

        # The lat/lon values are the cell-centers
        self.dx = (self.lon_var[-1] - self.lon_var[0]) / (self.nx - 1)
        self.dy = (self.lat_var[-1] - self.lat_var[0]) / (self.ny - 1)

        # Compute the cell corners
        x = self.lon_var
        y = self.lat_var
        dx2 = self.dx / 2
        dy2 = self.dy / 2

        self.cell_x = np.array([x + dx2, x + dx2, x - dx2, x - dx2])
        self.cell_y = np.array([y + dy2, y - dy2, y - dy2, y + dy2])

        super().__init__(name, ccrs.PlateCarree())

    def cell_corners(self, i, j):
        """Return the corners of the cell with indices (i,j).

        See also the docstring of Grid.cell_corners.

        Parameters
        ----------
        i : int
        j : int

        Returns
        -------
        tuple(np.array(shape=(4,), dtype=float),
              np.array(shape=(4,), dtype=float))
            Arrays containing the x and y coordinates of the corners
        """
        return self.cell_x[:, i], self.cell_y[:, j]

    def lon_range(self):
        """Return an array containing all the longitudinal points on the grid.

        Returns
        -------
        np.array(shape=(nx,), dtype=float)
        """
        return self.lon_var

    def lat_range(self):
        """Return an array containing all the latitudinal points on the grid.

        Returns
        -------
        np.array(shape=(ny,), dtype=float)
        """
        return self.lat_var


class EDGARGrid(Grid):
    """Contains the grid from the EDGAR emission inventory

    The grid is similar to the TNO grid.
    """

    def __init__(self, dataset_path, name="EDGAR"):
        """Open the netcdf-dataset and read the relevant grid information.

        Parameters
        ----------
        dataset_path : str
        name : str, optional
        """
        self.dataset_path = dataset_path

        with Dataset(dataset_path) as dataset:
            self.lon_var = np.array(dataset["lon"][:])
            self.lat_var = np.array(dataset["lat"][:])

        self.nx = len(self.lon_var)
        self.ny = len(self.lat_var)

        # The lat/lon values are the cell-centers
        self.dx = (self.lon_var[-1] - self.lon_var[0]) / (self.nx - 1)
        self.dy = (self.lat_var[-1] - self.lat_var[0]) / (self.ny - 1)

        # Compute the cell corners
        x = self.lon_var
        y = self.lat_var
        dx2 = self.dx / 2
        dy2 = self.dy / 2

        self.cell_x = np.array([x + dx2, x + dx2, x - dx2, x - dx2])
        self.cell_y = np.array([y + dy2, y - dy2, y - dy2, y + dy2])

        super().__init__(name, ccrs.PlateCarree())

    def cell_corners(self, i, j):
        """Return the corners of the cell with indices (i,j).

        See also the docstring of Grid.cell_corners.

        Parameters
        ----------
        i : int
        j : int

        Returns
        -------
        tuple(np.array(shape=(4,), dtype=float),
              np.array(shape=(4,), dtype=float))
            Arrays containing the x and y coordinates of the corners
        """
        return self.cell_x[:, i], self.cell_y[:, j]

    def lon_range(self):
        """Return an array containing all the longitudinal points on the grid.

        Returns
        -------
        np.array(shape=(nx,), dtype=float)
        """
        return self.lon_var

    def lat_range(self):
        """Return an array containing all the latitudinal points on the grid.

        Returns
        -------
        np.array(shape=(ny,), dtype=float)
        """
        return self.lat_var

    
    def cell_areas(self):
        """Return an array containing the grid cell areas.

        Returns
        -------
        np.array(shape=(nx,ny), dtype=float)
        """
        lons_c = np.append(self.cell_x[-1], self.cell_x[0, 0])
        lats_c = np.append(self.cell_y[-1], self.cell_y[0, 0])
        lats_c = np.deg2rad(lats_c)

        dlon = 2 * np.pi / self.nx
        areas = R_EARTH * R_EARTH * dlon * np.abs(np.sin(lats_c[:-1]) - np.sin(lats_c[1:]))
        areas = np.broadcast_to(areas[:, np.newaxis], (self.ny, self.nx))

        return areas.flatten()


class GeoPandasGrid(Grid):
    """A grid that can be easily constructed on a geopandas dataframe."""

    def __init__(self, gdf: gpd.GeoDataFrame, name: str = "gpd_grid"):
        super().__init__(name, gdf.crs)

        self.gdf = gdf

        self.nx = len(gdf)
        self.ny = 1

class VPRMGrid(Grid):
    """Contains the grid from the VPRM emission inventory.

    The grid projection is LambertConformal with a nonstandard globe.
    This means to generate the gridcell-corners a bit of work is
    required, as well as that the gridcell-sizes can't easily be read
    from a dataset.

    Be careful, the lon/lat_range methods return the gridpoint coordinates
    in the grid-projection (and likely have to be transformed to be usable).
    """

    def __init__(self, dataset_path, dx, dy, name):
        """Store the grid information.

        Parameters
        ----------
        dataset_path : str
            Is used to read the gridcell coordinates
        dx : float
            Longitudinal size of a gridcell in meters
        dy : float
            Latitudinal size of a gridcell in meters
        name : str, optional
        """
        self.dx = dx
        self.dy = dy

        projection = ccrs.LambertConformal(
            central_longitude=12.5,
            central_latitude=51.604,
            standard_parallels=[51.604],
            globe=ccrs.Globe(
                ellipse=None, semimajor_axis=6370000, semiminor_axis=6370000
            ),
        )

        # Read grid-values in lat/lon, which are distorted, then
        # project them to LambertConformal where the grid is
        # regular and rectangular.
        with Dataset(dataset_path) as dataset:
            proj_lon = np.array(dataset["lon"][:])
            proj_lat = np.array(dataset["lat"][:])

        self.lon_vals = projection.transform_points(
            ccrs.PlateCarree(), proj_lon[0, :], proj_lat[0, :]
        )[:, 0]
        self.lat_vals = projection.transform_points(
            ccrs.PlateCarree(), proj_lon[:, 0], proj_lat[:, 0]
        )[:, 1]

        # Cell corners
        x = self.lon_vals
        y = self.lat_vals
        dx2 = self.dx / 2
        dy2 = self.dy / 2

        self.cell_x = np.array([x + dx2, x + dx2, x - dx2, x - dx2])
        self.cell_y = np.array([y + dy2, y - dy2, y - dy2, y + dy2])

        super().__init__(name, projection)

    def cell_corners(self, i, j):
        """Return the corners of the cell with indices (i,j).

        See also the docstring of Grid.cell_corners.

        Parameters
        ----------
        i : int
        j : int

        Returns
        -------
        tuple(np.array(shape=(4,), dtype=float),
              np.array(shape=(4,), dtype=float))
            Arrays containing the x and y coordinates of the corners
        """
        return self.cell_x[:, i], self.cell_y[:, j]

    def lon_range(self):
        """Return an array containing all the longitudinal points on the grid.

        Returns
        -------
        np.array(shape=(nx,), dtype=float)
        """
        return self.lon_vals

    def lat_range(self):
        """Return an array containing all the latitudinal points on the grid.

        Returns
        -------
        np.array(shape=(ny,), dtype=float)
        """
        return self.lat_vals


class SwissGrid(Grid):
    """Represent a grid used by swiss inventories, such as meteotest, maiolica
    or carbocount."""

    dx: float
    dy: float
    xmin: float
    ymin: float

    def __init__(self, name, nx, ny, dx, dy, xmin, ymin, crs: int = WGS84):
        """Store the grid information.

        Swiss grids use LV03 coordinates, which switch the axes:

        * x <-> Northing
        * y <-> Easting

        For consistency with the other Grids, we use:

        * x <-> Longitude ~ "swiss y"
        * y <-> Latitude  ~ "swiss x"

        Thus, a header of a .asc file translates as follows:

        * ncols     -> nx
        * nrows     -> ny
        * xllcorner -> ymin
        * yllcorner -> xmin
        * cellsize  -> dx, dy

        Parameters
        ----------
        dx : float
            EASTERLY size of a gridcell in meters
        dy : float
            NORTHLY size of a gridcell in meters
        nx : int
            Number of cells in EASTERLY direction
        ny : int
            Number of cells in NORTHLY direction
        xmin : float
            EASTERLY distance of bottom left gridpoint in meters
        ymin : float
            NORTHLY distance of bottom left gridpoint in meters
        crs:
            The projection to use.
            only
            work with WGS84 or  SWISS .
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.xmin = xmin
        self.ymin = ymin

        self.crs = crs
        # The swiss grid is not technically using a PlateCarree projection
        # (in fact it is not using any projection implemented by cartopy),
        # however the points returned by the cell_corners() method are in
        # WGS84, which PlateCarree defaults to.
        super().__init__(name, ccrs.PlateCarree())

    def cell_corners(self, i, j):
        """Return the corners of the cell with indices (i,j).

        See also the docstring of Grid.cell_corners.

        Parameters
        ----------
        i : int
        j : int

        Returns
        -------
        tuple(np.array(shape=(4,), dtype=float),
              np.array(shape=(4,), dtype=float))
            Arrays containing the x and y coordinates of the corners

        """
        if self.crs == WGS84:
            x1, y1 = self._LV03_to_WGS84(
                self.xmin + i * self.dx, self.ymin + j * self.dy
            )
            x2, y2 = self._LV03_to_WGS84(
                self.xmin + (i + 1) * self.dx, self.ymin + (j + 1) * self.dy
            )

            cell_x = np.array([x2, x2, x1, x1])
            cell_y = np.array([y2, y1, y1, y2])

            return cell_x, cell_y
        elif self.crs == LV95:
            x1, y1 = self.xmin + i * self.dx, self.ymin + j * self.dy
            x2, y2 = self.xmin + (i + 1) * self.dx, self.ymin + (j + 1) * self.dy

            cell_x = np.array([x2, x2, x1, x1])
            cell_y = np.array([y2, y1, y1, y2])

            return cell_x, cell_y
        else:
            raise ValueError(f"Unkonw crs {self.crs}")
    @cache
    def lon_range(self):
        """Return an array containing all the longitudinal points on the grid.

        Returns
        -------
        np.array(shape=(nx,), dtype=float)
        """
        return np.array([self.xmin + i * self.dx for i in range(self.nx + 1)])
    @cache
    def lat_range(self):
        """Return an array containing all the latitudinal points on the grid.

        Returns
        -------
        np.array(shape=(ny,), dtype=float)
        """
        return np.array([self.ymin + j * self.dy for j in range(self.ny + 1)])

    def _LV03_to_WGS84(self, y, x):
        """Convert LV03 to WSG84.

        Based on swisstopo approximated solution (0.1" accuracy)

        For better comparability with other implementations, here:

        * x <-> Northing
        * y <-> Easting,

        contrary to the rest of this class.

        Parameters
        ----------
        y : float
            y coordinate in meters
        x : float
            x coordinate in meters

        Returns
        -------
        tuple(float, float)
            The coordinates of the point in WGS84 (lon, lat)
        """
        x = (x - 200_000) / 1_000_000
        y = (y - 600_000) / 1_000_000

        lon = (
            2.6779094
            + 4.728982 * y
            + 0.791484 * y * x
            + 0.1306 * y * x**2
            - 0.0436 * y**3
        ) / 0.36

        lat = (
            16.9023892
            + 3.238272 * x
            - 0.270978 * y**2
            - 0.002528 * x**2
            - 0.0447 * y**2 * x
            - 0.0140 * x**3
        ) / 0.36

        return lon, lat




class COSMOGrid(Grid):
    """Class to manage a COSMO-domain
    This grid is defined as a rotated pole coordinate system.
    The gridpoints are at the center of the cell.
    """

    nx: int
    ny: int
    dx: float
    dy: float
    xmin: float
    ymin: float
    pollon: float
    pollat: float

    def __init__(self, nx, ny, dx, dy, xmin, ymin, pollon=180, pollat=90):
        """Store the grid information.

        Parameters
        ----------
        nx : int
            Number of cells in longitudinal direction
        ny : int
            Number of cells in latitudinal direction
        dx : float
            Longitudinal size of a gridcell in degrees
        dy : float
            Latitudinal size of a gridcell in degrees
        xmin : float
            Longitude of bottom left gridpoint in degrees
        ymin : float
            Latitude of bottom left gridpoint in degrees
        pollon : float
            Longitude of the rotated pole
        pollat : float
            Latitude of the rotated pole
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.xmin = xmin
        self.ymin = ymin
        self.pollon = pollon
        self.pollat = pollat

        # cell corners
        x = self.xmin + np.arange(self.nx) * self.dx
        y = self.ymin + np.arange(self.ny) * self.dy
        dx2 = self.dx / 2
        dy2 = self.dy / 2

        self.cell_x = np.array([x + dx2, x + dx2, x - dx2, x - dx2])
        self.cell_y = np.array([y + dy2, y - dy2, y - dy2, y + dy2])

        super().__init__(
            "COSMO",
            ccrs.RotatedPole(pole_longitude=pollon, pole_latitude=pollat),
        )

    def gridcell_areas(self):
        """Calculate 2D array of the areas (m^2) of a regular rectangular grid
        on earth.

        Returns
        -------
        np.array
            2D array containing the areas of the gridcells in m^2
            shape: (nx, ny)
        """
        radius = 6375000.0  # the earth radius in meters
        dlon = np.deg2rad(self.dx)
        dlat = np.deg2rad(self.dy)

        # Cell area at equator
        dd = 2.0 * pow(radius, 2) * dlon * np.sin(0.5 * dlat)

        # Cell areas in y-direction
        areas = dd * np.cos(np.deg2rad(self.ymin) + np.arange(self.ny) * dlat)

        return np.broadcast_to(areas, (self.nx, self.ny))

    def lon_range(self):
        """Return an array containing all the longitudinal points on the grid.

        Returns
        -------
        np.array(shape=(nx,), dtype=float)
        """
        # Because of floating point math the original arange is not guaranteed
        # to contain the expected number of points.
        # This way we are sure that we generate at least the required number of
        # points and discard the possibly generated superfluous one.
        # Compared to linspace this method generates more exact steps at
        # the cost of a less accurate endpoint.
        try:
            lon_vals = self.lon_vals
        except AttributeError:
            self.lon_vals = np.arange(
                self.xmin, self.xmin + (self.nx + 0.5) * self.dx, self.dx
            )[: self.nx]
            lon_vals = self.lon_vals
        return lon_vals

    def lat_range(self):
        """Return an array containing all the latitudinal points on the grid.

        Returns
        -------
        np.array(shape=(ny,), dtype=float)
        """
        # See the comment in lon_range
        try:
            lat_vals = self.lat_vals
        except AttributeError:
            self.lat_vals = np.arange(
                self.ymin, self.ymin + (self.ny + 0.5) * self.dy, self.dy
            )[: self.ny]
            lat_vals = self.lat_vals
        return lat_vals

    def cell_corners(self, i, j):
        """Return the corners of the cell with indices (i,j).

        See also the docstring of Grid.cell_corners.

        Parameters
        ----------
        i : int
        j : int

        Returns
        -------
        tuple(np.array(shape=(4,), dtype=float),
              np.array(shape=(4,), dtype=float))
            Arrays containing the x and y coordinates of the corners
        """
        return self.cell_x[:, i], self.cell_y[:, j]

    def indices_of_point(self, lon, lat, proj=ccrs.PlateCarree()):
        """Return the indices of the grid cell that contains the point (lon, lat)

        Parameters
        ----------
        lat : float
            The latitude of the point source
        lon : float
            The longitude of the point source
        proj : cartopy.crs.Projection
            The cartopy projection of the lat/lon of the point source
            Default: cartopy.crs.PlateCarree

        Returns
        -------
        tuple(int, int)
            (cosmo_indx,cosmo_indy),
            the indices of the cosmo grid cell containing the source.

        Raises
        ------
        IndexError
            If the point lies outside the grid.
        """
        point = self.projection.transform_point(lon, lat, proj)

        indx = np.floor((point[0] - self.xmin) / self.dx)
        indy = np.floor((point[1] - self.ymin) / self.dy)

        if indx < 0 or indy < 0 or indx > self.nx - 1 or indy > self.ny - 1:
            raise IndexError("Point lies outside the COSMO Grid")

        return int(indx), int(indy)

    def intersected_cells(self, corners):
        """Given a inventory cell, return a list of cosmo-cell-indices and
        intersection fractions.

        The inventory cell is specified by it's corners. The output is a list
        of tuples, specifying the indices and overlap as a fraction of the
        inventory cell area.

        Parameters
        ----------
        corners : np.array(shape=(4,2), dtype=float)
            The corners of the inventory cell in the COSMO coordinate system

        Returns
        -------
        list(tuple(int, int, float))
            A list containing triplets (x,y,r)
               - x : longitudinal index of cosmo grid cell
               - y : latitudinal index of cosmo grid cell
               - r : ratio of the area of the intersection compared to the total
                     area of the inventory cell.
                     r is in (0,1] (only nonzero intersections are reported)
        """
        # Find around which cosmo grid index the inventory cell lies.
        # Since the inventory cell is in general not rectangular because
        # of different projections, we add a margin of to the extremal indices.
        # This way we're sure we don't miss any intersection.

        cell_xmin = min(k[0] for k in corners)
        lon_idx_min = int((cell_xmin - self.xmin) / self.dx) - 2

        if lon_idx_min > self.nx:
            # The inventory cell lies outside the cosmo grid
            return []

        cell_xmax = max(k[0] for k in corners)
        lon_idx_max = int((cell_xmax - self.xmin) / self.dx) + 3

        if lon_idx_max < 0:
            # The inventory cell lies outside the cosmo grid
            return []

        cell_ymin = min(k[1] for k in corners)
        lat_idx_min = int((cell_ymin - self.ymin) / self.dy) - 2

        if lat_idx_min > self.ny:
            # The inventory cell lies outside the cosmo grid
            return []

        cell_ymax = max(k[1] for k in corners)
        lat_idx_max = int((cell_ymax - self.ymin) / self.dy) + 3

        if lat_idx_max < 0:
            # The inventory cell lies outside the cosmo grid
            return []

        molly = ccrs.Mollweide(central_longitude=self.pollon)
        corners = molly.transform_points(self.projection, corners[:, 0], corners[:, 1])
        inv_cell = Polygon(corners)

        intersections = []
        # make sure we iterate only over valid gridpoint indices
        for i in range(max(0, lon_idx_min), min(self.nx, lon_idx_max)):
            for j in range(max(0, lat_idx_min), min(self.ny, lat_idx_max)):
                corners = np.array(list(zip(*self.cell_corners(i, j))))
                corners = molly.transform_points(
                    self.projection, corners[:, 0], corners[:, 1]
                )

                cosmo_cell = Polygon(corners)
                if cosmo_cell.intersects(inv_cell):
                    overlap = cosmo_cell.intersection(inv_cell)
                    intersections.append((i, j, overlap.area / inv_cell.area))

        return intersections


class ICONGrid(Grid):
    """Class to manage an ICON-domain
    
    This grid is defined as an unstuctured triangular grid (1D).
    The cells are ordered in a deliberate way and indexed with ascending integer numbers.
    The grid file contains variables like midpoint coordinates etc as a fct of the index.
    """

    def __init__(self, dataset_path, name="ICON"):
        """Open the netcdf-dataset and read the relevant grid information.

        Parameters
        ----------
        dataset_path : str
        name : str, optional
        """
        self.dataset_path = dataset_path

        with Dataset(dataset_path) as dataset:
            self.clon_var = np.rad2deg(dataset["clon"][:])
            self.clat_var = np.rad2deg(dataset["clat"][:])
            self.cell_areas = np.array(dataset["cell_area"][:])
            self.vlat = np.rad2deg(dataset["vlat"][:])
            self.vlon = np.rad2deg(dataset["vlon"][:])
            self.vertex_of_cell = np.array(dataset["vertex_of_cell"][:])
            self.cell_of_vertex = np.array(dataset["cells_of_vertex"][:])

        self.ncell = len(self.clat_var)

        # Initiate a list of polygons, which is updated whenever the polygon of a cell
        # is called for the first time
        self.polygons = [
            Polygon(zip(*self._cell_corners(i))) for i in range(self.ncell)
        ]

        # Create a geopandas df
        # ICON_FILE_CRS = 6422
        # Apparently the crs of icon is not what is written in the nc file.
        ICON_FILE_CRS = WGS84
        self.crs = ICON_FILE_CRS
        self.gdf = gpd.GeoDataFrame(geometry=self.polygons, crs=ICON_FILE_CRS)
        self.process_overlap_antimeridian()

        # Consider the ICON-grid as a 1-dimensional grid where ny=1
        self.nx = self.ncell
        self.ny = 1

        self.molly = ccrs.Mollweide()

        super().__init__(name, ccrs.PlateCarree())

    def _cell_corners(self, n):
        """Internal cell corners"""

        return (
            self.vlon[self.vertex_of_cell[:, n] - 1],
            self.vlat[self.vertex_of_cell[:, n] - 1],
        )

    def cell_corners(self, n, j):
        """Return the corners of the cell with index n.

        Parameters
        ----------
        n : int
        j : int

        Returns
        -------
        tuple(np.array(shape=(3,), dtype=float),
              np.array(shape=(3,), dtype=float))
            Arrays containing the lon and lat coordinates of the corners
        """

        return self.gdf.geometry.iloc[n].exterior.coords.xy

    def indices_of_point(self, lon, lat, proj=ccrs.PlateCarree()):
        """Return the indices of the grid cell that contains the point (lon, lat)

        Parameters
        ----------
        lat : float
            The latitude of the point source
        lon : float
            The longitude of the point source

        Returns
        -------
        int
            (icon_indn),
            the index of the ICON grid cell containing the source.

        Raises
        ------
        IndexError
            If the point lies outside the grid.
        """

        indn = -1

        pnt = Point(lon, lat)

        closest_vertex = ((self.vlon - lon) ** 2 + (self.vlat - lat) ** 2).argmin()
        cell_range = self.cell_of_vertex[:, closest_vertex] - 1

        for n in cell_range:

            if n == -1:
                continue

            if self.polygons[n] is not None:
                icon_cell = self.polygons[n]
            else:
                corners = np.array(list(zip(*self.cell_corners(n, 0))))
                icon_cell = Polygon(corners)
                self.polygons[n] = icon_cell

            if icon_cell.contains(pnt):
                indn = n
                break

        if indn == -1:
            raise IndexError("Point lies outside the ICON Grid")

        return int(indn), 0

    def gridcell_areas(self):
        """Calculate 2D array of the areas (m^2) of a regular rectangular grid
        on earth.

        Returns
        -------
        np.array
            2D array containing the areas of the gridcells in m^2
            shape: (ncell)
        """

        return self.cell_areas

    def intersected_cells(self, corners):
        """Given a inventory cell, return a list of ICON-cell-indices and
        intersection fractions.

        The inventory cell is specified by it's corners. The output is a list
        of tuples, specifying the indices and overlap as a fraction of the
        inventory cell area.

        Parameters
        ----------
        corners : np.array(shape=(4,2), dtype=float)
            The corners of the inventory cell in the COSMO coordinate system

        Returns
        -------
        list(tuple(int, float))
            A list containing triplets (x,y,r)
               - n : index of ICON grid cell
               - r : ratio of the area of the intersection compared to the total
                     area of the inventory cell.
                     r is in (0,1] (only nonzero intersections are reported)
        """
        # Find around which ICON grid index the inventory cell lies.

        cell_xmin = min(k[0] for k in corners)
        icon_xmax = max(self.vlon)

        if cell_xmin > icon_xmax:
            # The inventory cell lies outside the cosmo grid
            return []

        cell_xmax = max(k[0] for k in corners)
        icon_xmin = min(self.vlon)

        if cell_xmax < icon_xmin:
            # The inventory cell lies outside the cosmo grid
            return []

        cell_ymin = min(k[1] for k in corners)
        icon_ymax = max(self.vlat)

        if cell_ymin > icon_ymax:
            # The inventory cell lies outside the cosmo grid
            return []

        cell_ymax = max(k[1] for k in corners)
        icon_ymin = min(self.vlat)

        if cell_ymax < icon_ymin:
            # The inventory cell lies outside the cosmo grid
            return []

        corners = self.molly.transform_points(
            self.projection, corners[:, 0], corners[:, 1]
        )
        inv_cell = Polygon(corners)

        intersections = []
        # make sure we iterate only over valid gridpoint indices
        for n in np.arange(self.ncell):
            icon_cell_xmin = min(self.vlon[self.vertex_of_cell[:, n] - 1])
            if cell_xmax < icon_cell_xmin:
                continue
            icon_cell_xmax = max(self.vlon[self.vertex_of_cell[:, n] - 1])
            if cell_xmin > icon_cell_xmax:
                continue
            icon_cell_ymin = min(self.vlat[self.vertex_of_cell[:, n] - 1])
            if cell_ymax < icon_cell_ymin:
                continue
            icon_cell_ymax = max(self.vlat[self.vertex_of_cell[:, n] - 1])
            if cell_ymin > icon_cell_ymax:
                continue
            corners = np.array(list(zip(*self.cell_corners(n, 0))))
            corners = self.molly.transform_points(
                self.projection, corners[:, 0], corners[:, 1]
            )
            icon_cell = Polygon(corners)
            if icon_cell.intersects(inv_cell):
                overlap = icon_cell.intersection(inv_cell)
                intersections.append((n, 0, overlap.area / inv_cell.area))

        return intersections  


    def process_overlap_antimeridian(self):
        """Find polygons intersecting the antimeridian line 
        and split them into two polygons represented by a 
        MultiPolygon.
        """

        def shift_lon_poly(poly):
            coords = poly.exterior.coords 
            lons = np.array([coord[0] for coord in coords])
            lats = [coord[1] for coord in coords]
            if np.any(lons > 180):
                lons -= 360
            elif np.any(lons < -180):
                lons += 360
            return Polygon([*zip(lons, lats)]) 

        def detect_antimeridian_poly(poly):
            coords = poly.exterior.coords
            lon1, lon2, lon3 = coords[0][0], coords[1][0], coords[2][0]
            coords_cond1 = [list(c) for c in coords[:-1]]

            cond1 = np.count_nonzero(np.array([lon1, lon2, lon3]) > 180. - 1e-5) == 2
            if cond1:
                if lon1 < 0:
                    coords_cond1[1][0] = lon2 - 360
                    coords_cond1[2][0] = lon3 - 360
                elif lon2 < 0:
                    coords_cond1[0][0] = lon1 - 360
                    coords_cond1[2][0] = lon3 - 360
                elif lon3 < 0:
                    coords_cond1[0][0] = lon1 - 360
                    coords_cond1[1][0] = lon2 - 360

            vmin = -140
            vmax = 140
            lon1, lon2, lon3 = coords_cond1[0][0], coords_cond1[1][0], coords_cond1[2][0]
            cond2 = (lon1 > vmax or lon1 < vmin) or (lon2 > vmax or lon2 < vmin) or (lon3 > vmax or lon3 < vmin)
            coords_cond2 = [list(c) for c in coords_cond1]

            if cond2:

                if lon1 * lon2 < 0 and lon1 * lon3 < 0:
                    coords_cond2[0][0] = lon1 - math.copysign(1, lon1) * 360

                elif lon2 * lon1 < 0 and lon2 * lon3 < 0:
                    coords_cond2[1][0] = lon2 - math.copysign(1, lon2) * 360

                elif lon3 * lon1 < 0 and lon3 * lon2 < 0:
                    coords_cond2[2][0] = lon3 - math.copysign(1, lon3) * 360 

            return Polygon(coords_cond2) 

        crs = pyproj.CRS.from_epsg(WGS84)
        bounds = crs.area_of_use.bounds

        xx_bounds, yy_bounds = box(*bounds).exterior.coords.xy
        coords_bounds = [(x, y) for x, y in zip(xx_bounds, yy_bounds)]
        bounds_line = LineString(coords_bounds)

        self.gdf = self.gdf.set_geometry(self.gdf.geometry.apply(lambda poly: detect_antimeridian_poly(poly)))
        gdf_inter = self.gdf.loc[self.gdf.intersects(bounds_line)]
        gdf_inter = gdf_inter.set_geometry(gdf_inter.geometry.apply(lambda poly: MultiPolygon(split(poly, bounds_line))))
        gdf_inter = gdf_inter.set_geometry(gdf_inter.geometry.apply(lambda mpoly: MultiPolygon([shift_lon_poly(poly) for poly in mpoly.geoms])))
        self.gdf.loc[gdf_inter.index, 'geometry'] = gdf_inter.geometry
