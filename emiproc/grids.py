"""Emiproc Grids.

Classes handling different grids, namely the simulation grids and
grids used in different emissions inventories.
"""
from __future__ import annotations
from functools import cache, cached_property
from typing import Iterable
import numpy as np
import xarray as xr
import geopandas as gpd
import pyproj
import math

from netCDF4 import Dataset
from shapely.geometry import Polygon, Point, box, LineString, MultiPolygon
from shapely.ops import split

WGS84 = 4326
WGS84_PROJECTED = 3857
LV95 = 2056  # EPSG:2056, swiss CRS, unit: meters
WGS84_NSIDC = 6933  # Unit: meters

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
    crs: int | str
    gdf: gpd.GeoDataFrame | None = None

    def __init__(self, name: str, crs: int | str = WGS84):
        """
        Parameters
        ----------
        name : Name of the grid.
        crs : The coordinate reference system of the grid.
        """
        self.name = name
        self.crs = crs

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

    @cached_property
    def cells_as_polylist(self) -> list[Polygon]:
        """Return all the cells as a list of polygons."""
        return [
            Polygon(zip(*self.cell_corners(i, j)))
            for i in range(self.nx)
            for j in range(self.ny)
        ]

    @cached_property
    def cell_areas(self) -> Iterable[float]:
        """Return an array containing the area of each cell in m2."""
        return (
            gpd.GeoSeries(self.cells_as_polylist, crs=self.crs)
            # Convert to WGS84 to get the area in m^2
            .to_crs(epsg=WGS84_NSIDC)
            .area
        )


class RegularGrid(Grid):
    """Regular grid a grids with squared cells.

    This allows for some capabilities that are not available for
    irregular grids (rasterization, image like plotting).

    The grid can be defined in multiple ways.
    All the way need the reference (xmin, ymin).
    Then you need 2 of the three following:

    * xmax, ymax to define the bounding box
    * nx, ny to define the number of cells in each direction
    * dx, dy to define the size of the cells in each direction
    
    Leave the unused parameters as None.
    """

    # The centers of the cells (lon =x, lat = y)
    lon_range: np.ndarray
    lat_range: np.ndarray

    xmin: float
    xmax: float
    ymin: float
    ymax: float

    dx: float
    dy: float

    def __init__(
        self,
        xmin: float,
        ymin: float,
        xmax: float | None = None,
        ymax: float | None = None,
        nx: int | None = None,
        ny: int | None = None,
        dx: float | None = None,
        dy: float | None = None,
        name: str = "",
        crs: int | str = WGS84,
    ):
        self.xmin, self.ymin = xmin, ymin

        # Check if they did not specify all the optional parameters
        if all((p is not None for p in [xmax, ymax, nx, ny, dx, dy])):
            raise ValueError(
                "Specified too many parameters. "
                "Specify only 2 of the following: "
                "(xmax, ymax), (nx, ny), (dx, dy)"
            )
        
        if dx is None and dy is None and xmax is None and ymax is None:
            raise ValueError(
                "Cannot create grid with only nx and ny. "
                "Specify at least dx, dy or xmax, ymax."
            )

        # Calclate the number of cells if not specified
        if nx is None and ny is None:
            if dx is None or dy is None:
                raise ValueError(
                    "Either nx and ny or dx and dy must be specified. "
                    f"Received: {nx=}, {ny=}, {dx=}, {dy=}"
                )
            if xmax is None or ymax is None:
                raise ValueError(
                    "When using only dx dy, xmax and ymax must be specified. "
                    f"Received: {xmax=}, {ymax=}"
                )
            # Guess the nx and ny values, override the max
            nx = math.ceil((xmax - xmin) / dx)
            ny = math.ceil((ymax - ymin) / dy)
        
        if xmax is None and ymax is None:
            xmax = xmin + nx * dx
            ymax = ymin + ny * dy

        # Calculate all grid parameters
        self.nx, self.ny = nx, ny
        self.dx, self.dy = (xmax - xmin) / nx, (ymax - ymin) / ny

        self.lon_range = np.linspace(xmin, xmax, nx) + self.dx / 2
        self.lat_range = np.linspace(ymin, ymax, ny) + self.dy / 2

        super().__init__(name, crs)

    def cell_corners(self, i, j):
        """Return the corners of the cell with indices (i,j).

        The points are ordered clockwise, starting in the top
        right:

        """
        x = self.xmin + i * self.dx
        y = self.ymin + j * self.dy

        return (
            np.array([x, x + self.dx, x + self.dx, x]),
            np.array([y, y, y + self.dy, y + self.dy]),
        )

    @cached_property
    def shape(self) -> tuple[int, int]:
        return (self.nx, self.ny)
    
    @cached_property
    def bounds(self) -> tuple[int, int, int, int]:
        return self.xmin, self.ymin, self.xmax, self.ymax


class LatLonNcGrid(RegularGrid):
    """A regular grid with lat/lon values from a nc file.

    This is a copy of the tno grid basically, but reading a nc file.
    """

    def __init__(
        self, dataset_path, lat_name="clat", lon_name="clon", name="LatLon", crs=WGS84
    ):

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

        super().__init__(name, crs=crs)

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

    @cached_property
    def lon_range(self):
        """Return an array containing all the longitudinal points on the grid.

        Returns
        -------
        np.array(shape=(nx,), dtype=float)
        """
        return self.lon_var

    @cached_property
    def lat_range(self):
        """Return an array containing all the latitudinal points on the grid.

        Returns
        -------
        np.array(shape=(ny,), dtype=float)
        """
        return self.lat_var


class TNOGrid(RegularGrid):
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

        # by pass the regular grid __inti__ method, as variable  have been
        # initialized here
        Grid.__init__(self, name=name, crs=WGS84)

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

    @property
    def lon_range(self):
        """Return an array containing all the longitudinal points on the grid.

        Returns
        -------
        np.array(shape=(nx,), dtype=float)
        """
        return self.lon_var

    @property
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

        super().__init__(name, crs=WGS84)

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

    @cached_property
    def lon_range(self):
        """Return an array containing all the longitudinal points on the grid.

        Returns
        -------
        np.array(shape=(nx,), dtype=float)
        """
        return self.lon_var

    @cached_property
    def lat_range(self):
        """Return an array containing all the latitudinal points on the grid.

        Returns
        -------
        np.array(shape=(ny,), dtype=float)
        """
        return self.lat_var

    @cached_property
    def cell_areas(self):
        """Return an array containing the grid cell areas.

        Returns
        -------
        np.array(shape=(nx,ny), dtype=float)
        """
        lats_c = np.append(self.cell_y[1], self.cell_y[0, -1])
        lats_c = np.deg2rad(lats_c)

        dlon = 2 * np.pi / self.nx
        areas = (R_EARTH * R_EARTH * dlon * np.abs(np.sin(lats_c[:-1]) - np.sin(lats_c[1:])))
        areas = np.broadcast_to(areas[np.newaxis, :], (self.nx, self.ny))

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

    .. warning::

        This is not usable in emiproc v2.
        Please fix it before using it again.

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

        # TODO: FIX THIS GRID
        # projection = ccrs.LambertConformal(
        #     central_longitude=12.5,
        #     central_latitude=51.604,
        #     standard_parallels=[51.604],
        #     globe=ccrs.Globe(
        #         ellipse=None, semimajor_axis=6370000, semiminor_axis=6370000
        #     ),
        # )

        # Read grid-values in lat/lon, which are distorted, then
        # project them to LambertConformal where the grid is
        # regular and rectangular.
        with Dataset(dataset_path) as dataset:
            proj_lon = np.array(dataset["lon"][:])
            proj_lat = np.array(dataset["lat"][:])

        # self.lon_vals = projection.transform_points(
        #     ccrs.PlateCarree(), proj_lon[0, :], proj_lat[0, :]
        # )[:, 0]
        # self.lat_vals = projection.transform_points(
        #     ccrs.PlateCarree(), proj_lon[:, 0], proj_lat[:, 0]
        # )[:, 1]

        # Cell corners
        x = self.lon_vals
        y = self.lat_vals
        dx2 = self.dx / 2
        dy2 = self.dy / 2

        self.cell_x = np.array([x + dx2, x + dx2, x - dx2, x - dx2])
        self.cell_y = np.array([y + dy2, y - dy2, y - dy2, y + dy2])

        super().__init__(
            name,
        )

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

    @property
    def lon_range(self):
        """Return an array containing all the longitudinal points on the grid.

        Returns
        -------
        np.array(shape=(nx,), dtype=float)
        """
        return self.lon_vals

    @property
    def lat_range(self):
        """Return an array containing all the latitudinal points on the grid.

        Returns
        -------
        np.array(shape=(ny,), dtype=float)
        """
        return self.lat_vals


class SwissGrid(RegularGrid):
    """Represent a grid used by swiss inventories, such as meteotest, maiolica
    or carbocount."""

    dx: float
    dy: float
    xmin: float
    ymin: float

    def __init__(self, name, nx, ny, dx, dy, xmin, ymin, crs: int = LV95):
        """Store the grid information.

        Swiss grids use LV95 coordinates, which switch the axes:

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

        # The swiss grid is not technically using a PlateCarree projection
        # (in fact it is not using any projection implemented by cartopy),
        # however the points returned by the cell_corners() method are in
        # WGS84, which PlateCarree defaults to.
        super().__init__(
            name=name,
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            xmin=xmin,
            ymin=ymin,
            crs=crs
        )

    @cached_property
    def lon_range(self):
        """Return an array containing all the longitudinal points on the grid.

        Returns
        -------
        np.array(shape=(nx,), dtype=float)
        """
        return np.array([self.xmin + i * self.dx for i in range(self.nx + 1)])

    @cached_property
    def lat_range(self):
        """Return an array containing all the latitudinal points on the grid.

        Returns
        -------
        np.array(shape=(ny,), dtype=float)
        """
        return np.array([self.ymin + j * self.dy for j in range(self.ny + 1)])


class COSMOGrid(Grid):
    """Class to manage a COSMO-domain
    This grid is defined as a rotated pole coordinate system.
    The gridpoints are at the center of the cell.

    .. warning::

        This is not usable in emiproc v2.
        Please fix it before using it again.

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
            # fix projecction
            # ccrs.RotatedPole(pole_longitude=pollon, pole_latitude=pollat),
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

    @property
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

    @property
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

        self.gdf = gpd.GeoDataFrame(geometry=self.polygons, crs=ICON_FILE_CRS)
        self.process_overlap_antimeridian()

        # Consider the ICON-grid as a 1-dimensional grid where ny=1
        self.nx = self.ncell
        self.ny = 1

        super().__init__(name, crs=ICON_FILE_CRS)

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

            cond1 = np.count_nonzero(np.array([lon1, lon2, lon3]) > 180.0 - 1e-5) == 2
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
            lon1, lon2, lon3 = (
                coords_cond1[0][0],
                coords_cond1[1][0],
                coords_cond1[2][0],
            )
            cond2 = (
                (lon1 > vmax or lon1 < vmin)
                or (lon2 > vmax or lon2 < vmin)
                or (lon3 > vmax or lon3 < vmin)
            )
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

        self.gdf = self.gdf.set_geometry(
            self.gdf.geometry.apply(lambda poly: detect_antimeridian_poly(poly))
        )
        gdf_inter = self.gdf.loc[self.gdf.intersects(bounds_line)]
        gdf_inter = gdf_inter.set_geometry(
            gdf_inter.geometry.apply(
                lambda poly: MultiPolygon(split(poly, bounds_line))
            )
        )
        gdf_inter = gdf_inter.set_geometry(
            gdf_inter.geometry.apply(
                lambda mpoly: MultiPolygon(
                    [shift_lon_poly(poly) for poly in mpoly.geoms]
                )
            )
        )
        self.gdf.loc[gdf_inter.index, "geometry"] = gdf_inter.geometry
