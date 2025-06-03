"""Emiproc Grids.

Classes handling different grids, namely the simulation grids and
grids used in different emissions inventories.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
import warnings
from functools import cache, cached_property
from typing import Iterable

import geopandas as gpd
import numpy as np
import pyproj
import xarray as xr
from netCDF4 import Dataset
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import split
from shapely.creation import polygons

WGS84 = 4326
WGS84_PROJECTED = 3857
LV95 = 2056  # EPSG:2056, swiss CRS, unit: meters
WGS84_NSIDC = 6933  # Unit: meters

# Radius of the earth
R_EARTH = 6371000  # m


logger = logging.getLogger(__name__)

# Type alias
# minx, miny, maxx, maxy
BoundingBox = tuple[float, float, float, float]


class Grid:
    """Abstract base class for a grid.

    Derive your own grid implementation from this and make sure to provide
    an appropriate implementation of the required methods.

    :param name: Name of the grid.
    :type name: str
    :param crs: The coordinate reference system of the grid.
    :type crs: int | str | pyproj.CRS

    :param gdf: A geopandas dataframe containing the grid cells as geometries.
    :type gdf: gpd.GeoDataFrame
    :param cells_as_polylist: A list of polygons representing the grid cells.
    :type cells_as_polylist: list[shapely.geometry.Polygon]

    :param nx: Number of cells in the x direction.
    :type nx: int
    :param ny: Number of cells in the y direction. Set this to 1 for non-regular grids.
    :type ny: int
    :param shape: The shape of the grid as a tuple (nx, ny).
    :type shape: tuple[int, int]

    :param corners: Corners of the cells.
    :type corners: np.ndarray | None
    :param centers: Centers of the cells, as a GeoSeries of Points.
        You can use grid.centers.x and grid.centers.y to get the x and y coordinates.
    :type centers: gpd.GeoSeries
    :param cell_areas: Area of the cells in m^2.
    :type cell_areas: Iterable[float]
    """

    name: str

    nx: int
    ny: int

    # The crs value as an integer
    crs: int | str
    gdf: gpd.GeoDataFrame

    # Optional corners of the cells (used for irregular grids)
    corners: np.ndarray | None = None

    def __init__(self, name: str | None, crs: int | str = WGS84):
        """
        Parameters
        ----------
        name : Name of the grid.
        crs : The coordinate reference system of the grid.
        """
        if name is None:
            name = "unnamed"
        self.name = name
        self.crs = crs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        """Return a geopandas dataframe containing the grid."""
        if not hasattr(self, "_gdf"):
            self._gdf = gpd.GeoDataFrame(
                geometry=self.cells_as_polylist,
                crs=self.crs,
            )
        return self._gdf

    @gdf.setter
    def gdf(self, value: gpd.GeoDataFrame):
        warnings.warn(
            "deprectated to set the gdf of a grid. It is now automatically generated."
        )
        self._gdf = value

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
    def shape(self) -> tuple[int, int]:
        return (self.nx, self.ny)

    @cached_property
    def cell_areas(self) -> Iterable[float]:
        """Return an array containing the area of each cell in m2."""
        return (
            gpd.GeoSeries(self.cells_as_polylist, crs=self.crs)
            # Convert to WGS84 to get the area in m^2
            .to_crs(epsg=WGS84_NSIDC).area
        )

    @cached_property
    def centers(self) -> gpd.GeoSeries:
        """Return a GeoSeries with the points centers of the cells."""
        return self.gdf.centroid

    def __len__(self):
        """Return the number of cells in the grid."""
        return self.nx * self.ny


class RegularGrid(Grid):
    """Regular grid a grids with squared cells.

    This allows for some capabilities that are not available for
    irregular grids (rasterization, image like plotting).

    To create the grid, one mandatory parameter is the reference:

    :param xmin/ymin: The minimum x and y coordinate of the grid.

    Then you need two of the three following:

    :param xmax/ymax: The maximum x and y coordinate of the grid.
    :param nx/ny: The number of cells in both directions.
    :param dx/dy: The size of the cells.
        The number of decimals specified is used to round the coordinates.

    The grid will be constructed to fit the given parameters.
    """

    # The centers of the cells (lon =x, lat = y)
    lon_range: np.ndarray
    lat_range: np.ndarray

    # The edges of the cells
    lat_bounds: np.ndarray
    lon_bounds: np.ndarray

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
        name: str | None = None,
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
            nx = (xmax - xmin) / dx
            ny = (ymax - ymin) / dy

            # Round to avoid decimal errors
            # Get the decimals in the dx and dy
            get_rounding = (
                lambda x: (len(str(x).split(".")[1]) if isinstance(x, float) else 0)
                or None
            )
            clean = lambda n, d: math.ceil(
                round(n, get_rounding(d)) if get_rounding(d) is not None else n
            )
            nx = clean(nx, dx)
            ny = clean(ny, dy)

        elif dx is None and dy is None:
            dx = (xmax - xmin) / nx
            dy = (ymax - ymin) / ny

        # Set maxs or correct maxs to ensure consistency with ns and ds
        xmax = xmin + nx * dx
        ymax = ymin + ny * dy

        self.xmax, self.ymax = xmax, ymax

        # Calculate all grid parameters
        self.nx, self.ny = nx, ny
        self.dx, self.dy = dx, dy

        # Build arrays
        build_range = lambda min_, n_, d_: min_ + np.arange(n_) * d_ + d_ / 2
        self.lon_range = build_range(self.xmin, self.nx, self.dx)
        self.lat_range = build_range(self.ymin, self.ny, self.dy)

        self.lon_bounds = np.concatenate(
            [self.lon_range - self.dx / 2, [self.lon_range[-1] + self.dx / 2]]
        )
        self.lat_bounds = np.concatenate(
            [self.lat_range - self.dy / 2, [self.lat_range[-1] + self.dy / 2]]
        )

        assert len(self.lon_range) == nx, f"{len(self.lon_range)=} != {nx=}"
        assert len(self.lat_range) == ny, f"{len(self.lat_range)=} != {ny=}"
        assert len(self.lon_bounds) == nx + 1
        assert len(self.lat_bounds) == ny + 1

        super().__init__(name, crs)

    def __repr__(self) -> str:
        return (
            f"{super().__repr__()}_"
            f"nx({self.nx})_ny({self.ny})_"
            f"dx({self.dx})_dy({self.dy})_"
            f"x({self.xmin},{self.xmax})_"
            f"y({self.ymin},{self.ymax})_"
        )

    @cached_property
    def cells_as_polylist(self) -> list[Polygon]:

        x_coords, y_coords = np.meshgrid(
            self.lon_range - self.dx / 2.0, self.lat_range - self.dy / 2.0
        )
        # Reshape to 1D (order set for backward compatibility)
        x_coords = x_coords.flatten(order="F")
        y_coords = y_coords.flatten(order="F")
        dx = float(self.dx)
        dy = float(self.dy)
        coords = np.array(
            [
                [x, y]
                for x, y in zip(
                    [x_coords, x_coords, x_coords + dx, x_coords + dx],
                    [y_coords, y_coords + dy, y_coords + dy, y_coords],
                )
            ]
        )
        coords = np.rollaxis(coords, -1, 0)
        return polygons(coords)

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
    def bounds(self) -> tuple[int, int, int, int]:
        return self.xmin, self.ymin, self.xmax, self.ymax

    @cached_property
    def centers(self) -> gpd.GeoSeries:
        """Return a GeoSeries with the points centers of the cells."""

        # Require no calculation compared to the gdf.centroid
        return gpd.GeoSeries.from_xy(
            np.repeat(self.lon_range, self.ny),
            np.tile(self.lat_range, self.nx),
        )

    @classmethod
    def from_centers(
        cls,
        x_centers: np.ndarray,
        y_centers: np.ndarray,
        name=None,
        crs=WGS84,
        rounding: int | None = None,
    ) -> RegularGrid:
        """Create a regular grid from the center points of the cells.

        :param x_centers: The x-coordinates of the cell centers.
        :param y_centers: The y-coordinates of the cell centers.
        :param name: The name of the grid.
        :param crs: The coordinate reference system of the grid.
        :param rounding: The number of decimal places to round the dx and dy.
            This is useful to correct floating point errors.
        """

        # Calculate the dx and dy
        dxs = np.diff(x_centers)
        dys = np.diff(y_centers)

        dx = dxs[0]
        dy = dys[0]

        if rounding is not None:
            dx = round(dx, rounding)
            dy = round(dy, rounding)

            dxs = np.round(dxs, decimals=rounding)
            dys = np.round(dys, decimals=rounding)

        if not np.allclose(dxs, dx) or not np.allclose(dys, dy):
            raise ValueError("The centers are not equally spaced.")

        nx = len(x_centers)
        ny = len(y_centers)

        xmin = x_centers[0] - dx / 2
        ymin = y_centers[0] - dy / 2

        if rounding is not None:
            xmin = round(xmin, rounding)
            ymin = round(ymin, rounding)

        try:
            grid = cls(
                xmin=x_centers[0] - dx / 2,
                ymin=y_centers[0] - dy / 2,
                nx=nx,
                ny=ny,
                dx=dx,
                dy=dy,
                name=name,
                crs=crs,
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not create grid from centers. "
                "If there is a mismatch in coordinates, "
                "try to set `rounding` to some value."
            ) from e

        return grid


class HexGrid(Grid):
    """A grid with hexagonal cells.

    The grid is similar to a regular grid, but the cells are hexagons.
    This implies that the lines of the grid are not all parallel.

    For the arguments, look at :py:class:`RegularGrid`.
    Note that `dx and dy` have been replaced by `spacing` parameter,

    :arg spacing: The distance between the centers of two adjacent hexagons
        on a row. This is also the diameter of the circle inscribed in the hexagon.
    :arg oriented_north: If True, the hexagons are oriented with the top
        and bottom sides parallel on the y-axis. If False, the hexagons
        are oriented with the left and right sides parallel on the x-axis.

    """

    def __init__(
        self,
        xmin: float,
        ymin: float,
        xmax: float | None = None,
        ymax: float | None = None,
        nx: int | None = None,
        ny: int | None = None,
        spacing: float | None = None,
        oriented_north: bool = True,
        name: str | None = None,
        crs: int | str = WGS84,
    ):

        # Correct the delta in case the orientation is on the other ax
        correct = lambda x: x * np.sqrt(3) / 2

        self.xmin, self.ymin = xmin, ymin

        # Check if they did not specify all the optional parameters
        if all((p is not None for p in [xmax, ymax, nx, ny, spacing])):
            raise ValueError(
                "Specified too many parameters. "
                "Specify only 2 of the following: "
                "(xmax, ymax), (nx, ny), (spacing)"
            )

        if spacing is None and xmax is None and ymax is None:
            raise ValueError(
                "Cannot create grid with only nx and ny. "
                "Specify at least spacing or xmax, ymax."
            )

        if spacing is not None:
            # Guess the nx and ny values, override the max
            dx = spacing if oriented_north else correct(spacing)
            dy = correct(spacing) if oriented_north else spacing

        # Calclate the number of cells if not specified
        if nx is None and ny is None:
            if spacing is None:
                raise ValueError(
                    "Either nx and ny or dx and dy must be specified. "
                    f"Received: {nx=}, {ny=}, {spacing=}"
                )
            if xmax is None or ymax is None:
                raise ValueError(
                    "When using only dx dy, xmax and ymax must be specified. "
                    f"Received: {xmax=}, {ymax=}"
                )

            nx = math.ceil((xmax - xmin) / dx)
            ny = math.ceil((ymax - ymin) / dy)

        if xmax is None and ymax is None:
            xmax = xmin + nx * dx
            ymax = ymin + ny * dy
        self.xmax, self.ymax = xmax, ymax

        # Calculate all grid parameters
        self.nx, self.ny = nx, ny
        self.dx, self.dy = (xmax - xmin) / nx, (ymax - ymin) / ny

        self.lon_range = np.arange(xmin, xmax, self.dx) + self.dx / 2
        self.lat_range = np.arange(ymin, ymax, self.dy) + self.dy / 2

        assert len(self.lon_range) == nx, f"{len(self.lon_range)=} != {nx=}"
        assert len(self.lat_range) == ny, f"{len(self.lat_range)=} != {ny=}"

        if name is None:
            name = f"HexGrid x({xmin},{xmax})_y({ymin},{ymax})_nx({nx})_ny({ny})"

        self.oriented_north = oriented_north

        super().__init__(name, crs)

    @cached_property
    def cells_as_polylist(self) -> list[Polygon]:
        """Return all the cells as a list of polygons."""

        x_centers, y_centers = np.meshgrid(self.lon_range, self.lat_range)
        # Shift the odd rows
        if self.oriented_north:
            x_centers[1::2] += self.dx / 2
        else:
            y_centers[:, 1::2] += self.dy / 2

        # Calculat the corners of the hexagons
        corners = []
        flatten = lambda x: x.flatten(order="F")
        x_centers = flatten(x_centers)
        y_centers = flatten(y_centers)

        # half_offset = np.sqrt(3) / 2
        # half_offset = np.sqrt(3) / 8
        half_offset = 1 / (np.sqrt(3))
        offsets_x = [0, 1, 1, 0, -1, -1]
        offsets_y = [
            2 - half_offset,
            half_offset,
            -half_offset,
            -2 + half_offset,
            -half_offset,
            half_offset,
        ]
        if not self.oriented_north:
            offsets_x, offsets_y = offsets_y, offsets_x

        for off_x, off_y in zip(offsets_x, offsets_y):
            x = x_centers + self.dx / 2 * off_x
            y = y_centers + self.dy / 2 * off_y
            corners.append(np.stack([x, y], axis=-1))
        # Reshape to 1D (order set for backward compatibility)

        coords = np.stack(corners, axis=1)
        return polygons(coords)


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
        areas = (
            R_EARTH * R_EARTH * dlon * np.abs(np.sin(lats_c[:-1]) - np.sin(lats_c[1:]))
        )
        areas = np.broadcast_to(areas[np.newaxis, :], (self.nx, self.ny))

        return areas.flatten()


class GeoPandasGrid(Grid):
    """A grid that can be easily constructed on a geopandas dataframe."""

    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        name: str = "gpd_grid",
        shape: tuple[int, int] | None = None,
    ):
        super().__init__(name, gdf.crs)

        self._gdf = gdf

        if shape is not None:
            self.nx, self.ny = shape
        else:
            self.nx = len(gdf)
            self.ny = 1

    @property
    def cells_as_polylist(self) -> list[Polygon]:
        """Return all the cells as a list of polygons."""
        return self.gdf.geometry.tolist()


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
            name=name, nx=nx, ny=ny, dx=dx, dy=dy, xmin=xmin, ymin=ymin, crs=crs
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


class ICONGrid(Grid):
    """Class to manage an ICON-domain

    This grid is defined as an unstuctured triangular grid (1D).
    The cells are ordered in a deliberate way and indexed with ascending integer numbers.
    The grid file contains variables like midpoint coordinates etc as a fct of the index.
    """

    def __init__(self, dataset_path, name: str = None):
        """Open the netcdf-dataset and read the relevant grid information.

        Parameters
        ----------
        dataset_path : str
        name : str, optional
        """

        self.dataset_path = Path(dataset_path)

        if name is None:
            name = self.dataset_path.stem

        with Dataset(dataset_path) as dataset:
            self.clon_var = np.rad2deg(dataset["clon"][:])
            self.clat_var = np.rad2deg(dataset["clat"][:])
            self.cell_areas = np.array(dataset["cell_area"][:])
            self.vlat = np.rad2deg(dataset["vlat"][:])
            self.vlon = np.rad2deg(dataset["vlon"][:])
            self.vertex_of_cell = np.array(dataset["vertex_of_cell"][:])
            self.cell_of_vertex = np.array(dataset["cells_of_vertex"][:])

            self.ncell = len(self.clat_var)

            corners = np.zeros((self.ncell, 3, 2))
            corners[:, :, 0] = self.vlon[self.vertex_of_cell - 1].T
            corners[:, :, 1] = self.vlat[self.vertex_of_cell - 1].T
            self.corners = corners

        # Initiate a list of polygons, which is updated whenever the polygon of a cell
        # is called for the first time
        self.polygons = polygons(self.corners)

        # Create a geopandas df
        # ICON_FILE_CRS = 6422
        # Apparently the crs of icon is not what is written in the nc file.
        ICON_FILE_CRS = WGS84

        self._gdf = gpd.GeoDataFrame(geometry=self.polygons, crs=ICON_FILE_CRS)
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

        self._gdf = self.gdf.set_geometry(
            self.gdf.geometry.apply(lambda poly: detect_antimeridian_poly(poly))
        )
        gdf_inter = self._gdf.loc[self.gdf.intersects(bounds_line)]
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
        self._gdf.loc[gdf_inter.index, "geometry"] = gdf_inter.geometry
