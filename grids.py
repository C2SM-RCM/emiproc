"""
Classes handling different grids, namely the COSMO simulation grid and
grids used in different emissions inventories.
"""
import numpy as np
import cartopy.crs as ccrs

from netCDF4 import Dataset
from shapely.geometry import Polygon


class Grid:
    """Abstract base class for a grid.
    Derive your own grid implementation from this and make sure to provide
    an appropriate implementation of the required methods.
    As an example you can look at TNOGrid.
    """

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

        Returns a tuple of arrays with shape (4,). The first
        tuple element are the x-coordinates of the corners,
        the second are the y-coordinates.

        The points are ordered clockwise, starting in the top
        left:

        4.   1.
        ^    v
        3. < 2.

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


class TNOGrid(Grid):
    """Contains the grid from the TNO emission inventory"""

    def __init__(self, dataset_path):
        """Open the netcdf-dataset and read the relevant grid information.

        Parameters
        ----------
        dataset_path : str
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

        super().__init__("TNO", ccrs.PlateCarree())

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
        x = self.lon_var[i]
        y = self.lat_var[j]
        dx2 = self.dx / 2
        dy2 = self.dy / 2
        cell_x = np.array([x + dx2, x + dx2, x - dx2, x - dx2])

        cell_y = np.array([y + dy2, y - dy2, y - dy2, y + dy2])

        return cell_x, cell_y

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


class SwissGrid(Grid):
    """Represent a grid used by swiss inventories, such as meteotest, maiolica
    or carbocount."""

    nx: int
    ny: int
    dx: float
    dy: float
    xmin: float
    ymin: float

    def __init__(
        self,
        name,
        nx,
        ny,
        dx,
        dy,
        xmin,
        ymin,
        I_HAVE_UNDERSTOOD_THE_CONVENTION_SWITCH_MADE_IN_THIS_METHOD=False,
    ):
        """Store the grid information.

        Swiss grids use LV03 coordinates, which switch the axes:
        x <-> Northing
        y <-> Easting

        For consistency with the other Grids, we use:
        x <-> Longitude ~ "swiss y"
        y <-> Latitude  ~ "swiss x"

        Thus, a header of a .asc file translates as follows:
        ncols     -> nx
        nrows     -> ny
        xllcorner -> ymin
        yllcorner -> xmin
        cellsize  -> dx, dy

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
        I_HAVE_UNDERSTOOD_THE_CONVENTION_SWITCH_MADE_IN_THIS_METHOD : bool
            default: False
            To make sure you are aware of the switch y->x, x->y, set this to
            True.
        """
        if not I_HAVE_UNDERSTOOD_THE_CONVENTION_SWITCH_MADE_IN_THIS_METHOD:
            raise RuntimeError(
                "Please review your initialization of the SwissGrid"
            )

        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.xmin = xmin
        self.ymin = ymin

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
        x1, y1 = self._LV03_to_WGS84(
            self.xmin + i * self.dx, self.ymin + j * self.dy
        )
        x2, y2 = self._LV03_to_WGS84(
            self.xmin + (i + 1) * self.dx, self.ymin + (j + 1) * self.dy
        )

        cell_x = np.array([x2, x2, x1, x1])
        cell_y = np.array([y2, y1, y1, y2])

        return cell_x, cell_y

    def lon_range(self):
        """Return an array containing all the longitudinal points on the grid.

        Returns
        -------
        np.array(shape=(nx,), dtype=float)
        """
        return np.array([self.xmin + i * self.dx for i in range(self.nx)])

    def lat_range(self):
        """Return an array containing all the latitudinal points on the grid.

        Returns
        -------
        np.array(shape=(ny,), dtype=float)
        """
        return np.array([self.ymin + j * self.dy for j in range(self.ny)])

    def _LV03_to_WGS84(self, y, x):
        """Convert LV03 to WSG84.

        Based on swisstopo approximated solution (0.1" accuracy)

        For better comparability with other implementations, here:
        x <-> Northing
        y <-> Easting,
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
            + 0.1306 * y * x ** 2
            - 0.0436 * y ** 3
        ) / 0.36

        lat = (
            16.9023892
            + 3.238272 * x
            - 0.270978 * y ** 2
            - 0.002528 * x ** 2
            - 0.0447 * y ** 2 * x
            - 0.0140 * x ** 3
        ) / 0.36

        return lon, lat


class COSMOGrid(Grid):
    """Class to manage a COSMO-domain"""

    nx: int
    ny: int
    dx: float
    dy: float
    xmin: float
    ymin: float
    pollon: float
    pollat: float

    def __init__(self, nx, ny, dx, dy, xmin, ymin, pollon, pollat):
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
        areas = np.array(
            [
                [
                    dd * np.cos(np.deg2rad(self.ymin) + j * dlat)
                    for j in range(self.ny)
                ]
                for _ in range(self.nx)
            ]
        )
        return areas

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
        x = self.xmin + i * self.dx
        y = self.ymin + j * self.dy
        dx2 = self.dx / 2
        dy2 = self.dy / 2

        cell_x = np.array([x + dx2, x + dx2, x - dx2, x - dx2])
        cell_y = np.array([y + dy2, y - dy2, y - dy2, y + dy2])

        return cell_x, cell_y

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
            the indices of the cosmo grid cell containing the source
        """
        point = self.projection.transform_point(lon, lat, proj)

        indx = np.floor((point[0] - self.xmin) / self.dx)
        indy = np.floor((point[1] - self.ymin) / self.dy)

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
        # Here we assume a flat earth. The error is less than 1% for typical
        # grid sizes over europe. Since we're interested in the ratio of areas,
        # we can calculate in degrees^2.
        inv_cell = Polygon(corners)

        # Find around which cosmo grid index the inventory cell lies.
        # Since the inventory cell is in general not rectangular because
        # of different projections, we add a margin of to the extremal indices.
        # This way we're sure we don't miss any intersection.
        cell_xmin = min(k[0] for k in corners)
        cell_xmax = max(k[0] for k in corners)
        lon_idx_min = int((cell_xmin - self.xmin) / self.dx) - 2
        lon_idx_max = int((cell_xmax - self.xmin) / self.dx) + 3

        cell_ymin = min(k[1] for k in corners)
        cell_ymax = max(k[1] for k in corners)
        lat_idx_min = int((cell_ymin - self.ymin) / self.dy) - 2
        lat_idx_max = int((cell_ymax - self.ymin) / self.dy) + 3

        if (
            lon_idx_max < 0
            or lat_idx_max < 0
            or lon_idx_min > self.nx
            or lat_idx_min > self.ny
        ):
            # The inventory cell lies outside the cosmo grid
            return []

        intersections = []
        # make sure we iterate only over valid gridpoint indices
        for i in range(max(0, lon_idx_min), min(self.nx, lon_idx_max)):
            for j in range(max(0, lat_idx_min), min(self.ny, lat_idx_max)):
                cosmo_cell = Polygon(list(zip(*self.cell_corners(i, j))))

                if cosmo_cell.intersects(inv_cell):
                    overlap = cosmo_cell.intersection(inv_cell)
                    intersections.append((i, j, overlap.area / inv_cell.area))

        return intersections