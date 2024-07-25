"""Functions related to the WRF model."""

from os import PathLike
from pathlib import Path
from emiproc.grids import Grid, WGS84
import xarray as xr
import numpy as np
from shapely.creation import polygons


class WRF_Grid(Grid):
    """Grid of the wrf model.

    The grid is a pseudo regular grid, in the sense that the grid is regular
    under a certain projection, but is never given in that projection, but on a
    WG84 projection.

    The grid is constucted from the wrfinput file.

    """

    def __init__(self, grid_filepath: PathLike):
        """Initialize the grid.

        Parameters
        ----------
        grid_filepath : Pathlike
            The path to the grid file.
        """

        grid_filepath = Path(grid_filepath)
        super().__init__(name=grid_filepath.stem, crs=WGS84)

        ds = xr.open_dataset(grid_filepath, engine="netcdf4")

        # This will be necessary to reshape the arrays to a 1D array following the
        # emiproc convention
        reshape = lambda x: x.T.reshape(-1)

        # Access the grid coordinates
        center_lon = reshape(ds["XLONG"].isel(Time=0).values)
        center_lat = reshape(ds["XLAT"].isel(Time=0).values)

        self.nx = ds.sizes["west_east"]
        self.ny = ds.sizes["south_north"]

        # Grid vertices are given not at the vertices but at edges
        # It is the place where the wind beteween two cells is calculated (thus U and V)
        lon_u = ds["XLONG_U"].isel(Time=0).values
        lon_v = ds["XLONG_V"].isel(Time=0).values
        left_lon_u = reshape(lon_u[:, :-1])
        right_lon_u = reshape(lon_u[:, 1:])
        bottom_lon_v = reshape(lon_v[:-1, :])
        top_lon_v = reshape(lon_v[1:, :])

        lat_u = ds["XLAT_U"].isel(Time=0).values
        lat_v = ds["XLAT_V"].isel(Time=0).values
        bottom_lat_v = reshape(lat_v[:-1, :])
        top_lat_v = reshape(lat_v[1:, :])
        left_lat_u = reshape(lat_u[:, :-1])
        right_lat_u = reshape(lat_u[:, 1:])

        # Calculate the offsets, to be able to reconstruct the grid vertices
        d_lon_right = right_lon_u - center_lon
        d_lon_left = left_lon_u - center_lon
        d_lon_top = top_lon_v - center_lon
        d_lon_bottom = bottom_lon_v - center_lon

        d_lat_right = right_lat_u - center_lat
        d_lat_left = left_lat_u - center_lat
        d_lat_top = top_lat_v - center_lat
        d_lat_bottom = bottom_lat_v - center_lat

        # Reconstruct the grid vertices
        coords = np.array(
            [
                # Bottom left
                [
                    center_lon + d_lon_left + d_lon_bottom,
                    center_lat + d_lat_left + d_lat_bottom,
                ],
                # Bottom right
                [
                    center_lon + d_lon_right + d_lon_bottom,
                    center_lat + d_lat_right + d_lat_bottom,
                ],
                # Top right
                [
                    center_lon + d_lon_right + d_lon_top,
                    center_lat + d_lat_right + d_lat_top,
                ],
                # Top left
                [
                    center_lon + d_lon_left + d_lon_top,
                    center_lat + d_lat_left + d_lat_top,
                ],
            ]
        )

        coords = np.rollaxis(coords, -1, 0)

        # Create the polygons
        polys = polygons(coords)

        self.cells_as_polylist = polys
