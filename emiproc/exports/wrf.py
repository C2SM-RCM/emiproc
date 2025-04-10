"""Functions related to the WRF model."""

from __future__ import annotations

import itertools
import os
from datetime import datetime
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from shapely.creation import polygons

from emiproc.exports.utils import get_temporally_scaled_array
from emiproc.grids import WGS84, Grid, RegularGrid
from emiproc.inventories import Inventory
from emiproc.utilities import HOUR_PER_DAY, get_day_per_year
from emiproc.utils.constants import get_molar_mass


class WRF_Grid(RegularGrid):
    """Grid of the wrf model.

    The grid is a pseudo regular grid, in the sense that the grid is regular
    under a certain projection, but is never given in that projection, but on a
    WG84 projection.

    The grid is constucted from the wrfinput file.

    """

    attributes: dict[any, any]

    def __init__(self, grid_filepath: PathLike):
        """Initialize the grid.

        Parameters
        ----------
        grid_filepath : Pathlike
            The path to the grid file.
        """

        grid_filepath = Path(grid_filepath)
        Grid.__init__(self, name=grid_filepath.stem, crs=WGS84)

        ds = xr.open_dataset(grid_filepath, engine="netcdf4")
        self.attributes = ds.attrs

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

    def __repr__(self) -> str:
        return f"WRF_grid({self.name})_nx({self.nx})_ny({self.ny})"


def export_wrf_hourly_emissions(
    inv: Inventory,
    grid: WRF_Grid,
    time_range: tuple[datetime | str, datetime | str],
    output_dir: PathLike,
    variable_name: str = "E_{substance}_{category}",
) -> Path:
    """Export the inventory to WRF chemi files.

    Output units are in mole/km2/hour.

    .. note::
        When running this function on Windows, the files will be saved with the
        `:` replaced by `-` in the file name, because Windows does not allow `:` in
        file names.

    :param inv: the inventory to export
    :param grid: the grid of the WRF model. The inventory must be on this grid.
    :param time_range: the time range to export the inventory.
    :param output_dir: the directory where to save the files.
    :param variable_name: the name of the variable in the netcdf file.
        You can use the following placeholders:
        - {substance}
        - {category}
        example: "E_{substance}_{category}"

    :return: the directory where the files are saved.

    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the inventory is on the same grid as the WRF grid
    assert inv.grid == grid, "The inventory and the grid are not the same"

    # Create the time axis
    time_range = pd.date_range(time_range[0], time_range[1], freq="h")

    da = get_temporally_scaled_array(inv, time_range, sum_over_cells=False)

    # Molar mass conversion mol / kg
    mm_factor = 1 / (
        xr.DataArray([get_molar_mass(sub) for sub in inv.substances], dims="substance")
        * 1e-3
    )
    # year / hour
    temporal_conversion = 1 / float(get_day_per_year(inv.year) * HOUR_PER_DAY)
    # km2 / cell (km2/m2 * m2/cell)
    spatial_conversion = xr.DataArray(1e-6 / grid.cell_areas, dims="cell")

    da = da * mm_factor * temporal_conversion * spatial_conversion

    # Unstack the datarray to get on the regular 2D grid
    shape = grid.shape
    x_index = np.arange(shape[0])
    y_index = np.arange(shape[1])
    da = da.assign_coords(
        x=("cell", np.repeat(x_index, shape[1])),
        y=("cell", np.tile(y_index, shape[0])),
    )
    mindex_coords = xr.Coordinates.from_pandas_multiindex(
        pd.MultiIndex.from_arrays([da.x.values, da.y.values], names=["x", "y"]), "cell"
    )
    da = da.assign_coords(mindex_coords)
    da = da.unstack("cell")
    # Rename the dimensions to match the WRF grid
    da = da.rename({"x": "west_east", "y": "south_north"})

    for dt in time_range:
        variables = []
        for cat, sub in itertools.product(inv.categories, inv.substances):
            this_da = (
                da.sel(category=cat, substance=sub, time=dt)
                # Name the variable
                .rename(variable_name.format(substance=sub, category=cat))
                .drop_vars(["substance", "category"])
                # Add the time dimension as a one element dimension
                .expand_dims("Time")
                .expand_dims("emissions_zdim")
            )

            variables.append(this_da)

        ds_at_hour = (
            xr.merge(variables)
            # Transpose to have the dims in the right order
            .transpose("Time", "emissions_zdim", "south_north", "west_east")
            # Progagate the default attributes
            .assign_attrs(grid.attributes)
            # Add emiproc specific attributes
            .assign_attrs(
                {
                    "emiproc": f"This file was created by emiproc on {datetime.now()}",
                    "emiproc_history": f"Created from the inventory {inv.name} with {inv.history=}",
                    "unit": "moles/km2/h",
                }
            )
        )

        # Save the dataset
        str_format = "%Y-%m-%d_%H:%M:%S"

        ds_at_hour["Times"] = ("Time", [dt.strftime(str_format).encode()])

        # If windows, we cannot have : in the file name
        if os.name == "nt":
            str_format = "%Y-%m-%d_%H-%M-%S"
        file_name = output_dir / f"wrfchemi_d01_{dt.strftime(str_format)}"
        ds_at_hour.to_netcdf(file_name)

    return output_dir
