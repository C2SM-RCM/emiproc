from __future__ import annotations

import logging
from datetime import datetime
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from emiproc import PROCESS
from emiproc.exports.netcdf import NetcdfAttributes, nc_cf_attributes
from emiproc.exports.utils import get_temporally_scaled_array
from emiproc.grids import RegularGrid
from emiproc.inventories import Inventory
from emiproc.utilities import HOUR_PER_YR, PER_M2_UNITS, SEC_PER_YR, Units

logger = logging.getLogger(__name__)


def export_hourly_emissions(
    inv: Inventory,
    path: PathLike,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    netcdf_attributes: NetcdfAttributes = nc_cf_attributes(),
    var_name_format: str = "{substance}_{category}",
    filename_format: str = "%Y%m%dT%H%M%SZ.nc",
    unit: Units = Units.KG_PER_HOUR,
    freq: str = "h",
    inclusive: str = "both",
    chunk_size: int = 168,
) -> Path:
    """Export the inventory to hourly netcdf files.

    Supports structured and unstructured grids. Supports gridded emissions
    and point sources.

    A file given at a specific hour is valid for the whole hour.
    (ex file for 14h00 is valid from 14h00 to 14h59m59s)

    :param inv: the inventory to export
    :param path: the path to the output directory
    :param netcdf_attributes: NetCDF attributes to add to the file.
        These can be generated using
        :py:func:`emiproc.exports.netcdf.nc_cf_attributes` .
    :param weights_path: Optionally,
        The path to the weights file to use for regridding.
        If not given, the weights will be calculated on the fly.
    :param var_name_format: The format string to use for the variable names.
        The format string should contain two named fields: ``substance`` and ``category``.
    :param filename_format: The format string to use for the file names.
        The format string should contain fields for date and time.
    :param unit: The unit of the emissions.
    :param start_time: The start time of the output.
    :param end_time: The end time of the output.
    :param freq: The frequency of the output.


    """
    # Check if the inventory is gridded
    if inv.gdfs:
        raise NotImplementedError("Shapped sources are not implemented yet")

    if inv.t_profiles_indexes is None or inv.t_profiles_groups is None:
        raise ValueError(
            "The inventory does not contain temporal profiles required for hourly"
            " exports."
        )
    for invalid_dim in ["type", "country", "cell"]:
        if invalid_dim in inv.t_profiles_indexes:
            raise ValueError(f"Temporal profiles with {invalid_dim=} are not supported")

    grid = inv.grid
    is_regular_grid = isinstance(grid, RegularGrid)
    crs = grid.crs

    # add the history
    netcdf_attributes["emiproc_history"] = str(inv.history)
    netcdf_attributes["projection"] = f"{crs}"

    if unit == Units.KG_PER_YEAR:
        conversion_factor = 1.0
    elif unit == Units.KG_PER_HOUR:
        conversion_factor = 1.0 / HOUR_PER_YR
    elif unit == Units.KG_PER_M2_PER_S:
        conversion_factor = 1.0 / SEC_PER_YR / np.array(grid.cell_areas)
    else:
        raise NotImplementedError(f"Unknown {unit=}")

    if start_time is None:
        if inv.year is None:
            raise ValueError(
                "The inventory does not have a year. You need to set the start_time"
                " and end_time."
            )
        start_time = pd.Timestamp(f"{inv.year}-01-01 00:00:00")
    if end_time is None:
        assert inv.year is not None, "The inventory does not have a year."
        end_time = pd.Timestamp(f"{inv.year+1}-01-01 00:00:00")

    time_range = pd.date_range(
        start=start_time, end=end_time, freq=freq, inclusive=inclusive
    )

    coords = {
        "substance": inv.substances,
        "category": inv.categories,
        "cell": np.arange(len(grid)),
    }

    if is_regular_grid:
        coords["lat"] = (
            "lat",
            grid.lat_range,
            {
                "standard_name": "latitude",
                "long_name": "latitude",
                "units": "degrees_north",
                "comment": "center_of_cell",
                "bounds": "lat_bnds",
                "projection": f"{grid.crs}",
                "axis": "Y",
            },
        )
        coords["lon"] = (
            "lon",
            grid.lon_range,
            {
                "long_name": "longitude",
                "units": "degrees_east",
                "standard_name": "longitude",
                "comment": "center_of_cell",
                "bounds": "lon_bnds",
                "projection": f"{grid.crs}",
                "axis": "X",
            },
        )

    data_dim = ["lat", "lon"] if is_regular_grid else ["cell"]

    base_ds = xr.Dataset(
        coords=coords,
        attrs=netcdf_attributes,
    )
    if grid.crs:
        # add the cell area
        areas = np.array(grid.cell_areas).reshape(grid.shape).T
        if not is_regular_grid:
            areas = areas.reshape(-1)
        base_ds["cell_area"] = (
            data_dim,
            areas,
            {
                "standard_name": "cell_area",
                "long_name": "cell_area",
                "units": "m2",
                "comment": "area of the cell",
                "projection": f"{crs}",
            },
        )
    path = Path(path)
    logger.log(PROCESS, f"Exporting hourly emissions to {path}")

    # split the time range into chunks
    time_chunks = [
        time_range[i : i + chunk_size] for i in range(0, len(time_range), chunk_size)
    ]
    # Iterrate over time
    for sub_time_range in time_chunks:
        da = get_temporally_scaled_array(
            inv=inv, time_range=sub_time_range, sum_over_cells=False
        )

        # Multiply by the conversion factor
        da *= conversion_factor

        if is_regular_grid:
            shape = grid.shape
            da = da.assign_coords(
                lon=("cell", np.repeat(grid.lon_range, shape[1])),
                lat=("cell", np.tile(grid.lat_range, shape[0])),
            )
            mindex_coords = xr.Coordinates.from_pandas_multiindex(
                pd.MultiIndex.from_arrays(
                    [da.lon.values, da.lat.values], names=["lon", "lat"]
                ),
                "cell",
            )
            da = (
                da.assign_coords(mindex_coords).unstack("cell")
                # Reorder the dimensions to match (lat, lon, time)
                .transpose("lat", "lon", "time", "substance", "category")
            )

        for dt in sub_time_range:
            dt: pd.Timestamp

            ds = base_ds.copy()
            ds["time"] = [dt]
            vars = {}
            for cat in inv.categories:
                for sub in inv.substances:

                    name = var_name_format.format(substance=sub, category=cat)

                    emissions = da.sel(category=cat, substance=sub, time=[dt])

                    vars[name] = xr.DataArray(
                        emissions,
                        dims=data_dim + ["time"],
                        attrs={
                            "standard_name": f"{sub}_{cat}",
                            "long_name": f"{sub}_{cat}",
                            "units": str(unit.value),
                            "comment": f"emissions of {sub} from {cat}",
                        },
                        name=name,
                    )
            # Add to the dataset
            ds.update(vars)

            ds.to_netcdf(path / f"{dt.strftime(filename_format)}")
