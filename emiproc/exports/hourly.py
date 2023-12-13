from __future__ import annotations

import logging
from datetime import datetime
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from emiproc import PROCESS
from emiproc.exports.netcdf import NetcdfAttributes
from emiproc.grids import RegularGrid
from emiproc.inventories import Inventory
from emiproc.profiles.temporal_profiles import create_scaling_factors_time_serie
from emiproc.profiles.utils import get_desired_profile_index
from emiproc.regrid import remap_inventory
from emiproc.utilities import HOUR_PER_YR, PER_M2_UNITS, SEC_PER_YR, Units

logger = logging.getLogger(__name__)


def export_hourly_emissions(
    inv: Inventory,
    path: PathLike,
    start_time: datetime,
    end_time: datetime,
    netcdf_attributes: NetcdfAttributes,
    var_name_format: str = "{substance}_{category}",
    filename_format: str = "%Y%m%dT%H%M%SZ.nc",
    unit: Units = Units.KG_PER_HOUR,
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

    # Create the scaling factors for all the time profiles
    reqired_profiles_indexes = np.unique(inv.t_profiles_indexes)
    df_scaling_factors = pd.DataFrame(
        {
            index: create_scaling_factors_time_serie(
                start_time=start_time,
                end_time=end_time,
                profiles=inv.t_profiles_groups[index],
            )
            for index in reqired_profiles_indexes
            if index >= -1
        }
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
    if unit in PER_M2_UNITS:
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

    # Iterrate over time
    for dt, row in df_scaling_factors.iterrows():
        ds = base_ds.copy()
        ds["time"] = dt
        vars = {}
        for cat in inv.categories:
            for sub in inv.substances:
                # Get the scaling factor
                try:
                    index = get_desired_profile_index(
                        inv.t_profiles_indexes, cat=cat, sub=sub
                    )
                except ValueError as ve:
                    logger.warning(
                        f"Could not find profile for {cat=} {sub=}: {ve} \n Assuming"
                        " constant profile"
                    )
                    index = -1

                if index == -1:
                    scaling_factor = 1.0
                else:
                    scaling_factor = row[index]
                if (cat, sub) not in inv.gdf.columns:
                    # Ignore non present cat-sub
                    continue
                # Get the emissions
                emissions = inv.gdf[(cat, sub)].to_numpy().astype(float)
                # Multiply by the scaling factor
                emissions *= scaling_factor * conversion_factor
                name = var_name_format.format(substance=sub, category=cat)

                if is_regular_grid:
                    emissions = emissions.reshape(grid.shape).T

                vars[name] = xr.DataArray(
                    emissions,
                    dims=data_dim,
                    attrs={
                        "standard_name": f"{sub}_{cat}",
                        "long_name": f"{sub}_{cat}",
                        "units": str(unit.value),
                        "comment": f"emissions of {sub} in {cat}",
                    },
                    name=name,
                )
        # Add to the dataset
        ds.update(vars)
        dt: pd.Timestamp

        ds.to_netcdf(path / f"{dt.strftime(filename_format)}")
