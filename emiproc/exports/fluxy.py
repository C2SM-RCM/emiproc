import logging
from os import PathLike
from pathlib import Path
from emiproc.exports.utils import get_temporally_scaled_array
from emiproc.grids import RegularGrid
from emiproc.inventories import Inventory
import xarray as xr
import numpy as np
from emiproc.utilities import get_country_mask

from pyproj import CRS


logger = logging.getLogger(__name__)


def export_fluxy(
    invs: Inventory | list[Inventory],
    output_dir: PathLike,
    transport_model: str = "emiproc",
    frequency: str = "yearly",
    percentiles: list[float] = [0.159, 0.841],
) -> None:
    """Export emissions to Fluxy format.

    https://github.com/openghg/fluxy

    Fluxy is a python plotting tool for comparing inverse modeling results.
    As part of fluxy, it is possible to plot prior emissions.

    Fluxy files must have a specifc format
    `<inversionModel>_<transportModel>_<domain>_<prior>_<optional_tags>_<species>_<inversionFrequency>(_concentration).nc`.

    To handle this, this function uses the following parameters:

    - `inversionModel`: The name of the directory in which the files are stored.
    - `transportModel`: The `inversion_model` parameter.
    - `domain`: The name of the grid. This is the name of the grid in the inventory.
    - `prior`: The name of the inventory. This is the name of the inventory in the inventory.
    - `optional_tags`: The tags of the inventory. Nothing for now.
    - `species`: The name of the substance. This is the name of the substances in the inventory.
    - `inversionFrequency`: The frequency of the inventory.
        Currently only yearly is implemented.

    :param invs: Inventory or list of inventories to export.
        A list of inventory assumes that you want to plot the emissions over multiple years.
    :param output_dir: Directory to export the emissions to.
        This directory is the name of the transport model in fluxy.
    :param transport_model: The transport model name to "fake". (default: `emiproc`)

    :return: None


    """

    if "_" in transport_model:
        logger.warning(
            "The transport model name should not contain underscores. "
            f"Got {transport_model}. "
            "You might have issues with fluxy later on because of this."
        )

    output_dir = Path(output_dir) / transport_model
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(invs, Inventory):
        invs = [invs]
    if not isinstance(invs, list):
        raise TypeError(f"Expected Inventory or list of Inventory, got {type(invs)}.")
    if len(invs) == 0:
        raise ValueError("Expected at least one inventory.")

    names = [inv.name for inv in invs]
    # Check that all names are the same
    if len(set(names)) != 1:
        raise ValueError("All inventories must have the same name. " f"Got {names}.")
    inventory_name = names[0]
    inversion_model = output_dir.name

    # Check that the grids are all the same
    grids = [inv.grid for inv in invs]
    if len(set(grids)) != 1:
        raise ValueError(
            "All inventories must have the same grid. "
            f"Got {[grid.name for grid in grids]}."
        )
    grid = grids[0]
    assert isinstance(grid, RegularGrid), "Only regular grids are supported."

    # Check the grid is on WGS84
    if not CRS(grid.crs) == CRS("WGS84"):
        raise ValueError("The grid must be on WGS84. " f"Got {grid.crs}.")

    def cell_to_lat_lon(ds: xr.Dataset):
        ds = ds.assign_coords(
            latitude=("cell", np.tile(grid.lat_range, grid.nx)),
            longitude=("cell", np.repeat(grid.lon_range, grid.ny)),
        )
        # Set the stacked coordinate as a multi-index
        ds = ds.set_index(cell=["latitude", "longitude"])
        ds = ds.unstack({"cell": ("latitude", "longitude")})
        return ds

    da_fractions = get_country_mask(grid, return_fractions=True)
    da_fractions = cell_to_lat_lon(da_fractions)

    domain_name = grid.name

    substances = set(sum((inv.substances for inv in invs), []))

    # First create a template with the grid and the time dimension
    ds_template = xr.Dataset(
        coords={
            "longitude": (
                "longitude",
                grid.lon_range,
                {
                    "standard_name": "longitude",
                    "long_name": "longitude of grid cell centre",
                    "units": "degrees_east",
                    "axis": "X",
                },
            ),
            "latitude": (
                "latitude",
                grid.lat_range,
                {
                    "standard_name": "latitude",
                    "long_name": "latitude of grid cell centre",
                    "units": "degrees_north",
                    "axis": "Y",
                },
            ),
            "country": (
                "country",
                da_fractions["country"].data,
            ),
        },
        data_vars={
            "country_fraction": (
                ("country", "latitude", "longitude"),
                da_fractions.data,
                {
                    "long_name": "fraction of grid cell associated to country",
                    "units": "1",
                    "comments": "calculated by emiproc",
                },
            ),
        },
    )

    dss = [
        get_temporally_scaled_array(
            inv,
            time_range=inv.year,
            sum_over_cells=False,
        )
        for inv in invs
    ]

    # Put all together and sum over the categories to have total emissions
    ds = xr.concat(dss, dim="time").sum(dim="category")

    ds = cell_to_lat_lon(ds)
    units = {"units": "kg year-1"}

    for sub in substances:
        sub_dir = output_dir / sub
        sub_dir.mkdir(parents=True, exist_ok=True)
        file_stem = "_".join(
            [
                transport_model,
                sub,
                frequency,
            ]
        )
        da_this = (
            ds.sel(substance=sub)
            .drop_vars("substance")
            .assign_attrs(units)
        )
        ds_this = ds_template.assign({"flux_total_prior": da_this})
        # Calculate the country flux
        ds_this = ds_this.assign(
            country_flux_total_prior=(
                ds_this["flux_total_prior"] * ds_this["country_fraction"]
            ).sum(dim=["latitude", "longitude"]).assign_attrs(units),
        )
        ds_this = ds_this.assign(
            percentile_flux_total_prior=xr.concat(
                [
                    ds_this["flux_total_prior"]
                    / ds_this["flux_total_prior"]
                    .sum(dim=["latitude", "longitude"])
                    .assign_coords(percentiles=p)
                    for p in percentiles
                ],
                dim="percentile",
            ),
            percentile_country_flux_total_prior=xr.concat(
                [
                    ds_this["country_flux_total_prior"]
                    / ds_this["country_flux_total_prior"]
                    .sum(dim=["country"])
                    .assign_coords(percentiles=p)
                    for p in percentiles
                ],
                dim="percentile",
            ),
        )

        ds_this = ds_this.assign(
            **{
                # Assign the same variable names as the prior for the posterior variables
                var_prior.replace("prior", "posterior"): ds_this[var_prior]
                for var_prior in [
                    "flux_total_prior",
                    "country_flux_total_prior",
                    "percentile_flux_total_prior",
                    "percentile_country_flux_total_prior",
                ]
            }
        )

        ds_this.to_netcdf(
            sub_dir / f"{file_stem}.nc",
        )

    logger.info(f"Exported {len(substances)} substances to {output_dir}.")
