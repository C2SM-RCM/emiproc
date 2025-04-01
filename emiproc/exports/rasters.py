from __future__ import annotations
from os import PathLike
from pathlib import Path
import xarray as xr
import numpy as np
from datetime import date
from emiproc.inventories import Inventory
from emiproc.grids import RegularGrid
from emiproc.regrid import remap_inventory
from emiproc.exports.netcdf import NetcdfAttributes, nc_cf_attributes
from emiproc.utilities import Units, SEC_PER_YR, PER_CELL_UNITS, PER_M2_UNITS


def export_raster_netcdf(
    inv: Inventory,
    path: PathLike,
    grid: RegularGrid | None = None,
    netcdf_attributes: NetcdfAttributes | None = None,
    weights_path: PathLike | None = None,
    lon_name: str = "lon",
    lat_name: str = "lat",
    var_name_format: str | None = None,
    unit: Units = Units.KG_PER_YEAR,
    group_categories: bool = False,
    add_totals: bool = True,
    categories_description: dict[str, str] | None = None,
) -> Path:
    """Export the inventory to a netcdf file as a raster.

    This will first remap the invenotry to a raster file using
    :py:func:`emiproc.regrid.remap_inventory` and
    then export the result to a netcdf file.

    :param inv: the inventory to export
    :param path: the path to the output file
    :param grid: the raster grid to export to
        If not given, assumes the inventory is already on a raster grid.
    :param netcdf_attributes: NetCDF attributes to add to the file.
        These can be generated using
        :py:func:`emiproc.exports.netcdf.nc_cf_attributes` .
    :param weights_path: Optionally,
        The path to the weights file to use for regridding.
        If not given, the weights will be calculated on the fly.
    :param lon_name: The name of the longitude dimension in the nc file.
    :param lat_name: The name of the latitude dimension in the nc file.
    :param var_name_format: The format string to use for the variable names.
        The format string should contain two named fields: ``substance`` and ``category``.
    :param unit: The unit of the emissions.
    :param group_categories: If True the categories will be grouped in the output file.
        Intead of a variable for each pair (substance, category) there will be a variable
        only for each substance and the categories will be grouped in a dimension.
    :param add_totals: If True, the total emissions for each substance will be added
        as new variables to the file.
        One will be the raster sum of all categories and the other will be the total sum
        over all cells.
    :param categories_description: A dictionary with a detailed description
        of each categories.


    """

    if var_name_format is None:
        if group_categories:
            var_name_format = "{substance}"
        else:
            var_name_format = "{substance}_{category}"

    if categories_description is not None:
        # Check that all categories are described
        missing_categories = set(inv.categories) - set(categories_description.keys())
        if missing_categories:
            raise ValueError(
                "Categories are missing in `categories_description`: "
                f"{missing_categories}"
            )

    if grid is None:
        grid = inv.grid
    if grid != inv.grid:
        inv = remap_inventory(inv, grid, weights_path)

    if netcdf_attributes is None:
        netcdf_attributes = nc_cf_attributes()
    # add the history
    netcdf_attributes["emiproc_history"] = str(inv.history)

    crs = grid.crs

    if unit == Units.KG_PER_YEAR:
        conversion_factor = 1.0
    elif unit == Units.KG_PER_M2_PER_S:
        conversion_factor = (
            1 / SEC_PER_YR / np.array(grid.cell_areas).reshape(grid.shape).T
        )
    elif unit == Units.MUG_PER_M2_PER_S:
        conversion_factor = (
            1 / SEC_PER_YR / np.array(grid.cell_areas).reshape(grid.shape).T
        ) * 1e9
    else:
        raise NotImplementedError(f"Unknown {unit=}")

    unit_str = str(unit.value)
    if unit in PER_CELL_UNITS:
        unit_str += " cell-1"

    ds = xr.Dataset(
        data_vars=(
            {
                var_name_format.format(substance=sub, category=cat): (
                    [lat_name, lon_name],
                    inv.gdf[(cat, sub)].to_numpy().reshape(grid.shape).T
                    * conversion_factor,
                    {
                        "standard_name": f"{sub}_{cat}",
                        "long_name": f"{sub}_{cat}",
                        "units": unit_str,
                        "comment": (
                            categories_description[cat]
                            if categories_description
                            else f"emissions of {sub} from {cat}"
                        ),
                        "projection": f"{crs}",
                    },
                )
                for sub in inv.substances
                for cat in inv.categories
                if (cat, sub) in inv.gdf
            }
            if not group_categories
            else {
                var_name_format.format(substance=sub): (
                    ["category", lat_name, lon_name],
                    np.array(
                        [
                            (
                                inv.gdf[(cat, sub)].to_numpy().reshape(grid.shape).T
                                if (cat, sub) in inv.gdf
                                else np.zeros(grid.shape).T
                            )
                            for cat in inv.categories
                        ]
                    )
                    * conversion_factor,
                    {
                        "standard_name": f"tendency_of_atmosphere_mass_content_of_{sub}_due_to_emission",
                        "long_name": f"{sub}",
                        "units": unit_str,
                        "comment": f"emissions of {sub}",
                        "projection": f"{crs}",
                    },
                )
                for sub in inv.substances
            }
        ),
        coords={
            "substance": inv.substances,
            "category": inv.categories,
            # Grid coordinates
            lon_name: (
                lon_name,
                grid.lon_range,
                {
                    "standard_name": "longitude",
                    "long_name": "longitude",
                    "units": "degrees_east",
                    "comment": "center_of_cell",
                    "bounds": f"{lon_name}_bnds",
                    "projection": f"{crs}",
                    "axis": "X",
                },
            ),
            lat_name: (
                lat_name,
                grid.lat_range,
                {
                    "standard_name": "latitude",
                    "long_name": "latitude",
                    "units": "degrees_north",
                    "comment": "center_of_cell",
                    "bounds": f"{lat_name}_bnds",
                    "projection": f"{crs}",
                    "axis": "Y",
                },
            ),
            f"{lon_name}_bnds": (
                f"{lon_name}_bnds",
                grid.lon_bounds,
                {
                    "standard_name": "longitude",
                    "long_name": "longitude of cell boundaries",
                    "units": "degrees_east",
                    "comment": "cell_boundaries",
                    "axis": "X",
                },
            ),
            f"{lat_name}_bnds": (
                f"{lat_name}_bnds",
                grid.lat_bounds,
                {
                    "standard_name": "latitude",
                    "long_name": "latitude of cell boundaries",
                    "units": "degrees_north",
                    "comment": "cell_boundaries",
                    "axis": "Y",
                },
            ),
        },
        attrs=netcdf_attributes,
    )

    if inv.year is not None:
        ds["time"] = (
            "time",
            np.array([date(inv.year, 7, 1)], dtype="datetime64"),
            {
                "axis": "T",
                "standard_name": "time",
                "long_name": "Inventory year",
                "comment": "Midpoint of the year",
            },
        )

        ds.attrs["year"] = inv.year
    else:
        ds.attrs["year"] = "not specified in inventory.year"

    if categories_description is not None:
        # Add a variable for the description of the categories
        ds["categories_description"] = (
            "category",
            [categories_description[cat] for cat in inv.categories],
            {
                "long_name": "Description of the categories",
                "comment": "Description of the categories",
            },
        )

    if add_totals:
        for sub in inv.substances:
            if group_categories:
                da = ds[var_name_format.format(substance=sub)].sum("category")
            else:
                var_names = [
                    var_name_format.format(substance=sub, category=cat)
                    for cat in inv.categories
                    if var_name_format.format(substance=sub, category=cat)
                    in ds.data_vars
                ]
                da = sum([ds[name] for name in var_names])

            ds[f"emi_{sub}_all_sectors"] = da
            ds[f"emi_{sub}_all_sectors"].attrs = {
                "standard_name": f"tendency_of_atmosphere_mass_content_of_{sub}_due_to_emission",
                "long_name": f"Aggregated Emissions of {sub} from all sectors",
                "units": unit_str,
                "comment": "annual mean emission rate",
                "projection": f"{crs}",
            }
            if group_categories:
                total_emission = ds[var_name_format.format(substance=sub)]
            else:
                da_of_cat = {
                    cat: (
                        ds[var_name_format.format(substance=sub, category=cat)]
                        if var_name_format.format(substance=sub, category=cat)
                        in ds.data_vars
                        else xr.zeros_like(da)
                    )
                    for cat in inv.categories
                }
                total_emission = xr.concat(list(da_of_cat.values()), dim="category")
            # Total emission is not weighted by the cell area
            # So we always give kg/year
            if unit == Units.KG_PER_M2_PER_S:
                total_emission = (
                    total_emission
                    * np.array(grid.cell_areas).reshape(grid.shape).T
                    * SEC_PER_YR
                )
            elif unit == Units.KG_PER_YEAR:
                pass
            elif unit == Units.MUG_PER_M2_PER_S:
                total_emission = (
                    total_emission
                    * np.array(grid.cell_areas).reshape(grid.shape).T
                    * SEC_PER_YR
                    * 1e-9
                )
            else:
                raise NotImplementedError(f"Unknown {unit=}")
            ds[f"emi_{sub}_total"] = total_emission.sum([lon_name, lat_name])
            ds[f"emi_{sub}_total"].attrs = {
                "long_name": f"Total Emissions of {sub}",
                "units": "kg yr-1",
                "comment": "annual total emission",
            }

    # add the cell area
    ds["cell_area"] = (
        [lat_name, lon_name],
        np.array(grid.cell_areas).reshape(grid.shape).T,
        {
            "standard_name": "cell_area",
            "long_name": "cell_area",
            "units": "m2",
            "comment": "area of the cell",
            "projection": f"{crs}",
        },
    )
    path = Path(path)
    out_filepath = path.with_suffix(".nc")
    ds.to_netcdf(out_filepath)

    return out_filepath
