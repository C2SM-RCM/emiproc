from __future__ import annotations
from os import PathLike
from pathlib import Path
import xarray as xr
import numpy as np
from emiproc.inventories import Inventory
from emiproc.grids import RegularGrid
from emiproc.profiles.utils import get_desired_profile_index
from emiproc.regrid import remap_inventory
from emiproc.exports.netcdf import NetcdfAttributes
from emiproc.utilities import Units, SEC_PER_YR



def export_raster_netcdf(
    inv: Inventory,
    path: PathLike,
    grid: RegularGrid,
    netcdf_attributes: NetcdfAttributes,
    weights_path: PathLike | None = None,
    lon_name: str = "lon",
    lat_name: str = "lat",
    var_name_format: str = "{substance}_{category}",
    unit: Units = Units.KG_PER_YEAR,
) -> Path:
    """Export the inventory to a netcdf file as a raster.

    This will first remap the invenotry to a raster file using
    :py:func:`emiproc.regrid.remap_inventory` and
    then export the result to a netcdf file.

    :param inv: the inventory to export
    :param path: the path to the output file
    :param grid: the raster grid to export to
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

    """

    remapped_inv = remap_inventory(inv, grid, weights_path)

    # add the history
    netcdf_attributes["emiproc_history"] = str(remapped_inv.history)

    crs = grid.crs

    if unit == Units.KG_PER_YEAR:
        conversion_factor = 1.0
    elif unit == Units.KG_PER_M2_PER_S:
        conversion_factor = (
            1 / SEC_PER_YR / np.array(grid.cell_areas).reshape(grid.shape).T
        )
    else:
        raise NotImplementedError(f"Unknown {unit=}")

    ds = xr.Dataset(
        data_vars={
            var_name_format.format(substance=sub, category=cat): (
                [lat_name, lon_name],
                remapped_inv.gdf[(cat, sub)].to_numpy().reshape(grid.shape).T
                * conversion_factor,
                {
                    "standard_name": f"{sub}_{cat}",
                    "long_name": f"{sub}_{cat}",
                    "units": str(unit.value),
                    "comment": f"emissions of {sub} in {cat}",
                    "projection": f"{crs}",
                },
            )
            for sub in inv.substances
            for cat in inv.categories
            if (cat, sub) in remapped_inv.gdf
        },
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
                    "bounds": "lon_bnds",
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
                    "bounds": "lat_bnds",
                    "projection": f"{crs}",
                    "axis": "Y",
                },
            ),
        },
        attrs=netcdf_attributes,
    )

    if unit in [Units.KG_PER_M2_PER_S]:
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
