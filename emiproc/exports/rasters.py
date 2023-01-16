from __future__ import annotations
from os import PathLike
from pathlib import Path
import xarray as xr
import numpy as np
from emiproc.inventories import Inventory
from emiproc.grids import RegularGrid
from emiproc.regrid import remap_inventory
from emiproc.exports.netcdf import NetcdfAttributes
from emiproc.utilities import Units, SEC_PER_YR


def export_raster_netcdf(
    inv: Inventory,
    path: PathLike,
    grid: RegularGrid,
    netcdf_attributes: NetcdfAttributes,
    weights_path: PathLike | None = None,
    lon_name: str ="lon",
    lat_name: str ="lat",
    var_name_format: str ="{substance}_{category}",
    unit: Units = Units.KG_PER_YEAR,
    
):
    """Export to a netcdf file.

    # TODO: add the grid
    """

    remapped_inv = remap_inventory(inv, grid, weights_path)

    # add the history
    netcdf_attributes["emiproc_history"] = str(remapped_inv.history)

    crs = grid.crs

    if unit == Units.KG_PER_YEAR:
        conversion_factor = 1.
    elif unit == Units.KG_PER_M2_PER_S:
        conversion_factor = 1 / SEC_PER_YR / np.array(grid.cell_areas).reshape(grid.shape).T
    else:
        raise NotImplementedError(f"Unknown {unit=}")


    ds = xr.Dataset(
        data_vars={
            var_name_format.format(substance=sub, category=cat): (
                [lat_name, lon_name],
                remapped_inv.gdf[(cat, sub)].to_numpy().reshape(grid.shape).T * conversion_factor,
                {
                    "standard_name": f"{sub}_{cat}",
                    "long_name": f"{sub}_{cat}",
                    "units": str(unit.value),
                    "comment": f"emissions of {sub} in {cat}",
                    "projection": f"{crs}",
                }
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
    out_filepath = Path(path).with_suffix(".nc")
    ds.to_netcdf(out_filepath)
