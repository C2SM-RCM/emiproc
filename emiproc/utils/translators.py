from __future__ import annotations
import geopandas as gpd
import xarray as xr

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from emiproc.inventories import Inventory


def inv_to_xarray(inv: Inventory) -> xr.DataArray:
    """Convert the total emissions of an inventory to an xarray.

    :arg inv: The inventory to convert.

    :return array: A xarray containing the emissions at each coordinates:
        (substance, category, cell).
    """
    n_cells = len(inv.gdf)
    substances = inv.substances
    categories = inv.categories
    out_array = xr.DataArray(
        data=0.0,
        coords=dict(
            substance=substances,
            category=categories,
            cell=range(n_cells),
        ),
        dims=["substance", "category", "cell"],
    )

    if inv.gdfs:
        raise ValueError("The inventory cannot contain gdfs. Please remap it first.")

    for col in inv.gdf.columns:
        if isinstance(inv.gdf[col].dtype, gpd.array.GeometryDtype):
            continue
        cat, sub = col

        # Add this column to the array
        serie = inv.gdf.loc[:, col]
        out_array.loc[dict(category=cat, substance=sub)] = serie

    out_array.attrs["units"] = "kg/year/cell"
    out_array.attrs["description"] = f"Emissions of {inv.name}"
    out_array.attrs["history"] = inv.history + ["Converted to xarray."]
    out_array.attrs["source"] = "Created by emiproc.inv_to_xarray "

    return out_array
