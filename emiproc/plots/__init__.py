import geopandas as gpd
from typing import Any
import numpy as np
import pandas as pd

from emiproc.inventories import Inventory


def explore_multilevel(gdf: gpd.GeoDataFrame, colum: Any, logscale: bool = False):
    """Explore a multilevel GeodataFrame.
    
    There is a bug with multilevel datframes that makes them impossible
    to call with the gpd.explore method.
    You can use this instead.
    """
    col_name = str(colum)
    data = gdf[colum]
    if logscale:
        data[data == 0] = np.nan
        data = np.log(data)
    gdf_plot = gpd.GeoDataFrame({col_name: data}, geometry=gdf.geometry)
    return gdf_plot.explore(gdf[colum])


def explore_inventory(
    inv: Inventory, category: None | str = None, substance: None | str = None
):
    """Explore the emission of an inventory."""
    # First check if the data is available
    if (
        category is not None
        and category not in inv.gdfs
        and category not in inv.gdf.columns
    ):
        raise IndexError(f"Category '{category}' not in inventory '{inv}'")
    if (
        substance is not None
        and all((substance not in gdf for gdf in inv.gdfs))
        and substance not in inv.gdf.columns.swaplevel(0, 1)
    ):
        raise IndexError(f"Substance '{substance}' not in inventory '{inv}'")
    if (
        substance is not None
        and category is not None
        and (category, substance) not in inv.gdf
        and (category not in inv.gdfs or substance not in inv.gdfs[category])
    ):
        raise IndexError(
            f"Substance '{substance}' for Category '{category}' not in inventory '{inv}'"
        )

    if category is None and substance is None:
        gdf = gpd.GeoDataFrame(
            geometry=pd.concat(
                [inv.geometry, *(gdf.geometry for gdf in inv.gdfs.values())]
            )
        )
        return gdf.explore()
    elif category is not None and substance is None:

        gdf = gpd.GeoDataFrame(
            geometry=pd.concat(
                ([inv.geometry] if category in inv.gdf.columns else [])
                + ([inv.gdfs[category].geometry] if category in inv.gdfs else [])
            )
        )
        return gdf.explore()
    elif category is not None and substance is not None:
        on_main_grid = (category, substance) in inv.gdf.columns
        on_others_gdfs = category in inv.gdfs and substance in inv.gdfs[category]
        gdf = gpd.GeoDataFrame(
            {
                str((category, substance)): pd.concat(
                    ([inv.gdf[(category, substance)]] if on_main_grid else [])
                    + ([inv.gdfs[category][substance]] if on_others_gdfs else [])
                )
            },
            geometry=pd.concat(
                ([inv.geometry] if on_main_grid else [])
                + ([inv.gdfs[category].geometry] if on_others_gdfs else [])
            ),
        )
        return gdf.explore(gdf[str((category, substance))])
    raise NotImplementedError()
