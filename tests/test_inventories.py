#%%
import pandas as pd
from pathlib import Path
import numpy as np
import geopandas as gpd
from typing import Any, Iterable
from shapely.geometry import Point, MultiPolygon, Polygon
from emiproc.inventories.utils import crop_with_shape
from emiproc.utilities import ProgressIndicator
from emiproc.regrid import (
    calculate_weights_mapping,
    geoserie_intersection,
    get_weights_mapping,
    remap_inventory,
)
from emiproc.inventories import Inventory
from emiproc.grids import GeoPandasGrid

#%%
serie = gpd.GeoSeries(
    [
        Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),
        Polygon(((0, 1), (0, 2), (1, 2), (1, 1))),
        Polygon(((1, 0), (1, 1), (2, 1), (2, 0))),
        Polygon(((1, 1), (1, 2), (2, 2), (2, 1))),
        Polygon(((2, 1), (2, 2), (3, 2), (3, 1))),
    ]
)
triangle = Polygon(((0.5, 0.5), (1.5, 0.5), (1.5, 1.5)))
cropped, weights = geoserie_intersection(
    serie, triangle, keep_outside=True, drop_unused=False
)
#%%
# gdf = gpd.GeoDataFrame({"weights": weights}, geometry=cropped)
# gdf.explore("weights")
# %%

inv = Inventory.from_gdf(
    gpd.GeoDataFrame(
        {
            ("adf", "CH4"): [i + 3 for i in range(len(serie))],
            ("adf", "CO2"): [i for i in range(len(serie))],
            ("liku", "CO2"): [i for i in range(len(serie))],
            ("test", "NH3"): [i + 1 for i in range(len(serie))],
        },
        geometry=serie,
    )
)

inv.gdf

# %%

inv_with_pnt_sources = inv.copy()
inv_with_pnt_sources.gdfs["blek"] = gpd.GeoDataFrame(
    {
        "CO2": [1, 2, 3],
    },
    geometry=[Point(0.75, 0.75), Point(0.25, 0.25), Point(1.2, 1)],
)
inv_with_pnt_sources.gdfs["liku"] = gpd.GeoDataFrame(
    {
        "CO2": [1, 2],
    },
    geometry=[Point(0.65, 0.75), Point(1.1, 0.8)],
)
inv_with_pnt_sources.gdfs["other"] = gpd.GeoDataFrame(
    {
        "AITS": [1, 2],
    },
    geometry=[Point(0.65, 0.75), Point(1.1, 0.8)],
)

#%%


cropped = crop_with_shape(inv, triangle, keep_outside=False)
cropped
# %%


gdf = gpd.GeoDataFrame(
    columns=pd.MultiIndex.from_product([("cat1", "cat2"), ("sub", "sub2")]),
    geometry=serie,
)
gdf[("cat1", "sub2")] = 1
gdf
# %%
def explore_multilevel(gdf: gpd.GeoDataFrame, colum: Any, logscale: bool = False):
    """Explore a multilevel GeodataFrame."""
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


#%%
explore_multilevel(cropped.gdf, ("test", "NH3"))
# %%
explore_inventory(inv_with_pnt_sources, category="liku", substance="CO2")
# %%
cropped = crop_with_shape(inv_with_pnt_sources, triangle, keep_outside=False)
# %%
explore_inventory(cropped, category="liku", substance="CO2")
# %%
inv.substances, inv_with_pnt_sources.substances
# %%

test_remap_grid = GeoPandasGrid(
    gpd.GeoDataFrame(
        geometry=[
            Polygon(((0.5, 0.5), (0.5, 1.5), (1.5, 1.5))),
            Polygon(((0.5, 0.5), (1.5, 0.5), (1.5, 1.5))),
            Polygon(((2.5, 0.5), (1.5, 1.5), (1.5, 0.5))),
            Polygon(((2.5, 0.5), (2.5, 1.5), (1.5, 1.5))),
        ]
    )
)

# %%
remapped_inv = remap_inventory(
    inv_with_pnt_sources, test_remap_grid, weigths_file=".weightstest"
)
# %%
import importlib
import emiproc.regrid

importlib.reload(emiproc.regrid)

from emiproc.regrid import calculate_weights_mapping

calculate_weights_mapping(
    inv.gdf,
    #inv_with_pnt_sources.gdfs["blek"],
    test_remap_grid.gdf.geometry,
    loop_over_inv_objects=True,
)

# %%
