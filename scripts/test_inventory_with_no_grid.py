"""Same a test_inventories, but for a inv without the gdf, only gdfs
"""


import pandas as pd
import geopandas as gpd

from shapely.geometry import Point, MultiPolygon, Polygon
from emiproc.inventories.utils import crop_with_shape
from emiproc.plots import explore_inventory, explore_multilevel
from emiproc.utilities import ProgressIndicator
from emiproc.regrid import (
    calculate_weights_mapping,
    geoserie_intersection,
    get_weights_mapping,
    remap_inventory,
)
from emiproc.inventories import Inventory
from emiproc.grids import GeoPandasGrid
from emiproc.inventories.utils import group_categories

#Create the geometetries of an inventory
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

# Check the intersection
cropped, weights = geoserie_intersection(
    serie, triangle, keep_outside=True, drop_unused=False
)

# %% Generate a 'template' inventory

inv = Inventory.from_gdf(
    gdf=None,
    gdfs={
        "adf": gpd.GeoDataFrame(
            {
                "CH4": [i + 3 for i in range(len(serie))],
                "CO2": [i for i in range(len(serie))],
                "NH3": [i + 1 for i in range(len(serie))],
            },
            geometry=serie,
        ),
        "liku": gpd.GeoDataFrame(
            {
                "CO2": [i +3 for i in range(len(serie))],
                "NH3": [i + 1 for i in range(len(serie))],
            },
            geometry=serie,
        )
    },
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
explore_inventory(inv_with_pnt_sources, category="liku", substance="CO2")
# %%
cropped = crop_with_shape(inv_with_pnt_sources, triangle, keep_outside=False)
# %%
explore_inventory(cropped, category="liku", substance="CO2")
# %%
inv.substances, inv_with_pnt_sources.substances
# %% Create a grid for remapping

# Few trangles indide the grid
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

# Squareas way bigger than grid
test_remap_grid_big = GeoPandasGrid(
    gpd.GeoDataFrame(
        geometry=[
            Polygon(((-1, -1), (-1, 2), (1, 2), (1, -1))),
            Polygon(((5, -1), (5, 2), (1, 2), (1, -1))),
        ]
    )
)



groupped_inv = group_categories(
    inv_with_pnt_sources, {"New": ["liku", "adf", "other"], "other": ["blek"]}
)

# TODO: make sure the nan values are 0 instead
# %%
inv2 = remap_inventory(groupped_inv, test_remap_grid, weigths_file=".weightstest2")
# %%
import importlib
import emiproc.inventories.utils

importlib.reload(emiproc.inventories.utils)
from emiproc.inventories.utils import add_inventories

added_inv = add_inventories(groupped_inv, inv_with_pnt_sources)

# %%
