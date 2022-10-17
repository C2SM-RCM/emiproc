#%%
import pandas as pd
from pathlib import Path
import numpy as np
import geopandas as gpd
from typing import Any, Iterable
from shapely.geometry import Point, MultiPolygon, Polygon
from emiproc.inventories.utils import add_inventories, crop_with_shape
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

xmin = 0
dx = 1
nx = 10
ymin = 0
dy = 0.5
ny = 20

xs = np.arange(
    xmin,
    xmin + nx * dx,
    step=dx,
)
ys = np.arange(
    ymin,
    ymin + ny * dy,
    step=dy,
)


grid = gpd.GeoSeries(
    [
        Polygon(
            (
                (x, y),
                (x, y + dy),
                (x + dx, y + dy),
                (x + dx, y),
            )
        )
        for y in reversed(ys)
        for x in xs
    ]
)

poly = Polygon(((5.5, 7.5), (5.5, 3.5), (2.5, 3.5), (2.5, 8.5)))

# intersected_shapes , weigths = geoserie_intersection(
#     grid, poly, keep_outside=True, drop_unused=True
# )
# 
# inter_df = gpd.GeoDataFrame({'weigths': weigths}, geometry=[s for s in intersected_shapes])
# inter_df.explore('weigths')

#TODO: write the tests properly

def test_basic_crop():
    intersected_shapes , weigths = geoserie_intersection(
        grid, poly, keep_outside=False, drop_unused=False
    )


def test_with_modify_grid():
    intersected_shapes , weigths = geoserie_intersection(
        grid, poly, keep_outside=False, drop_unused=False
    )
