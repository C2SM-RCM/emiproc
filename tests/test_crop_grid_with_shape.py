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
    geoserie_intersection,
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


inv = Inventory.from_gdf(
    gpd.GeoDataFrame({"val": np.linspace(0, 1, len(grid))}, geometry=grid)
)

point_inside = Point((4.5, 5.5))
point_outside = Point((1.5, 5.5))
inv_with_point_sources = Inventory.from_gdf(
    gpd.GeoDataFrame({"val": np.linspace(0, 1, len(grid))}, geometry=grid),
    gdfs={
        "pnt_sources": gpd.GeoDataFrame(
            {"val": [1, 2]}, geometry=[point_inside, point_outside]
        )
    },
)
# intersected_shapes , weigths = geoserie_intersection(
#     grid, poly, keep_outside=True, drop_unused=True
# )
#
# inter_df = gpd.GeoDataFrame({'weigths': weigths}, geometry=[s for s in intersected_shapes])
# inter_df.explore('weigths')

# TODO: write the tests properly


def test_basic_crop():
    intersected_shapes, weigths = geoserie_intersection(
        grid, poly, keep_outside=False, drop_unused=False
    )


def test_with_modify_grid():
    intersected_shapes, weigths = geoserie_intersection(
        grid, poly, keep_outside=False, drop_unused=False
    )


#%%


def test_crop_inventory():
    inv_cropped = crop_with_shape(inv, poly, keep_outside=False)


def test_crop_inventory_outside():
    inv_cropped = crop_with_shape(inv, poly, keep_outside=True)


def test_crop_inventory_inside_pnt_sources():
    inv_cropped = crop_with_shape(inv_with_point_sources, poly, keep_outside=True)
    assert len(inv_cropped.gdfs["pnt_sources"]) == 1
    assert inv_cropped.gdfs["pnt_sources"]["val"].iloc[0] == 2


def test_crop_inventory_outside_pnt_sources():
    inv_cropped = crop_with_shape(inv_with_point_sources, poly, keep_outside=False)
    assert len(inv_cropped.gdfs["pnt_sources"]) == 1
    assert inv_cropped.gdfs["pnt_sources"]["val"].iloc[0] == 1


def test_read_weights():
    w_file = ".emiproc.__test_weights__"

    inv_cropped = crop_with_shape(inv, poly, weight_file=w_file)
    # This time the weights should have been created
    inv_cropped = crop_with_shape(inv, poly, weight_file=w_file)


# %%
