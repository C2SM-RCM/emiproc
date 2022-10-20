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

def test_basic_crop():

    cropped = crop_with_shape(inv, triangle)

def test_with_modify_grid():

    cropped = crop_with_shape(inv, triangle, modify_grid=True)

    assert 4 not in cropped.gdf.index

def test_with_modify_grid_and_cached():
    w_file = Path('.emiproc_test_with_modify_grid_and_cached')
    cropped = crop_with_shape(inv, triangle, weight_file=w_file, modify_grid=True)

    assert 4 not in cropped.gdf.index
    cropped = crop_with_shape(inv, triangle, weight_file=w_file, modify_grid=True)

    assert 4 not in cropped.gdf.index