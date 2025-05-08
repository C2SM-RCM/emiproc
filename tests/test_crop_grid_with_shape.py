"""Test case of a grid that intersects a shape."""
import shutil
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from emiproc.inventories.utils import crop_with_shape
from emiproc.regrid import (
    geoserie_intersection,
)
from emiproc.inventories import Inventory
from emiproc.tests_utils import WEIGHTS_DIR


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
non_convex_poly = Polygon(
    (
        (5.5, 7.5),
        (5.5, 3.5),
        (2.5, 3.5),
        (2.5, 6.5),
        (4.5, 6.5),
        (4.5, 7.5),
        (2.5, 7.5),
        (2.5, 8.5),
    )
)

inv = Inventory.from_gdf(
    gpd.GeoDataFrame({("main", "val"): np.linspace(0, 1, len(grid))}, geometry=grid)
)

point_inside = Point((4.5, 5.5))
point_outside = Point((1.5, 5.5))
inv_with_point_sources = Inventory.from_gdf(
    gpd.GeoDataFrame({("main", "val"): np.linspace(0, 1, len(grid))}, geometry=grid),
    gdfs={
        "pnt_sources": gpd.GeoDataFrame(
            {"val": [1, 2]}, geometry=[point_inside, point_outside]
        )
    },
)


def test_basic_crop():
    intersected_shapes, weigths = geoserie_intersection(
        grid, poly, keep_outside=False, drop_unused=False
    )


def test_with_modify_grid():
    intersected_shapes, weigths = geoserie_intersection(
        grid, poly, keep_outside=False, drop_unused=False
    )


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
    w_file = WEIGHTS_DIR / ".emiproc__test_read_weights"
    if w_file.with_suffix(".gdb").is_dir():
        shutil.rmtree(w_file.with_suffix(".gdb"))
    w_file.with_suffix(".npy").unlink(missing_ok=True)

    inv_cropped_1 = crop_with_shape(inv, poly, weight_file=w_file)
    # This time the weights should have been created
    inv_cropped_2 = crop_with_shape(inv, poly, weight_file=w_file)
    assert all(inv_cropped_1.gdf.index == inv_cropped_2.gdf.index)
    # Check that the polygons are the same
    assert inv_cropped_1.gdf.geometry.equals(inv_cropped_2.gdf.geometry)
    # Check that there is not none geomtery
    assert all([t is not None for t in inv_cropped_1.gdf.geometry])


def test_read_weights_modified_grid():
    w_file = WEIGHTS_DIR / ".emiproc_test_read_weights_modified_grid"
    if w_file.with_suffix(".gdb").is_dir():
        shutil.rmtree(w_file.with_suffix(".gdb"))
    w_file.with_suffix(".npy").unlink(missing_ok=True)
    inv_cropped_1 = crop_with_shape(inv, poly, weight_file=w_file, modify_grid=True)
    # This time the weights should have been created
    inv_cropped_2 = crop_with_shape(inv, poly, weight_file=w_file, modify_grid=True)

    assert all(inv_cropped_1.gdf.index == inv_cropped_2.gdf.index)
    # Check that the polygons are the same
    assert inv_cropped_1.gdf.geometry.equals(inv_cropped_2.gdf.geometry)
    # Check that there is not none geomtery
    assert all([t is not None for t in inv_cropped_1.gdf.geometry])


def test_non_convex_poly():
    intersected_shapes, weigths = geoserie_intersection(
        grid, non_convex_poly, keep_outside=True, drop_unused=True
    )

    cropped_inv = crop_with_shape(
        inv_with_point_sources, non_convex_poly, keep_outside=True, modify_grid=True
    )

    cropped_inv.gdf.geom_type
