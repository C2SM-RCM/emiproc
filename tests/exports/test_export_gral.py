import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

from emiproc.exports.gral import export_to_gral
from emiproc.inventories import EmissionInfo, Inventory


@pytest.mark.skipif(
    pytest.importorskip("pygg", reason="pygg is required for GRAL export tests")
    is None,
    reason="pygg is required for GRAL export tests",
)
def test_export_gral_inventory(tmp_path):
    from pygg.grids import GralGrid

    inv = Inventory.from_gdf(
        gdfs={
            "adf": gpd.GeoDataFrame(
                {
                    "CO2": [2.0, 3.0, 4.0],
                    "CH4": [1.0, 2.0, 3.0],
                },
                geometry=[
                    Point(0.5, 0.5),
                    LineString([(1.0, 1.0), (3.0, 1.0)]),
                    Polygon([(6.0, 6.0), (6.0, 7.0), (7.0, 7.0), (7.0, 6.0)]),
                ],
            ),
        }
    )
    inv.emission_infos = {cat: EmissionInfo() for cat in inv.categories}

    grid = GralGrid(
        nx=10,
        ny=10,
        xmin=0,
        xmax=10,
        ymin=0,
        ymax=10,
        dz0=1.0,
        ddz=1.0,
        crs=inv.crs,
    )
    grid.building_heights = np.zeros((grid.ny, grid.nx), dtype=float)

    out_dir = tmp_path / "gral_output"
    out_dir.mkdir()

    export_to_gral(inv, grid, out_dir, polygon_raster_size=1.0)

    assert (out_dir / "point.dat").exists()
    assert (out_dir / "line.dat").exists()
    assert (out_dir / "cadastre.dat").exists()
    assert (out_dir / "source_groups.json").exists()

    # Read each file and check that the number of lines matches the number of
    # sources in the inventory (one row per substance).
    df_point = pd.read_csv(out_dir / "point.dat", header=1)
    df_line = pd.read_csv(out_dir / "line.dat", header=4)
    df_polygon = pd.read_csv(out_dir / "cadastre.dat", header=0)

    for df in [df_point, df_line, df_polygon]:
        print(df)
        assert len(df) == 2

    source_groups = json.load(open(out_dir / "source_groups.json"))

    sg_of_sub = {tup[0]: int(sg) for sg, tup in source_groups.items()}
    sg_index = [sg_of_sub["CO2"], sg_of_sub["CH4"]]

    pd.testing.assert_series_equal(
        df_point.set_index("source_group").loc[sg_index, "emission[kg/h]"],
        pd.Series([2.0, 1.0], name="emission[kg/h]"),
        check_index=False,
    )
    pd.testing.assert_series_equal(
        # Convert back to shape emissions from /km
        df_line.set_index("source_group").loc[sg_index, "emission_rate[kg/h/km]"]
        * 2e-3,
        # Total emissions per shape
        pd.Series([3.0, 2.0], name="emission_rate[kg/h/km]"),
        check_index=False,
    )

    # Note this is a special case where we have only one polygon, if this part 
    # breaks, it might be because of another issue.
    pd.testing.assert_series_equal(
        df_polygon.set_index("source_group").loc[sg_index, "emission_rate[kg/h]"],
        pd.Series([4.0, 3.0], name="emission_rate[kg/h]"),
        check_index=False,
    )


@pytest.mark.skipif(
    pytest.importorskip("pygg", reason="pygg is required for GRAL export tests")
    is None,
    reason="pygg is required for GRAL export tests",
)
def test_export_gral_polygons(tmp_path):
    from pygg.grids import GralGrid

    inv = Inventory.from_gdf(
        gdfs={
            "adf": gpd.GeoDataFrame(
                {
                    "CO2": [2.0],
                    "CH4": [1.0],
                },
                geometry=[
                    Polygon([(6.5, 6.5), (6.5, 8.0), (8.0, 8.0), (8.0, 6.5)]),
                ],
            ),
        }
    )
    inv.emission_infos = {cat: EmissionInfo() for cat in inv.categories}

    grid = GralGrid(
        nx=10,
        ny=10,
        xmin=0,
        xmax=10,
        ymin=0,
        ymax=10,
        dz0=1.0,
        ddz=1.0,
        crs=inv.crs,
    )
    grid.building_heights = np.zeros((grid.ny, grid.nx), dtype=float)

    out_dir = tmp_path / "gral_output_polygon"
    out_dir.mkdir()

    export_to_gral(inv, grid, out_dir, polygon_raster_size=1.0)

    assert (out_dir / "cadastre.dat").exists()
    assert (out_dir / "source_groups.json").exists()

    # Read each file and check that the number of lines matches the number of
    # sources in the inventory (one row per substance).
    df_polygon = pd.read_csv(out_dir / "cadastre.dat", header=0)

    assert len(df_polygon) == 8

    print(df_polygon)

    source_groups = json.load(open(out_dir / "source_groups.json"))

    sg_of_sub = {tup[0]: int(sg) for sg, tup in source_groups.items()}
    sg_index = [sg_of_sub["CO2"], sg_of_sub["CH4"]]

    # Note this is a special case where we have only one polygon, if this part 
    # breaks, it might be because of another issue.
    pd.testing.assert_series_equal(
        df_polygon.groupby("source_group").sum().loc[sg_index, "emission_rate[kg/h]"],
        pd.Series([2.0, 1.0], name="emission_rate[kg/h]"),
        check_index=False,
    )
