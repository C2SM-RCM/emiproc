import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

from emiproc import TESTS_DIR
from emiproc.inventories.gral import GralInventory
from emiproc.utils.constants import HOUR_PER_YR

GRAL_TEST_DIR = TESTS_DIR / "inventories" / "gral"


@pytest.fixture
def gral_inventory_dir():
    GRAL_TEST_DIR.mkdir(exist_ok=True, parents=True)
    return GRAL_TEST_DIR


def test_read_gral_inventory(gral_inventory_dir):

    inv = GralInventory(gral_inventory_dir, crs="LV95")


def test_read_gral_inventory_content(gral_inventory_dir):
    inv = GralInventory(gral_inventory_dir, crs="LV95")

    assert len(inv.gdfs["blek"]) == 1
    assert len(inv.gdfs["adf"]) == 3
    assert len(inv.gdfs["line_only"]) == 1
    assert all(
        geom.geom_type in {"Point", "LineString", "Polygon"}
        for gdf in inv.gdfs.values()
        for geom in gdf.geometry
    )

    point_values = inv.gdfs["blek"]["CO2"].values
    assert point_values.size == 1
    assert point_values[0] > 0

    # Check the units
    pd.testing.assert_series_equal(
        inv.total_emissions.loc["CO2", ["blek", "line_only", "adf"]],
        pd.Series(
            {
                "blek": (2.0) * HOUR_PER_YR,
                # Line is given per km
                "line_only": 5.0 * 2e-3 * HOUR_PER_YR,
                "adf": (4.0 + 5.0 * (2e-3) + 8.0) * HOUR_PER_YR,
            },
            name="CO2",
        ),
    )


def test_raise_bad_crs(gral_inventory_dir):

    with pytest.raises(ValueError, match="crs WGS84 is geographic."):
        inv = GralInventory(gral_inventory_dir, crs="WGS84")


if __name__ == "__main__":
    pytest.main(["-q", __file__])
