import pytest

import geopandas as gpd

from emiproc import TESTS_DIR
from emiproc.exports.icon import export_icon_oem
from emiproc.grids import ICONGrid
from emiproc.inventories import Inventory
from emiproc.regrid import remap_inventory
from emiproc.tests_utils.icon import SIMPLE_ICON_GRID_PATH, get_test_grid
from emiproc.tests_utils.test_inventories import inv_with_pnt_sources


def test_get_test_grid():
    """Test the function to get the test grid."""

    grid = get_test_grid()
    assert grid is not None


def test_export_icon():
    """Test the export of ICON files."""

    grid = ICONGrid(SIMPLE_ICON_GRID_PATH)

    inv = Inventory.from_gdf(
        gdfs={
            "point": gpd.GeoDataFrame(
                data={
                    "CO2": 3 * [1],
                },
                geometry=gpd.points_from_xy(
                    y=[38.695852, 48.695852, 59.695852],
                    x=[69.089789, 79.089789, 89.089789],
                ),
                crs="WGS84",
            ),
        }
    )

    inv_on_icon = remap_inventory(inv, grid)

    export_icon_oem(
        inv_on_icon, SIMPLE_ICON_GRID_PATH, output_dir=TESTS_DIR / "export_icon"
    )


def test_export_wrong_grid():
    with pytest.raises(ValueError):
        export_icon_oem(
            inv_with_pnt_sources,
            SIMPLE_ICON_GRID_PATH,
            output_dir=TESTS_DIR / "export_icon",
        )


if __name__ == "__main__":
    pytest.main(["-x", __file__])
