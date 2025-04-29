from emiproc.exports.fluxy import export_fluxy
from emiproc import TESTS_DIR
from emiproc.tests_utils.test_inventories import inv
from emiproc.tests_utils.test_grids import regular_grid
from emiproc.tests_utils.temporal_profiles import three_profiles
from emiproc.regrid import remap_inventory
import pytest


def test_export_fluxy_fails_on_non_regular_grid():

    with pytest.raises(AssertionError):
        export_fluxy(
            invs=inv,
            output_dir=TESTS_DIR / "fluxy",
        )


def test_export_fluxy_no_profiles_raises():
    inventory = inv.copy()
    inventory.set_crs(regular_grid.crs)
    with pytest.raises(ValueError):
        export_fluxy(
            invs=remap_inventory(inventory, grid=regular_grid),
            output_dir=TESTS_DIR / "fluxy",
        )


def test_export_fluxy_no_year_raises():
    inventory = inv.copy()
    inventory.set_crs(regular_grid.crs)

    inventory.set_profile(
        three_profiles[0],
        category="test",
    )
    inventory.set_profile(
        three_profiles[1],
        category="adf",
    )
    with pytest.raises(ValueError):
        export_fluxy(
            invs=remap_inventory(inventory, grid=regular_grid),
            output_dir=TESTS_DIR / "fluxy",
        )


def test_export_flux():

    inventory = inv.copy()
    inventory.set_crs(regular_grid.crs)
    inventory.year = 2020

    inventory.set_profile(
        three_profiles[0],
        category="test",
    )
    inventory.set_profile(
        three_profiles[1],
        category="adf",
    )

    export_fluxy(
        invs=remap_inventory(inventory, grid=regular_grid),
        output_dir=TESTS_DIR / "fluxy",
    )
