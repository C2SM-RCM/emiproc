import pytest

import geopandas as gpd
import numpy as np
import xarray as xr

from emiproc import TESTS_DIR
from emiproc.exports.icon import TemporalProfilesTypes, export_icon_oem
from emiproc.grids import ICONGrid
from emiproc.inventories import Inventory
from emiproc.regrid import remap_inventory
from emiproc.tests_utils.icon import SIMPLE_ICON_GRID_PATH, get_test_grid, inv
from emiproc.tests_utils.temporal_profiles import (
    get_oem_const_hour_of_year_profile,
    oem_test_profile,
    oem_const_profile,
    HourOfLeapYearProfile,
)
from emiproc.tests_utils.test_inventories import inv_with_pnt_sources


def test_get_test_grid():
    """Test the function to get the test grid."""

    grid = get_test_grid()
    assert grid is not None


def test_utils_function():

    profile_type = get_oem_const_hour_of_year_profile(2020)
    assert isinstance(profile_type[0], HourOfLeapYearProfile)


def test_export_icon():
    """Test the export of ICON files."""

    grid = ICONGrid(SIMPLE_ICON_GRID_PATH)

    inv_on_icon = remap_inventory(inv, grid)

    export_icon_oem(
        inv_on_icon, SIMPLE_ICON_GRID_PATH, output_dir=TESTS_DIR / "export_icon"
    )


def test_export_icon_hour_of_year():
    """Test the export of ICON files."""

    grid = ICONGrid(SIMPLE_ICON_GRID_PATH)

    inv_on_icon = remap_inventory(inv, grid)

    kwargs = dict(
        icon_grid_file=SIMPLE_ICON_GRID_PATH,
        output_dir=TESTS_DIR / "export_icon",
        temporal_profiles_type=TemporalProfilesTypes.HOUR_OF_YEAR,
    )

    with pytest.raises(ValueError):
        # No year given
        export_icon_oem(inv_on_icon, **kwargs)

    inv_on_icon.year = 2021  # Set the year in the inventory
    export_icon_oem(inv_on_icon, **kwargs)

    # Leap year
    inv_on_icon.year = 2020  # Set the year in the inventory
    export_icon_oem(inv_on_icon, **kwargs)


def test_export_icon_with_profiles():
    """Test the export of ICON files with profiles."""

    grid = ICONGrid(SIMPLE_ICON_GRID_PATH)

    inv_on_icon = remap_inventory(inv, grid)

    inv_on_icon.set_profiles(
        [oem_test_profile, oem_const_profile],
        indexes=xr.DataArray(
            np.random.choice([0, 1, -1], size=[1, grid.ncell]),
            dims=["substance", "cell"],
            coords={"substance": ["CO2"], "cell": np.arange(grid.ncell)},
        ),
    )

    kwargs = dict(
        icon_grid_file=SIMPLE_ICON_GRID_PATH,
        output_dir=TESTS_DIR / "export_icon",
        temporal_profiles_type=TemporalProfilesTypes.THREE_CYCLES,
    )

    export_icon_oem(inv_on_icon, **kwargs)


def test_export_wrong_grid():
    with pytest.raises(ValueError):
        export_icon_oem(
            inv_with_pnt_sources,
            SIMPLE_ICON_GRID_PATH,
            output_dir=TESTS_DIR / "export_icon",
        )


if __name__ == "__main__":
    pytest.main(["-x", __file__])
