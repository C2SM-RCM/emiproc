import pytest

import geopandas as gpd
import numpy as np
import xarray as xr

from emiproc import TESTS_DIR
from emiproc.exports.icon import (
    TemporalProfilesTypes,
    export_icon_oem,
    make_icon_time_profiles,
)
from emiproc.grids import ICONGrid
from emiproc.inventories import Inventory
from emiproc.regrid import remap_inventory
from emiproc.profiles.temporal.composite import CompositeTemporalProfiles
from emiproc.profiles.temporal.profiles import (
    DailyProfile,
    WeeklyProfile,
    MounthsProfile,
)
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


# A single region with no tz shift, used when the shift itself is not under test.
no_shift_regions = xr.DataArray(
    ["r0", "r1"],
    dims=["region"],
    coords={
        "temporal_profile_id": ("region", [0, 0]),
        "tz_region": ("region", ["UTC", "UTC"]),
        "tz_shift": ("region", [0, 0]),
    },
)


def _single_profile_indexes() -> xr.DataArray:
    # Both category/substance combinations point to the single profile group (index 0).
    return xr.DataArray(
        np.zeros((1, 1), dtype=int),
        dims=["category", "substance"],
        coords={"category": ["point"], "substance": ["CO2"]},
    ).expand_dims({"region": np.array([0], dtype=int)})


def test_make_icon_time_profiles_no_profiles_given():
    """When no profiles are set, the scaling factors should be constant (equal to 1)."""

    catsubs = {("point", "CO2"): "point-CO2"}
    time_profiles = CompositeTemporalProfiles(
        [[DailyProfile(), WeeklyProfile(), MounthsProfile()]]
    )

    dict_ds = make_icon_time_profiles(
        catsubs=catsubs,
        time_profiles=time_profiles,
        inv=inv,
        profiles_indexes=_single_profile_indexes(),
        regions=no_shift_regions,
        profiles_type=TemporalProfilesTypes.THREE_CYCLES,
        out_dir=None,
    )

    assert set(dict_ds.keys()) == {"hourofday", "dayofweek", "monthofyear"}
    for ds in dict_ds.values():
        np.testing.assert_array_equal(ds["point-CO2"].values, 1.0)


def test_make_icon_time_profiles_tz_shift():
    """The hour of day profile should be rolled by -tz_shift, other cycles untouched."""

    catsubs = {("point", "CO2"): "point-CO2"}

    daily_ratios = np.zeros(24)
    daily_ratios[0] = 1.0
    time_profiles = CompositeTemporalProfiles(
        [[DailyProfile(ratios=daily_ratios), WeeklyProfile(), MounthsProfile()]]
    )

    tz_shift = 2
    regions = xr.DataArray(
        ["r0", "r1"],
        dims=["region"],
        coords={
            "temporal_profile_id": ("region", [0, 0]),
            "tz_region": ("region", ["UTC", "Etc/GMT+2"]),
            "tz_shift": ("region", [0, tz_shift]),
        },
    )

    dict_ds = make_icon_time_profiles(
        catsubs=catsubs,
        time_profiles=time_profiles,
        inv=inv,
        profiles_indexes=_single_profile_indexes(),
        regions=regions,
        profiles_type=TemporalProfilesTypes.THREE_CYCLES,
        out_dir=None,
        correct_tz_shift=True,
    )

    hourofday = dict_ds["hourofday"]["point-CO2"].values
    expected = np.zeros((24, 2))
    expected[0, 0] = 24.0
    expected[(0 - tz_shift) % 24, 1] = 24.0
    np.testing.assert_array_equal(hourofday, expected)

    # Cycles other than hour of day are not shifted, so both countries are identical.
    np.testing.assert_array_equal(
        dict_ds["dayofweek"]["point-CO2"].values[:, 0],
        dict_ds["dayofweek"]["point-CO2"].values[:, 1],
    )
    np.testing.assert_array_equal(
        dict_ds["monthofyear"]["point-CO2"].values[:, 0],
        dict_ds["monthofyear"]["point-CO2"].values[:, 1],
    )


if __name__ == "__main__":
    pytest.main(["-x", __file__])
