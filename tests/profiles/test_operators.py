import numpy as np
import pytest
import xarray as xr
import pandas as pd

from emiproc.profiles.operators import (
    combine_profiles,
    country_to_cells,
    get_weights_of_gdf_profiles,
    group_profiles_indexes,
    weighted_combination,
)
from emiproc.profiles.temporal.composite import CompositeTemporalProfiles
from emiproc.profiles.temporal.operators import get_index_in_profile
from emiproc.profiles.temporal.profiles import (
    DailyProfile,
    Hour3OfDayPerMonth,
    HourOfLeapYearProfile,
    HourOfYearProfile,
    SpecificDayProfile,
    WeeklyProfile,
)
from emiproc.profiles.temporal.specific_days import SpecificDay
from emiproc.tests_utils import temporal_profiles, vertical_profiles
from emiproc.tests_utils.temporal_profiles import (
    TEST_COPENICUS_PROFILES,
    get_random_profiles,
    indexes_african_2d,
    indexes_african_simple,
    read_test_copernicus,
)
from emiproc.tests_utils.test_grids import regular_grid_africa
from emiproc.tests_utils.vertical_profiles import (
    get_random_profiles as get_random_profiles_vertical,
)


def test_reading_copernicus():
    profiles = read_test_copernicus()

    assert len(profiles) == len(TEST_COPENICUS_PROFILES)


def test_weighted_combination():
    weighted_combination([DailyProfile(), DailyProfile()], [0.5, 0.5])


def test_weighted_combination_fails_on_different_profiles_types():
    pytest.raises(
        TypeError, weighted_combination, [DailyProfile(), WeeklyProfile()], [0.5, 0.5]
    )


def test_weighted_combination_fails_on_wrong_weights_lenghts():
    pytest.raises(
        ValueError, weighted_combination, [DailyProfile(), DailyProfile()], [0.5]
    )


def test_combine_profiles_single_dimension():
    new_profiles, new_indices = combine_profiles(
        profiles=vertical_profiles.get_random_profiles(2),
        profiles_indexes=vertical_profiles.single_dim_profile_indexes,
        weights=vertical_profiles.single_dim_weights,
        dimension="category",
    )


def test_combine_profiles():
    weights = get_weights_of_gdf_profiles(
        vertical_profiles.inv.gdf, vertical_profiles.corresponding_2d_profiles
    )
    new_profiles, new_indices = combine_profiles(
        profiles=temporal_profiles.three_profiles,
        profiles_indexes=vertical_profiles.corresponding_2d_profiles,
        weights=weights,
        dimension="category",
    )
    assert "category" not in new_indices.dims


def test_group_profiles_with_time_profiles():
    weights = get_weights_of_gdf_profiles(
        vertical_profiles.inv.gdf, vertical_profiles.corresponding_2d_profiles
    )
    new_profiles, new_indices = group_profiles_indexes(
        profiles=temporal_profiles.three_profiles,
        profiles_indexes=vertical_profiles.corresponding_2d_profiles,
        indexes_weights=weights,
        categories_group=vertical_profiles.inv_groups_dict,
        groupping_dimension="category",
    )
    assert "category" in new_indices.dims


def test_get_random_profiles():
    p = get_random_profiles(3)
    assert len(p) == 3


def test_get_random_profies_vertical():
    p = get_random_profiles_vertical(5)
    assert len(p) == 5


@pytest.mark.parametrize(
    "profiles, indexes",
    [
        (
            get_random_profiles_vertical(indexes_african_simple.max().values + 1),
            indexes_african_simple,
        ),
        (
            get_random_profiles(indexes_african_simple.max().values + 1),
            indexes_african_simple,
        ),
        (
            get_random_profiles_vertical(indexes_african_2d.max().values + 1),
            indexes_african_2d,
        ),
        (
            get_random_profiles(indexes_african_2d.max().values + 1),
            indexes_african_2d,
        ),
    ],
)
def test_countries_to_cells(profiles, indexes: xr.DataArray):
    grid = regular_grid_africa

    if isinstance(profiles, list):
        profiles = CompositeTemporalProfiles(profiles)

    new_profiles, new_indexes = country_to_cells(profiles, indexes, grid)

    assert "cell" in new_indexes.dims
    assert "country" not in new_indexes.dims

    if isinstance(profiles, CompositeTemporalProfiles):
        assert profiles.types == new_profiles.types
    # test for some cells that we now what it should be
    # Full in MRT, profiles should be the same as the original
    np.testing.assert_almost_equal(
        new_profiles.scaling_factors[new_indexes.sel(cell=78).drop_vars("cell")],
        profiles.scaling_factors[indexes.sel(country="MRT").drop_vars("country")],
    )
    # THis is shared between SEN and ocean, so only in SEN for the profile
    np.testing.assert_almost_equal(
        new_profiles.scaling_factors[new_indexes.sel(cell=26).drop_vars("cell")],
        profiles.scaling_factors[indexes.sel(country="SEN").drop_vars("country")],
    )

    # This is just ocean
    assert np.all(new_indexes.sel(cell=0).drop_vars("cell") == -1)


test_data_index_in_profles = pd.DatetimeIndex(
    [
        # With hours
        pd.Timestamp("2020-01-01 00:00:00"),  # Wednesday
        pd.Timestamp("2020-01-02 17:00:00"),  # Thursday
        pd.Timestamp("2020-05-25 12:00:00"),  # Monday
    ]
)


@pytest.mark.parametrize(
    "profile_type, expected",
    [
        (DailyProfile, [0, 17, 12]),
        (WeeklyProfile, [2, 3, 0]),
        ((SpecificDayProfile, SpecificDay.MONDAY), [-1, -1, 12]),
        ((SpecificDayProfile, SpecificDay.WEEKDAY), [0, 17, 12]),
        ((SpecificDayProfile, SpecificDay.WEEKEND), [-1, -1, -1]),
        (HourOfLeapYearProfile, [0, 41, 3492]),
        (Hour3OfDayPerMonth, [0, 5, 36]),
    ],
)
def test_index_in_profile(profile_type, expected):

    indices = get_index_in_profile(profile_type, test_data_index_in_profles)
    pd.testing.assert_index_equal(indices, pd.Index(expected, dtype=indices.dtype))


if __name__ == "__main__":
    pytest.main([__file__])
    # test_group_profiles_with_time_profiles()
