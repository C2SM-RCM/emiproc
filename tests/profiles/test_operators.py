import pytest

import xarray as xr
from emiproc.profiles.temporal_profiles import (
    CompositeTemporalProfiles,
    DailyProfile,
    WeeklyProfile,
)
from emiproc.tests_utils.temporal_profiles import (
    read_test_copernicus,
    TEST_COPENICUS_PROFILES,
    get_random_profiles,
    indexes_african_simple,
    indexes_african_2d,
)
from emiproc.profiles.operators import (
    get_weights_of_gdf_profiles,
    group_profiles_indexes,
    weighted_combination,
    country_to_cells,
    combine_profiles,
)
from emiproc.tests_utils.test_grids import regular_grid_africa
from emiproc.tests_utils.vertical_profiles import (
    get_random_profiles as get_random_profiles_vertical,
)
from emiproc.tests_utils import vertical_profiles
from emiproc.tests_utils import temporal_profiles


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

    print(profiles, indexes)
    new_profiles, new_indexes = country_to_cells(profiles, indexes, grid)

    assert "cell" in new_indexes.dims
    assert "country" not in new_indexes.dims

    # test for some cells that we now what it should be
    # Full in MRT, profiles should be the same as the original
    xr.testing.assert_equal(
        new_indexes.sel(cell=78).drop_vars("cell"),
        indexes.sel(country="MRT").drop_vars("country"),
    )
    # THis is shared between SEN and ocean, so only in SEN for the profile
    xr.testing.assert_equal(
        new_indexes.sel(cell=26).drop_vars("cell"),
        indexes.sel(country="SEN").drop_vars("country"),
    )
    # assert da.sel(cell=26, country="SEN").values > 0.01
    # assert da.sel(cell=26, country="SEN").values < 0.5
    # assert total_fractions.sel(cell=26).values < 0.5
    # This is just ocean
    assert 0 not in new_indexes.coords["cell"].values
    print("new profiles", len(new_profiles), "indexes", new_indexes)
    assert len(new_profiles) > new_indexes.max().values


if __name__ == "__main__":
    pytest.main([__file__])
    # test_group_profiles_with_time_profiles()
