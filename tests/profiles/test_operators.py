import pytest
from emiproc.profiles.temporal_profiles import DailyProfile, WeeklyProfile
from emiproc.tests_utils.temporal_profiles import (
    read_test_copernicus,
    TEST_COPENICUS_PROFILES,
)
from emiproc.profiles.operators import get_weights_of_gdf_profiles, group_profiles_indexes, weighted_combination


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


def test_group_profiles_with_time_profiles():
    from emiproc.tests_utils import vertical_profiles
    from emiproc.tests_utils import temporal_profiles


    weights = get_weights_of_gdf_profiles(vertical_profiles.inv.gdf, vertical_profiles.corresponding_2d_profiles)
    new_profiles,new_indices = group_profiles_indexes(
        profiles = temporal_profiles.three_profiles,
        profiles_indexes = vertical_profiles.corresponding_2d_profiles,
        indexes_weights=weights,
        categories_group=vertical_profiles.inv_groups_dict,
        groupping_dimension="category"
    )
    
    

if __name__ == "__main__":
    pytest.main([__file__])
    #test_group_profiles_with_time_profiles()
