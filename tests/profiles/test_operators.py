import pytest
from emiproc.profiles.temporal_profiles import DailyProfile, WeeklyProfile
from emiproc.tests_utils.temporal_profiles import (
    read_test_copernicus,
    TEST_COPENICUS_PROFILES,
)
from emiproc.profiles.operators import weighted_combination


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


if __name__ == "__main__":
    pytest.main([__file__])
