import pytest
from emiproc.tests_utils.profiles import (
    da_profiles_indexes_catsub,
    da_profiles_indexes_sub,
)

from emiproc.profiles.utils import get_desired_profile_index


def test_get_desired_profile_index_working_cases():
    assert (
        get_desired_profile_index(da_profiles_indexes_catsub, cat="b", sub="CO2") == 2
    )
    assert (
        get_desired_profile_index(da_profiles_indexes_catsub, cat="a", sub="CH4") == 1
    )
    # Test when you add too specific it still works
    assert get_desired_profile_index(da_profiles_indexes_sub, cat="a", sub="CH4") == 1
    assert (
        get_desired_profile_index(
            da_profiles_indexes_catsub, cell=43, cat="a", sub="CH4"
        )
        == 1
    )


def test_get_desired_profile_index_errors():
    # Not specified enough
    pytest.raises(
        ValueError, get_desired_profile_index, da_profiles_indexes_catsub, cat="a"
    )

def test_get_desired_profile_index_wrong_catsub():
    # Category does not exist
    pytest.raises(
        ValueError,
        get_desired_profile_index,
        da_profiles_indexes_catsub,
        cat="CH4",
        sub="CH4",
    )


if __name__ == "__main__":
    pytest.main([__file__])
