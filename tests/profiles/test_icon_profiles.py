"""Test that the icon profiles can be saved."""

from emiproc.exports.icon import make_icon_time_profiles, TemporalProfilesTypes
from emiproc.tests_utils.temporal_profiles import oem_const_profile, oem_test_profile
from emiproc.utilities import HOUR_PER_YR


def test_three_profiles():
    dss = make_icon_time_profiles(
        {"cat_const": oem_const_profile, "cat_test": oem_test_profile},
        time_zones=["Europe/Zurich", "UTC", "Asia/Tokyo", "America/Montreal"],
        profiles_type=TemporalProfilesTypes.THREE_CYCLES,
    )
    # Test we have the three data sets
    assert "hourofday" in dss
    assert "dayofweek" in dss
    assert "monthofyear" in dss


def test_hour_of_year():
    dss = make_icon_time_profiles(
        {"cat_const": oem_const_profile, "cat_test": oem_test_profile},
        time_zones=["Europe/Zurich", "UTC", "Asia/Tokyo", "America/Montreal"],
        profiles_type=TemporalProfilesTypes.HOUR_OF_YEAR,
        year=2022,
    )
    assert "hourofyear" in dss
