from emiproc.profiles.temporal.composite import CompositeTemporalProfiles
from emiproc.profiles.temporal.operators import interpolate_profiles_hour_of_year
from emiproc.profiles.temporal.profiles import HourOfLeapYearProfile, HourOfYearProfile
from emiproc.tests_utils import temporal_profiles


def test_interpolate_day():

    out_profiles = interpolate_profiles_hour_of_year(
        CompositeTemporalProfiles([[temporal_profiles.daily_test_profile]]),
        year=2022,
        return_profiles=True,
    )

    assert isinstance(out_profiles, CompositeTemporalProfiles)
    assert len(out_profiles.types) == 1
    assert out_profiles.types[0] == HourOfYearProfile


def test_leap_year():

    out_profiles = interpolate_profiles_hour_of_year(
        CompositeTemporalProfiles([[temporal_profiles.daily_test_profile]]),
        year=2024,
        return_profiles=True,
    )

    assert isinstance(out_profiles, CompositeTemporalProfiles)
    assert len(out_profiles.types) == 1
    assert out_profiles.types[0] == HourOfLeapYearProfile


def test_from_day_of_year():
    out_profiles = interpolate_profiles_hour_of_year(
        CompositeTemporalProfiles([[temporal_profiles.day_of_year_test_profile]]),
        year=2022,
        return_profiles=True,
    )

    assert isinstance(out_profiles, CompositeTemporalProfiles)
    assert len(out_profiles.types) == 1
    assert out_profiles.types[0] == HourOfYearProfile


def test_from_day_of_year_after_leap():
    out_profiles = interpolate_profiles_hour_of_year(
        CompositeTemporalProfiles([[temporal_profiles.day_of_year_test_profile]]),
        year=2021,
        return_profiles=True,
    )

    assert isinstance(out_profiles, CompositeTemporalProfiles)
    assert len(out_profiles.types) == 1
    assert out_profiles.types[0] == HourOfYearProfile
