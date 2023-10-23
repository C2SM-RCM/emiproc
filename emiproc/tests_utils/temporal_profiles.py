import emiproc
from emiproc.profiles.temporal_profiles import (
    AnyTimeProfile,
    DailyProfile,
    TemporalProfile,
    WeeklyProfile,
    MounthsProfile,
    from_csv,
    read_temporal_profiles,
)


copernicus_profiles_dir = emiproc.FILES_DIR / "profiles" / "copernicus"

TEST_COPENICUS_PROFILES = ["hour_in_day", "day_in_week", "month_in_year"]


def read_test_copernicus() -> dict[str, list[AnyTimeProfile]]:
    """Read the test copernicus profiles."""

    return {
        p: from_csv(copernicus_profiles_dir / f"timeprofiles-{p}.csv")
        for p in TEST_COPENICUS_PROFILES
    }


three_profiles = [
    [
        WeeklyProfile(ratios=[0.1, 0.2, 0.3, 0.1, 0.15, 0.05, 0.1]),
        MounthsProfile(
            ratios=[0.25, 0.02, 0.03, 0.01, 0.015, 0.005, 0.11, 0.01, 0, 0.3, 0.1, 0.15]
        ),
    ],
    [
        WeeklyProfile(ratios=[0.1, 0.2, 0.3, 0.1, 0.15, 0.05, 0.1]),
        MounthsProfile(
            ratios=[
                0.01,
                0.02,
                0.03,
                0.01,
                0.015,
                0.005,
                0.11,
                0.01,
                0.24,
                0.3,
                0.1,
                0.15,
            ]
        ),
    ],
    [
        WeeklyProfile(ratios=[0.3, 0.2, 0.3, 0.1, 0.05, 0.05, 0.0]),
        MounthsProfile(
            ratios=[
                0.01,
                0.02,
                0.03,
                0.01,
                0.015,
                0.005,
                0.11,
                0.01,
                0.24,
                0.3,
                0.1,
                0.15,
            ]
        ),
    ],
]


oem_const_profile = [
    DailyProfile(),
    MounthsProfile(),
    WeeklyProfile(),
]
weekly_test_profile = WeeklyProfile(ratios=[0.11, 0.09, 0.10, 0.09, 0.14, 0.24, 0.23])
mounths_test_profile = MounthsProfile(
    ratios=[0.19, 0.17, 0.13, 0.06, 0.05, 0.00, 0.00, 0.00, 0.01, 0.04, 0.15, 0.20]
)
# 24 hours
daily_test_profile = DailyProfile(
    ratios=[
        0.02667,
        0.01667,
        0.00667,
        0.00667,
        0.01667,
        0.02467,
        0.02700,
        0.05000,
        0.06250,
        0.06458,
        0.06250,
        0.05417,
        0.04583,
        0.04375,
        0.04375,
        0.04167,
        0.04167,
        0.04167,
        0.05000,
        0.05833,
        0.06250,
        0.05624,
        0.05416,
        0.04166,
    ]
)
oem_test_profile = [
    weekly_test_profile,
    mounths_test_profile,
    daily_test_profile,
]
