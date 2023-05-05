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
        p: from_csv(copernicus_profiles_dir / f"timeprofiles-{p}.csv") for p in TEST_COPENICUS_PROFILES
    }


three_profiles = [
    [
        WeeklyProfile(ratios=[0.1, 0.2, 0.3, 0.1, 0.15, 0.05, 0.1]),
        MounthsProfile(ratios=[0.25, 0.02, 0.03, 0.01, 0.015, 0.005, 0.11, 0.01, 0, 0.3, 0.1, 0.15]),
    ], 
    [
        WeeklyProfile(ratios=[0.1, 0.2, 0.3, 0.1, 0.15, 0.05, 0.1]),
        MounthsProfile(ratios=[0.01, 0.02, 0.03, 0.01, 0.015, 0.005, 0.11, 0.01, 0.24, 0.3, 0.1, 0.15]),
    ], 
    [
        WeeklyProfile(ratios=[0.3, 0.2, 0.3, 0.1, 0.05, 0.05, 0.]),
        MounthsProfile(ratios=[0.01, 0.02, 0.03, 0.01, 0.015, 0.005, 0.11, 0.01, 0.24, 0.3, 0.1, 0.15]),
    ]
]
