import emiproc
from emiproc.profiles.temporal_profiles import (
    AnyTimeProfile,
    DailyProfile,
    TemporalProfile,
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
