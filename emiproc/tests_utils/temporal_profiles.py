import random
from dataclasses import dataclass, field
from typing import Type

import numpy as np
import xarray as xr

import emiproc
from emiproc.profiles.temporal.composite import CompositeTemporalProfiles
from emiproc.profiles.temporal.io import from_csv
from emiproc.profiles.temporal.profiles import (
    AnyTimeProfile,
    DailyProfile,
    DayOfLeapYearProfile,
    DayOfYearProfile,
    MounthsProfile,
    TemporalProfile,
    WeeklyProfile,
)

copernicus_profiles_dir = emiproc.FILES_DIR / "profiles" / "copernicus"

TEST_COPENICUS_PROFILES = ["hour_in_day", "day_in_week", "month_in_year"]


def read_test_copernicus() -> dict[str, list[AnyTimeProfile]]:
    """Read the test copernicus profiles."""

    return {
        p: from_csv(copernicus_profiles_dir / f"timeprofiles-{p}.csv")
        for p in TEST_COPENICUS_PROFILES
    }


def get_random_profiles(
    num: int,
    profile_types: list[Type[TemporalProfile]] = [
        DailyProfile,
        WeeklyProfile,
        MounthsProfile,
    ],
) -> list[list[AnyTimeProfile]]:
    """Get random profiles for testing."""
    return [
        [
            type(ratios=((ratios := np.random.rand(type.size)) / ratios.sum()))
            for type in profile_types
            if random.choice([True, False])
        ]
        for _ in range(num)
    ]


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
three_composite_profiles = CompositeTemporalProfiles(three_profiles)


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
day_of_year_ratios = np.arange(365, dtype=float)
day_of_year_test_profile = DayOfYearProfile(
    day_of_year_ratios / day_of_year_ratios.sum()
)
oem_test_profile = [
    weekly_test_profile,
    mounths_test_profile,
    daily_test_profile,
]

# For the inventories in test_inventories.py

indexes_inv_catsub = xr.DataArray(
    data=np.array(
        [
            [0, 1, 2],
            [3, 2, -1],
        ]
    ),
    dims=["category", "substance"],
    coords={
        "category": ["adf", "liku"],  # omit one category on purpose
        "substance": ["CH4", "CO2", "NH3"],
    },
)
indexes_inv_catsub_missing = xr.DataArray(
    data=np.array(
        [
            [0, 1],
        ]
    ),
    dims=["category", "substance"],
    # omit many on purpose
    coords={
        "category": ["adf"],
        "substance": ["CH4", "CO2"],
    },
)
indexes_inv_catsubcell = xr.DataArray(
    data=np.array(
        [
            [[1, 1, 1, 1, 1], [-1, -1, -1, 0, 1], [-1, -1, -1, 0, 2]],
            [[0, 0, 1, 2, -1], [2, 2, 2, 2, 2], [-1, -1, 0, 2, 1]],
        ]
    ),
    dims=["category", "substance", "cell"],
    coords={
        "category": ["adf", "liku"],  # omit one category on purpose
        "substance": ["CH4", "CO2", "NH3"],
        "cell": np.arange(5),
    },
)

# For the african test set
african_countries_test_set = ["SEN", "MLI", "MRT", "GIN", "GNB", "LBR", "SLE", "GMB"]
indexes_african_simple = xr.DataArray(
    data=np.arange(len(african_countries_test_set)),
    dims=["country"],
    coords={"country": african_countries_test_set},
)
indexes_african_2d = xr.DataArray(
    data=np.arange(len(african_countries_test_set) * 3).reshape(
        (len(african_countries_test_set), 3)
    ),
    dims=["country", "category"],
    coords={
        "country": african_countries_test_set,
        "category": ["liku", "blek", "test"],
    },
)


@dataclass(eq=False)
class Profile2(TemporalProfile):
    """Test profile of size 2."""

    size: int = field(default=2, init=False)


@dataclass(eq=False)
class Profile3(TemporalProfile):
    """Test profile of size 3."""

    size: int = field(default=3, init=False)
