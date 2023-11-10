import pytest

import numpy as np
import xarray as xr
from emiproc.profiles.temporal_profiles import (
    AnyProfiles,
    TemporalProfile,
    DailyProfile,
    WeeklyProfile,
    MounthsProfile,
    CompositeTemporalProfiles,
    SpecificDayProfile,
    HourOfYearProfile,
    HourOfLeapYearProfile,
    make_composite_profiles,
)
from emiproc.profiles.utils import merge_indexes


@pytest.mark.parametrize(
    "profile_type",
    [
        TemporalProfile,
        DailyProfile,
        WeeklyProfile,
        MounthsProfile,
        HourOfYearProfile,
        HourOfLeapYearProfile,
    ],
)
def test_create_profiles(profile_type):
    profile = profile_type()
    pytest.approx(profile.ratios.sum(axis=1), 1)
    assert profile.ratios.shape == (1, profile.size)


def test_multiple_profiles():
    p = TemporalProfile(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
        ],
        size=4,
    )
    assert p.ratios.shape == (3, 4)
    assert p.n_profiles == 3
    assert p.size == 4
    assert len(p) == 3


def test_composite_temporal_profiles():
    p = CompositeTemporalProfiles(
        [
            [TemporalProfile(), DailyProfile()],
            [WeeklyProfile()],
        ]
    )
    assert p.n_profiles == 2
    assert len(p[0]) == 2
    assert len(p[1]) == 1

def test_equality():
    p1 = WeeklyProfile()
    p2 = WeeklyProfile()
    p3 = WeeklyProfile([0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0])
    p4 = WeeklyProfile([0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0])
    assert p1 == p2
    assert p1 != p3
    assert p3 == p4
    

def test_merging_profiles():
    profiles = AnyProfiles(
        [
            MounthsProfile(
                [
                    [0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.2, 0.4, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ),
            WeeklyProfile([0.1, 0.2, 0.3, 0.1, 0.0, 0.0, 0.3]),
        ]
    )
    assert profiles.n_profiles == 3
    dss = [
        xr.DataArray(
            np.array([[0, 1, 0], [1, -1, 0]]),
            coords={
                "category": ["blek", "liku"],
                "substance": ["CO2", "CH4", "N2O"],
            },
        ).expand_dims(
            {
                "profile": ["MounthsProfile"],
            }
        ),
        xr.DataArray(
            np.array([[2, -1], [2, 2]]),
            coords={
                "category": ["blek", "liku"],
                "substance": ["CO2", "CH4"],
            },
        ).expand_dims(
            {
                "profile": ["WeeklyProfile"],
            }
        ),
    ]

    combined_indexes = merge_indexes(dss)

    make_composite_profiles(profiles, combined_indexes)


if __name__ == "__main__":
    pytest.main([__file__])
