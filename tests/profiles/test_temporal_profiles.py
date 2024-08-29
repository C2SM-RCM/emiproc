import numpy as np
import pytest
import xarray as xr

from emiproc.profiles.operators import concatenate_profiles, weighted_combination
from emiproc.profiles.temporal_profiles import (
    AnyProfiles,
    CompositeTemporalProfiles,
    DailyProfile,
    HourOfLeapYearProfile,
    HourOfYearProfile,
    MounthsProfile,
    SpecificDay,
    SpecificDayProfile,
    TemporalProfile,
    WeeklyProfile,
    make_composite_profiles,
)
from emiproc.profiles.utils import merge_indexes
from emiproc.tests_utils.temporal_profiles import (
    daily_test_profile,
    mounths_test_profile,
    weekly_test_profile,
)


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


@pytest.mark.parametrize(
    "profiles_list_list",
    [
        (
            [
                [TemporalProfile(), DailyProfile()],
                [WeeklyProfile()],
            ]
        ),
        (
            [
                [],
                [WeeklyProfile()],
            ]
        ),
        (
            [
                [],
                [],
            ]
        ),
        (
            [
                [],
            ]
        ),
        (
            [
                [TemporalProfile(), DailyProfile()],
                [
                    WeeklyProfile(),
                    SpecificDayProfile(
                        np.array(23 * [0.01] + [0.77]), specific_day=SpecificDay.SUNDAY
                    ),
                ],
            ]
        ),
    ],
)
def test_composite_temporal_profiles(profiles_list_list):
    p = CompositeTemporalProfiles(profiles_list_list)
    assert p.n_profiles == len(profiles_list_list)
    for key, expected in enumerate(profiles_list_list):
        assert len(p[key]) == len(expected)
        for profile, expected_profile in zip(sorted(p[key]), sorted(expected)):
            assert profile == expected_profile

    # Test append
    append_profile = DailyProfile()
    p.append([append_profile])
    assert p.n_profiles == len(profiles_list_list) + 1
    assert len(p[-1]) == 1
    assert p[-1][0] == append_profile
    # Check that the first profile has not changed (assume okay if the others are)
    assert len(p[0]) == len(profiles_list_list[0])
    for profile, expected_profile in zip(sorted(p[0]), sorted(profiles_list_list[0])):
        assert profile == expected_profile

    # Now append specific days
    append_profiles = [
        SpecificDayProfile(
            np.array(23 * [0.02] + [0.54]), specific_day=SpecificDay.SATURDAY
        ),
        SpecificDayProfile(
            np.array(23 * [0.01] + [0.77]), specific_day=SpecificDay.SUNDAY
        ),
    ]
    p.append(append_profiles)
    assert p.n_profiles == len(profiles_list_list) + 2
    assert len(p[-1]) == 2
    for profile, expected_profile in zip(sorted(p[-1]), sorted(append_profiles)):
        assert profile == expected_profile
    # Check that the first profile has not changed (assume okay if the others are)
    assert len(p[0]) == len(profiles_list_list[0])
    for profile, expected_profile in zip(sorted(p[0]), sorted(profiles_list_list[0])):
        assert profile == expected_profile


def test_composite_copy():
    p = CompositeTemporalProfiles(
        [
            [TemporalProfile(), DailyProfile()],
            [],
            [WeeklyProfile()],
        ]
    )
    p2 = p.copy()
    assert p2.n_profiles == p.n_profiles
    for key, expected in enumerate(p):
        assert len(p2[key]) == len(expected)
        for profile, expected_profile in zip(sorted(p2[key]), sorted(expected)):
            assert profile == expected_profile


def test_composite_error_wrong_type():
    with pytest.raises(TypeError):
        CompositeTemporalProfiles(
            [
                [TemporalProfile(), DailyProfile()],
                [WeeklyProfile()],
                [1],
            ]
        )


def test_iterate_over_composite():
    profiles = CompositeTemporalProfiles(
        [
            [TemporalProfile(), DailyProfile()],
            [WeeklyProfile()],
        ]
    )
    for profile in profiles:
        assert isinstance(profile, list)
        for p in profile:
            assert isinstance(p, TemporalProfile)


def test_composite_ratios():
    profiles = CompositeTemporalProfiles(
        [
            [WeeklyProfile(), DailyProfile()],
            [WeeklyProfile()],
        ]
    )
    assert profiles.ratios.shape == (2, 7 + 24)

    # test we can create back from the ratios
    new_profiles = CompositeTemporalProfiles.from_ratios(
        profiles.ratios, profiles.types
    )

    for old, new in zip(profiles, new_profiles):
        assert len(old) == len(new)
        for o, n in zip(sorted(old), sorted(new)):
            assert isinstance(o, TemporalProfile)
            assert isinstance(n, TemporalProfile)
            assert o == n


@pytest.mark.parametrize(
    "ratios, types, expected",
    [
        (
            np.concatenate(
                [
                    (np.ones((2, 7)) / 7.0),
                    (np.ones((2, 24)) / 24.0),
                ],
                axis=1,
            ),
            [WeeklyProfile, DailyProfile],
            {
                0: [WeeklyProfile(), DailyProfile()],
                1: [WeeklyProfile(), DailyProfile()],
            },
        ),
        (
            np.full((2, 7 + 24), np.nan),
            [WeeklyProfile, DailyProfile],
            {0: [], 1: []},
        ),
        (
            np.full((1, 7 + 24), np.nan),
            [WeeklyProfile, DailyProfile],
            {0: []},
        ),
    ],
)
def test_composite_profiles_from_ratios(
    ratios: np.ndarray,
    types: list[TemporalProfile],
    expected: dict[int, list[TemporalProfile]],
):
    profiles = CompositeTemporalProfiles.from_ratios(ratios, types)

    assert profiles.n_profiles == len(expected)
    for index, expected_profiles in expected.items():
        assert len(profiles[index]) == len(expected_profiles)
        for profile, expected_profile in zip(
            sorted(profiles[index]), sorted(expected_profiles)
        ):
            assert profile == expected_profile


def test_internals():
    # This test should not need to exist and could be invalid if someone changes the internal mechanics

    p = CompositeTemporalProfiles(
        [
            [WeeklyProfile(), DailyProfile()],
            [WeeklyProfile()],
        ]
    )
    assert WeeklyProfile in p._profiles
    assert DailyProfile in p._profiles
    assert WeeklyProfile in p._indexes
    assert DailyProfile in p._indexes
    np.testing.assert_array_equal(p._indexes[WeeklyProfile], [0, 1])
    np.testing.assert_array_equal(p._indexes[DailyProfile], [0, -1])
    assert len(p._profiles[WeeklyProfile]) == 2
    assert len(p._profiles[DailyProfile]) == 1


@pytest.mark.parametrize(
    "profiles, expected",
    [
        (
            [
                CompositeTemporalProfiles(
                    [
                        [WeeklyProfile(), DailyProfile()],
                        [WeeklyProfile()],
                    ]
                ),
                CompositeTemporalProfiles(
                    [
                        [WeeklyProfile(), DailyProfile()],
                        [WeeklyProfile()],
                    ]
                ),
                CompositeTemporalProfiles(
                    [
                        [WeeklyProfile()],
                    ]
                ),
            ],
            {
                0: [WeeklyProfile(), DailyProfile()],
                1: [WeeklyProfile()],
                2: [WeeklyProfile(), DailyProfile()],
                3: [WeeklyProfile()],
                4: [WeeklyProfile()],
            },
        ),
        (
            [
                CompositeTemporalProfiles(
                    [
                        [],
                        [],
                    ]
                ),
                CompositeTemporalProfiles(
                    [
                        [],
                        [],
                    ]
                ),
            ],
            {0: [], 1: [], 2: [], 3: []},
        ),
        (
            [
                CompositeTemporalProfiles(
                    [
                        [WeeklyProfile(), DailyProfile()],
                        [WeeklyProfile()],
                    ]
                ),
            ],
            {
                0: [WeeklyProfile(), DailyProfile()],
                1: [WeeklyProfile()],
            },
        ),
    ],
)
def test_join_composites(profiles, expected):
    joined = CompositeTemporalProfiles.join(*profiles)

    for key, expected_profiles in expected.items():
        assert len(joined[key]) == len(expected_profiles)
        for profile, expected_profile in zip(
            sorted(joined[key]), sorted(expected_profiles)
        ):
            assert profile == expected_profile


def test_equality():
    p1 = WeeklyProfile()
    p2 = WeeklyProfile()
    p3 = WeeklyProfile([0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0])
    p4 = WeeklyProfile([0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0])
    assert p1 == p2
    assert p1 != p3
    assert p3 == p4


def test_concatenate_composite():
    cat_profiles = concatenate_profiles(
        [
            CompositeTemporalProfiles(
                [
                    [WeeklyProfile(), DailyProfile()],
                    [WeeklyProfile()],
                ]
            ),
            CompositeTemporalProfiles(
                [
                    [WeeklyProfile(), DailyProfile()],
                    [WeeklyProfile()],
                ]
            ),
        ]
    )
    assert cat_profiles.n_profiles == 4
    assert len(cat_profiles[0]) == 2
    assert len(cat_profiles[1]) == 1
    assert len(cat_profiles[2]) == 2
    assert len(cat_profiles[3]) == 1


def test_concatenate_lists():
    cat_profiles = concatenate_profiles(
        [
            [
                [WeeklyProfile(), DailyProfile()],
                [WeeklyProfile()],
            ],
            [
                [WeeklyProfile(), DailyProfile()],
                [WeeklyProfile()],
            ],
        ]
    )
    assert cat_profiles.n_profiles == 4
    assert len(cat_profiles[0]) == 2
    assert len(cat_profiles[1]) == 1
    assert len(cat_profiles[2]) == 2
    assert len(cat_profiles[3]) == 1


def test_wrong_list_concatenate():
    pytest.raises(
        TypeError,
        concatenate_profiles,
        [
            [WeeklyProfile(), DailyProfile()],
            [WeeklyProfile()],
        ],
    )


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


def test_weighted_combination():
    weights = np.array([1, 2, 3])
    new_profile = weighted_combination(
        [
            WeeklyProfile([0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0]),
            WeeklyProfile([0.2, 0.1, 0.3, 0.4, 0.0, 0.0, 0.0]),
            WeeklyProfile([0.3, 0.0, 0.3, 0.4, 0.0, 0.0, 0.0]),
        ],
        weights=weights,
    )

    # The combination should give the same as summing the emissions one by one
    assert np.allclose(
        new_profile.ratios.reshape(-1),
        np.array([7 / 30, 2 / 30, 0.3, 0.4, 0.0, 0.0, 0.0]),
    )


if __name__ == "__main__":
    pytest.main([__file__])
