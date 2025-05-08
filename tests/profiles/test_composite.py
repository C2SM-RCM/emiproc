"""Test for composite temporal profiles."""
from __future__ import annotations

import numpy as np
import pytest

from emiproc.profiles.temporal.composite import CompositeTemporalProfiles
from emiproc.profiles.temporal.profiles import (
    DailyProfile,
    SpecificDay,
    SpecificDayProfile,
    TemporalProfile,
    WeeklyProfile,
)
from emiproc.tests_utils.temporal_profiles import Profile2, Profile3


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


# Test data for the composite profiles
profiles_list_list_data = [
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
]


@pytest.mark.parametrize("profiles_list_list", profiles_list_list_data)
def test_composite_temporal_profiles_creation(profiles_list_list):
    p = CompositeTemporalProfiles(profiles_list_list)
    assert p.n_profiles == len(profiles_list_list)
    for key, expected in enumerate(profiles_list_list):
        assert len(p[key]) == len(expected)
        for profile, expected_profile in zip(sorted(p[key]), sorted(expected)):
            assert profile == expected_profile


@pytest.mark.parametrize("profiles_list_list", profiles_list_list_data)
def test_composite_temporal_profiles_append(profiles_list_list):
    p = CompositeTemporalProfiles(profiles_list_list)

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


@pytest.mark.parametrize("profiles_list_list", profiles_list_list_data)
def test_composite_temporal_profiles_append_specific_days(profiles_list_list):
    p = CompositeTemporalProfiles(profiles_list_list)

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
    assert p.n_profiles == len(profiles_list_list) + 1
    assert len(p[-1]) == 2
    for profile, expected_profile in zip(sorted(p[-1]), sorted(append_profiles)):
        assert profile == expected_profile
    # Check that the first profile has not changed (assume okay if the others are)
    assert len(p[0]) == len(profiles_list_list[0])
    for profile, expected_profile in zip(sorted(p[0]), sorted(profiles_list_list[0])):
        assert profile == expected_profile


@pytest.mark.parametrize("profiles_list_list", profiles_list_list_data)
def test_repr(profiles_list_list):
    p = CompositeTemporalProfiles(profiles_list_list)

    # Test repr
    repr_str = repr(p)
    assert isinstance(repr_str, str)


@pytest.mark.parametrize("profiles_list_list", profiles_list_list_data)
def test_types(profiles_list_list):
    p = CompositeTemporalProfiles(profiles_list_list)

    p.types


@pytest.mark.parametrize("profiles_list_list", profiles_list_list_data)
def test_composite_scaling_factors(profiles_list_list):
    p = CompositeTemporalProfiles(profiles_list_list)

    # Test scaling factors
    scaling_factors = p.scaling_factors

    assert scaling_factors.shape == (p.n_profiles, p.size)
    if scaling_factors.shape[1] > 1:
        np.testing.assert_allclose(scaling_factors.mean(axis=1), 1.0)


@pytest.mark.parametrize("profiles_list_list", profiles_list_list_data)
def test_composite_ratios(profiles_list_list):
    p = CompositeTemporalProfiles(profiles_list_list)

    # Test scaling factors
    ratios = p.ratios

    assert ratios.shape == (p.n_profiles, p.size)
    # We cannot test here, because the ratios can contain NaN values
    # if ratios.shape[1] > 1:
    #    np.testing.assert_allclose(ratios.sum(axis=1), p.n_profiles)


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


def test_from_ratios():

    profile = CompositeTemporalProfiles.from_ratios(
        ratios=np.array([[0.1, 0.9, 0.3, 0.5, 0.2]]),
        types=[Profile2, Profile3],
    )


def test_from_rescale():

    profile = CompositeTemporalProfiles.from_ratios(
        ratios=np.array([[0.1, 0.2, 0.3, 0.5, 0.2]]),
        types=[Profile2, Profile3],
        rescale=True,
    )


def test_with_zero():
    """Test when one of the profile is zero."""
    profile = CompositeTemporalProfiles.from_ratios(
        ratios=np.array([[0.0, 0.0, 0.3, 0.5, 0.2]]),
        types=[Profile2, Profile3],
        rescale=True,
    )

    # Will be replaced by a constant profile
    # Test both, because it might be reversed
    assert (
        np.array_equal(profile.ratios, np.array([[0.5, 0.5, 0.3, 0.5, 0.2]]))
        and profile.types == [Profile2, Profile3]
    ) or (
        np.array_equal(profile.ratios, np.array([[0.3, 0.5, 0.2, 0.5, 0.5]]))
        and profile.types == [Profile3, Profile2]
    )
