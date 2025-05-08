from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from emiproc import FILES_DIR
from emiproc.profiles.temporal.io import from_yaml
from emiproc.profiles.temporal.operators import (
    create_scaling_factors_time_serie,
    get_profile_da,
)
from emiproc.profiles.temporal.profiles import (
    DailyProfile,
    SpecificDay,
    SpecificDayProfile,
)
from emiproc.profiles.temporal.specific_days import days_of_specific_day
from emiproc.profiles.temporal.utils import ensure_specific_days_consistency
from emiproc.tests_utils import temporal_profiles


def test_ensure_specific_days_consistency_non_specific():
    out = ensure_specific_days_consistency(temporal_profiles.oem_const_profile)
    assert len(out) == len(temporal_profiles.oem_const_profile)

    out = ensure_specific_days_consistency(
        [SpecificDayProfile(specific_day=SpecificDay("monday"))]
    )
    assert len(out) == 7  # Contains the full week

    half_double_ratios = np.concatenate((np.full(12, 0.5), np.full(12, 1.5))) / 24
    thirds_triple_ratios = np.concatenate((np.full(12, 1 / 3), np.full(12, 5 / 3))) / 24
    quaters_ratios = np.concatenate((np.full(12, 1 / 4), np.full(12, 7 / 4))) / 24
    out = ensure_specific_days_consistency(
        [
            SpecificDayProfile(
                ratios=half_double_ratios, specific_day=SpecificDay.MONDAY
            ),
            SpecificDayProfile(
                ratios=thirds_triple_ratios, specific_day=SpecificDay.WEEKEND
            ),
            SpecificDayProfile(ratios=quaters_ratios, specific_day=SpecificDay.SUNDAY),
            DailyProfile(),
        ],
    )
    for p in out:
        assert isinstance(p, SpecificDayProfile)
        assert len(days_of_specific_day(p.specific_day)) == 1
        if p.specific_day == SpecificDay.MONDAY:
            equalt_to = half_double_ratios
        elif p.specific_day == SpecificDay.SATURDAY:
            equalt_to = thirds_triple_ratios
        elif p.specific_day == SpecificDay.SUNDAY:
            equalt_to = quaters_ratios
        else:
            equalt_to = np.full(24, 1.0) / 24
        np.testing.assert_array_equal(
            p.ratios.reshape(-1), equalt_to, err_msg=f"Failed for {p.specific_day=}"
        )


@pytest.mark.parametrize(
    "profiles",
    [
        temporal_profiles.oem_const_profile,
        from_yaml(FILES_DIR / "profiles" / "yamls" / "heat.yaml"),
        from_yaml(FILES_DIR / "profiles" / "yamls" / "human_home.yaml"),
        from_yaml(FILES_DIR / "profiles" / "yamls" / "human.yaml"),
        from_yaml(FILES_DIR / "profiles" / "yamls" / "ship.yaml"),
        from_yaml(FILES_DIR / "profiles" / "yamls" / "heavy.yaml"),
        from_yaml(FILES_DIR / "profiles" / "yamls" / "light.yaml"),
    ],
)
def test_create_scaling_factors_time_serie(profiles):
    start = datetime(2018, 1, 1)
    end = datetime(2019, 1, 1)
    ts = create_scaling_factors_time_serie(start, end, profiles)

    assert np.isclose(ts.mean(), 1.0, atol=0.1)


@pytest.mark.parametrize(
    "profile",
    [
        temporal_profiles.daily_test_profile,
        temporal_profiles.weekly_test_profile,
        temporal_profiles.mounths_test_profile,
    ],
)
def test_profile_da(profile):

    da = get_profile_da(profile, year=2018)
    assert isinstance(da, xr.DataArray)
