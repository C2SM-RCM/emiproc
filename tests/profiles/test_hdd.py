import numpy as np
import pandas as pd

from emiproc import FILES_DIR
from emiproc.profiles.hdd import create_HDD_scaling_factor
from emiproc.profiles.temporal.io import from_yaml
from emiproc.profiles.temporal.operators import create_scaling_factors_time_serie


def _sample_temperature_series(start_date="2022-01-01 00:00") -> pd.Series:
    index = pd.date_range(start_date, periods=72, freq="h", tz="UTC")
    values = np.concatenate(
        [
            np.full(24, 0.0),
            np.full(24, 15.0),
            np.full(24, 0.0),
        ]
    )
    return pd.Series(values, index=index)


heating_profile = from_yaml(FILES_DIR / "profiles" / "yamls" / "heat.yaml")
dhw_profile = from_yaml(FILES_DIR / "profiles" / "yamls" / "no_factor.yaml")
no_factor = from_yaml(FILES_DIR / "profiles" / "yamls" / "no_factor.yaml")


def test_create_hdd_scaling_factor_returns_valid_hourly_series():
    serie_t = _sample_temperature_series()

    scaling = create_HDD_scaling_factor(
        serie_t,
        heating_profile=heating_profile,
        dhw_profile=dhw_profile,
    )

    assert isinstance(scaling, pd.Series)
    assert scaling.index.equals(serie_t.index)
    assert scaling.index.tz == serie_t.index.tz
    assert np.isfinite(scaling).all()


def test_create_hdd_scaling_factor_no_heating_days():
    serie_t = pd.Series(
        np.full(72, 20.0),
        index=pd.date_range("2022-01-01 00:00", periods=72, freq="h", tz="UTC"),
    )

    scaling = create_HDD_scaling_factor(
        serie_t,
        heating_profile=heating_profile,
        dhw_profile=dhw_profile,
    )

    assert isinstance(scaling, pd.Series)
    assert scaling.index.equals(serie_t.index)
    assert np.isfinite(scaling).all()


def test_create_hdd_scaling_factor_returns_valid_hourly_series_mid_year():
    serie_t = _sample_temperature_series(start_date="2022-06-02 00:00")

    scaling = create_HDD_scaling_factor(
        serie_t,
        heating_profile=heating_profile,
        dhw_profile=dhw_profile,
    )

    assert isinstance(scaling, pd.Series)
    assert scaling.index.equals(serie_t.index)
    assert scaling.index.tz == serie_t.index.tz
    assert np.isfinite(scaling).all()


def test_create_hdd_scaling_factor_matches_dhw_profile_when_dhw_scaling_is_one():
    serie_t = _sample_temperature_series()

    scaling = create_HDD_scaling_factor(
        serie_t,
        heating_profile=heating_profile,
        dhw_profile=dhw_profile,
        dhw_scaling=1.0,
    )

    expected = create_scaling_factors_time_serie(
        serie_t.index.min(),
        serie_t.index.max(),
        dhw_profile,
    )

    np.testing.assert_allclose(scaling.values, expected.values)


def test_create_hdd_scaling_factor_deactivates_heating_on_warm_days():
    serie_t = _sample_temperature_series()

    scaling = create_HDD_scaling_factor(
        serie_t,
        heating_profile=no_factor,
        dhw_profile=no_factor,
        dhw_scaling=0.0,
        min_heating_T=12.0,
    )

    day_means = scaling.resample("D").mean()

    assert day_means.iloc[0] > 0
    assert np.isclose(day_means.iloc[1], 0.0)
    assert day_means.iloc[2] > 0


def test_create_hdd_scaling_factor_with_naive_datetime_index():
    index = pd.date_range("2022-01-01 00:00", periods=24, freq="h")
    serie_t = pd.Series(np.full(24, 5.0), index=index)

    create_HDD_scaling_factor(
        serie_t,
        heating_profile=no_factor,
        dhw_profile=no_factor,
    )


def test_create_hdd_scaling_factor_other_tz():
    index = pd.date_range("2022-01-01 00:00", periods=24, freq="h", tz="Europe/Zurich")
    serie_t = pd.Series(np.full(24, 5.0), index=index)

    sf = create_HDD_scaling_factor(
        serie_t,
        heating_profile=no_factor,
        dhw_profile=no_factor,
    )

    assert sf.index.tz == serie_t.index.tz
