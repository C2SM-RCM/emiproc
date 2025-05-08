"""Heating degree days (HDD) profile."""

import numpy as np
import pandas as pd
from emiproc.profiles.temporal.profiles import TemporalProfile
from emiproc.profiles.temporal.operators import create_scaling_factors_time_serie


def create_HDD_scaling_factor(
    serie_T: pd.Series,
    heating_profile: list[TemporalProfile],
    dhw_profile: list[TemporalProfile],
    min_heating_T: float = 12.0,
    inside_T: float = 20.0,
    dhw_scaling: float = 5.59 / 24,
) -> pd.Series:
    """Generate the scaling factor for the heating degree days formula.

    The HDD formula procceds this way:
    first Calculate the mean temprature of the day, and the heating demeand for the day

    .. math::
        HDD = (T_{inside} - T_{mean})

    Where :math:`T_{inside}` is the inside temperature
    and :math:`T_{mean}` is the mean temperature of the day.

    If :math:`T_{mean} > T_{min}` then the heating is not activated and
    :math:`HDD = 0`.

    A day of year profile can then be calculated using

    .. math::
        a_{HDD} = \\frac{HDD}{\\overline{HDD}}

    Where :math:`\\overline{HDD}` is the yearly average of the HDD.

    A profile for domestic hot water is then added to this profile.

    .. math::
        a_{H} = (1 - f_{DHW}) * HDD_{H} * a_{HDD} + a_{DHW} * f_{DHW}

    Where :math:`HDD_{H}` is the hourly heating profile and :math:`a_{DHW}` is the
    hourly domestic hot water profile and :math:`f_{DHW}` is the scaling factor for
    the domestic hot water profile.

    Thus the return profile correspond to both heating and domestic hot water,
    with a hourly resolution.

    :arg serie_T: the timeserie of the temperature (in Celsius)
    :arg heating_profile: the heating profile
    :arg dhw_profile: the domestic hot water profile
    :arg min_heating_T: the minimum temperature for which heating is activated
    :arg inside_T: the inside temperature
    :arg dhw_scaling: :math:`f_{DHW}`, the scaling of domesting hot water demand vs surface heating
        must be a ratio between 0 and 1. If 0, the profiles will correspond to
        space heating only.
    """
    serie_daily_T = serie_T.resample("D").mean()

    # Heating degrees of the day
    heating_activated = serie_daily_T < min_heating_T
    HDD = (inside_T - serie_daily_T) * heating_activated.astype(float)

    # Yearly avergage
    yearly_means = HDD.resample("Y").mean()
    ts_mean = pd.Series(np.nan, index=HDD.index)
    for dt, mean in yearly_means.items():
        ts_mean.loc[ts_mean.index.year == dt.year] = mean
    # Scale with the yearly means
    a_HDD = HDD / ts_mean

    start = serie_T.index.min()
    end = serie_T.index.max()
    hdd_ts = pd.Series(np.nan, pd.date_range(start, end, freq="H"))
    # Get hourly values
    a_HDD_hourly = (
        a_HDD.reindex(index=a_HDD.index.union(hdd_ts.index))
        .interpolate("pad")
        .reindex(hdd_ts.index)
    )

    heating_ts = create_scaling_factors_time_serie(
        start, end, heating_profile, local_tz="Europe/Zurich"
    )
    dhw_ts = create_scaling_factors_time_serie(
        start, end, dhw_profile, local_tz="Europe/Zurich"
    )

    return (1.0 - dhw_scaling) * a_HDD_hourly * heating_ts + dhw_ts * dhw_scaling
