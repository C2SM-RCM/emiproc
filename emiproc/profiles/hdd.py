"""Heating degree days (HDD) profile."""
import numpy as np
import pandas as pd
from emiproc.profiles.temporal_profiles import TemporalProfile, create_scaling_factors_time_serie

def create_HDD_scaling_factor(
    serie_T: pd.Series,
    heating_profile: list[TemporalProfile],
    dhw_profile: list[TemporalProfile],
    min_heating_T: float = 12.0,
    inside_T: float = 20.0,
    dhw_scaling: float = 5.59 / 24,
) -> pd.Series:
    """Generate the scaling factor for the heating degree days formula.

    :arg serie_T: the timeserie of the temperature (in Celsius)
    :arg heating_profile: the heating profile
    :arg dhw_profile: the domestic hot water profile
    :arg min_heating_T: the minimum temperature for which heating is activated
    :arg inside_T: the inside temperature
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

    heating_ts = create_scaling_factors_time_serie(start, end, heating_profile)
    dhw_ts = create_scaling_factors_time_serie(start, end, dhw_profile)

    return a_HDD_hourly * heating_ts + dhw_ts * dhw_scaling

