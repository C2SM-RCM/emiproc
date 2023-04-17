"""Temporal profiles."""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd


# Constants
N_HOUR_DAY = 24
N_DAY_WEEK = 7
N_MONTH_YEAR = 12
N_DAY_YEAR = 365
N_DAY_LEAPYEAR = 366
N_HOUR_YEAR = N_DAY_YEAR * N_HOUR_DAY
N_HOUR_LEAPYEAR = N_DAY_LEAPYEAR * N_HOUR_DAY


@dataclass
class TemporalProfile:
    """Temporal profile.

    Temporal profile defines how the emission is distributed over time.

    The ratios must sum up to 1.
    """

    size: int = 0
    ratios: np.ndarray = field(default_factory=lambda: np.ones(0))

    def __post_init__(self) -> None:
        # Make sure the size is a int and the array has the correct size
        if self.size != len(self.ratios):
            raise ValueError(
                f"{len(self.ratios)=} does not match profile's {self.size=}."
            )

        # Make sure the ratios sum up to 1
        if not np.isclose(self.ratios.sum(), 1.0):
            raise ValueError(f"{self.ratios.sum()=} does not sum up to 1.")


@dataclass
class DailyProfile(TemporalProfile):
    """Daily profile.

    Daily profile defines how the emission is distributed over the day.
    """

    size: int = field(default=N_HOUR_DAY, init=False)
    ratios: np.ndarray = field(default_factory=lambda: np.ones(N_HOUR_DAY) / N_HOUR_DAY)


@dataclass
class WeeklyProfile(TemporalProfile):
    """Weekly profile.

    Weekly profile defines how the emission is distributed over the week.
    """

    size: int = N_DAY_WEEK
    ratios: np.ndarray = field(default_factory=lambda: np.ones(N_DAY_WEEK) / N_DAY_WEEK)


@dataclass
class MounthsProfile(TemporalProfile):
    """Yearly profile.

    Yearly profile defines how the emission is distributed over the year using months.
    """

    size: int = N_MONTH_YEAR
    ratios: np.ndarray = field(
        default_factory=lambda: np.ones(N_MONTH_YEAR) / N_MONTH_YEAR
    )


@dataclass
class HourOfYearProfile(TemporalProfile):
    """Hour of year profile.

    Hour of year profile defines how the emission is distributed over the year using hours.
    """

    size: int = N_HOUR_YEAR
    ratios: np.ndarray = field(
        default_factory=lambda: np.ones(N_HOUR_YEAR) / N_HOUR_YEAR
    )


@dataclass
class HourOfLeapYearProfile(TemporalProfile):
    """Hour of leap year profile.

    Hour of leap year profile defines how the emission is distributed over the year using hours.
    """

    size: int = N_HOUR_LEAPYEAR
    ratios: np.ndarray = field(
        default_factory=lambda: np.ones(N_HOUR_LEAPYEAR) / N_HOUR_LEAPYEAR
    )


def create_time_serie(
    start_time: datetime, end_time: datetime, profiles: list[TemporalProfile]
) -> pd.Series:
    """Create a time serie of ratios for the requested time range."""

    # Create the time serie
    time_serie = pd.date_range(start_time, end_time, freq="H")

    # Create the scaling factors
    ratios = np.ones(len(time_serie)) / len(time_serie)

    # Apply the profiles
    for profile in profiles:
        ratios *= profile_to_scaling_factors(time_serie, profile)

    return pd.Series(ratios, index=time_serie)


def profile_to_scaling_factors(
    time_serie: pd.DatetimeIndex, profile: TemporalProfile
) -> np.ndarray:
    """Apply a temporal profile to a time serie.

    :return: Scaling factors
        An array by which you can multiply the emission factor using the time serie.
    """

    # Get scaling factors, convert the ratios to scaling factors
    factors = profile.ratios * profile.size

    scaling_factors = np.ones(len(time_serie))

    # Get the profile
    if isinstance(profile, DailyProfile):
        # Get the mask for each hour of day and apply the scaling factor
        for hour, factor in enumerate(factors):
            scaling_factors[time_serie.hour == hour] *= factor
    elif isinstance(profile, WeeklyProfile):
        # Get the mask for each day of week and apply the scaling factor
        for day, factor in enumerate(factors):
            scaling_factors[time_serie.dayofweek == day] *= factor
    elif isinstance(profile, MounthsProfile):
        # Get the mask for each month of year and apply the scaling factor
        for month, factor in enumerate(factors):
            # Months start with 1
            month += 1
            scaling_factors[time_serie.month == month] *= factor
    else:
        raise NotImplementedError(
            f"Cannot apply {profile=}, {type(profile)=} is not implemented."
        )

    # Return the scaling factors
    return scaling_factors
