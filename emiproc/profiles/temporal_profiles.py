"""Temporal profiles."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import logging
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Constants
N_HOUR_DAY = 24
N_DAY_WEEK = 7
N_MONTH_YEAR = 12
N_DAY_YEAR = 365
N_DAY_LEAPYEAR = 366
N_HOUR_WEEK = N_HOUR_DAY * N_DAY_WEEK
N_HOUR_YEAR = N_DAY_YEAR * N_HOUR_DAY
N_HOUR_LEAPYEAR = N_DAY_LEAPYEAR * N_HOUR_DAY


# An enum to define specific days
class SpecificDay(Enum):
    # Make it automatically assign the value to the name
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

    MONDAY = auto()
    TUESDAY = auto()
    WEDNESDAY = auto()
    THURSDAY = auto()
    FRIDAY = auto()
    SATURDAY = auto()
    SUNDAY = auto()

    WEEKDAY = auto()  # One of the 5 first days
    WEEKEND = auto()  # One of the 2 last days


def get_days_as_ints(specific_day: SpecificDay) -> list[int]:
    """Return the days corresponding for a specific day."""

    if not isinstance(specific_day, SpecificDay):
        raise TypeError(f"{specific_day=} must be a {SpecificDay}.")

    if specific_day == SpecificDay.MONDAY:
        return [0]
    elif specific_day == SpecificDay.TUESDAY:
        return [1]
    elif specific_day == SpecificDay.WEDNESDAY:
        return [2]
    elif specific_day == SpecificDay.THURSDAY:
        return [3]
    elif specific_day == SpecificDay.FRIDAY:
        return [4]
    elif specific_day == SpecificDay.SATURDAY:
        return [5]
    elif specific_day == SpecificDay.SUNDAY:
        return [6]
    elif specific_day == SpecificDay.WEEKDAY:
        return [0, 1, 2, 3, 4]
    elif specific_day == SpecificDay.WEEKEND:
        return [5, 6]
    else:
        raise NotImplementedError(
            f"{specific_day=} is implemented in  {get_days_as_ints}"
        )


def get_emep_shift(country_code: int) -> int:
    """Retunr the time shift form the country code of emep."""
    logging.error("'get_emep_shift' is not implemented yet. Time shifts are all 0.")
    return 0

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

        if isinstance(self.ratios, list):
            self.ratios = np.array(self.ratios)

        # Make sure the ratios sum up to 1
        if not np.isclose(self.ratios.sum(), 1.0):
            raise ValueError(f"{self.ratios.sum()=} does not sum up to 1.")


@dataclass
class DailyProfile(TemporalProfile):
    """Daily profile.

    Daily profile defines how the emission is distributed over the day.
    The profile starts at 00:00.
    """

    size: int = field(default=N_HOUR_DAY, init=False)
    ratios: np.ndarray = field(default_factory=lambda: np.ones(N_HOUR_DAY) / N_HOUR_DAY)


@dataclass
class SpecificDayProfile(DailyProfile):
    """Same as DailyProfile but with a specific day of the week."""

    specific_day: SpecificDay | None = None


@dataclass
class WeeklyProfile(TemporalProfile):
    """Weekly profile.

    Weekly profile defines how the emission is distributed over the week.
    The profile starts on Monday.
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
class HourOfWeekProfile(TemporalProfile):
    """Hour of week profile.

    Hour of week profile defines how the emission is distributed over the week using hours.
    This is useful if you want to account for different daily patterns over the
    days of the week (usually for the weekend)

    The profile starts on Monday at 00:00.
    """

    size: int = N_HOUR_WEEK
    ratios: np.ndarray = field(
        default_factory=lambda: np.ones(N_HOUR_WEEK) / N_HOUR_WEEK
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
    elif isinstance(profile, SpecificDayProfile):
        # Find the days corresponding to this factor
        days_allowed = get_days_as_ints(profile.specific_day)
        mask_matching_day = np.isin(time_serie.day_of_week, days_allowed)
        for hour, factor in enumerate(factors):
            # Other days will not have a scaling factor
            scaling_factors[(time_serie.hour == hour) & mask_matching_day] *= factor
    else:
        raise NotImplementedError(
            f"Cannot apply {profile=}, {type(profile)=} is not implemented."
        )

    # Return the scaling factors
    return scaling_factors


AnyTimeProfile = (
    DailyProfile
    | WeeklyProfile
    | MounthsProfile
    | HourOfYearProfile
    | HourOfLeapYearProfile
)


def from_csv(
    file: PathLike,
) -> dict[str | tuple[str, str] : AnyTimeProfile]:
    """Create the profile from a csv file.

    Based on the file, guess the correct profile
    to create
    """
    logger = logging.getLogger("emiproc.profiles.from_csv")

    file = Path(file)

    df = pd.read_csv(file)

    if "Category" in df.columns:
        cat_header = "Category"
    else:
        raise ValueError(f"Cannot find 'Category' header in {file=}")

    if "Substance" in df.columns:
        sub_header = "Substance"
    else:
        sub_header = None
        logger.warning(
            f"Cannot find 'Substance' header in {file=}."
            "All substances will be treated the same way."
        )

    logger.error(f"{df.columns=}")

    if "Mon" in df.columns:
        # Weekly profile with 3 letters identification
        data_columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        profile = WeeklyProfile
    elif "Monday" in df.columns:
        # Weekly profile with full name identification
        data_columns = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        profile = WeeklyProfile
    elif "Jan" in df.columns:
        # Yearly profile with 3 letters identification
        data_columns = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        profile = MounthsProfile
    elif "January" in df.columns:
        # Yearly profile with full name identification
        data_columns = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        profile = MounthsProfile
    elif (
        "0" not in df.columns
        and "1" in df.columns
        and "24" in df.columns
        and "25" not in df.columns
    ):
        # Daily profile
        data_columns = [str(i) for i in range(1, 25)]
        profile = DailyProfile
    else:
        raise NotImplementedError(f"Cannot guess the profile from {file=}")

    # Check that all the data columns are in the file
    for col in data_columns:
        if not col in df.columns:
            raise ValueError(
                f"Cannot find {col=} in {file=}, required to build {profile=}"
            )

    # Read all rows to generate the profiles
    profiles = {}
    for _, row in df.iterrows():
        cat = row[cat_header]
        if sub_header is not None:
            sub = row[sub_header]
            key = (cat, sub)
        else:
            key = cat
        ratios = np.array([row[col] for col in data_columns])

        # Check if scaling factors are given instead of ratios
        if np.isclose(ratios.sum(), len(ratios)):
            ratios = ratios / ratios.sum()
        elif not np.isclose(ratios.sum(), 1.0):
            pass  # Already ratios
        else:
            raise ValueError(
                f"Cannot guess if {ratios=} are ratios or scaling factors for {row=} in {file=}."
            )
        profiles[key] = profile(ratios=ratios)

    return profiles


def from_yaml(yaml_file: PathLike) -> list[AnyTimeProfile]:
    """Read a yml file containing a temporal profile.

    Only one temporal profile is currently accepted in the yaml definition.
    """
    logger = logging.getLogger("emiproc.profiles.from_yaml")
    yaml_file = Path(yaml_file)

    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        logger.warning(f"Empty yaml file {yaml_file=}")
        return []
    elif not isinstance(data, dict):
        raise ValueError(f"Invalid yaml file {yaml_file=}, expected to load a dict.")

    profiles = []

    # Create possible aliases for the names
    profiles_mapping: dict[AnyTimeProfile, list[str]] = {
        DailyProfile: ["diurn", "daily", "day"],
        SpecificDayProfile: [
            "diurn_weekday",
            "diurn_weekend",
            "diurn_saturday",
            "diurn_sunday",
        ],
        WeeklyProfile: ["weekly", "week"],
        MounthsProfile: ["season", "year", "monthly", "month"],
    }
    profile_of_key = {
        key: profile for profile, keys in profiles_mapping.items() for key in keys
    }

    _types_added = []
    # Check that the yaml does not contain any unkown key
    for key in data.keys():
        if not any(key in profile_names for profile_names in profiles_mapping.values()):
            logger.warning(f"Unknown key {key=} in {yaml_file=}")
            continue
        profile_class = profile_of_key[key]
        # Check that a profile of that type was not already added
        if profile_class in _types_added:
            raise ValueError(
                f"Cannot add {key=} to {yaml_file=} as a {profile_class=} was already added."
            )
        # add the profile
        ratios = data[key]
        # Check the ratio
        if not np.isclose(np.sum(ratios), 1.0):
            raise ValueError(
                f"{ratios=} in {yaml_file=} do not sum to 1 but {np.sum(ratios)}."
            )

        # Add additional information on the profiles if requried
        kwargs = {}
        if profile_class is SpecificDayProfile:
            # get the type of the profile
            profile_type = key.split("_")[-1]
            # Add the selected day
            kwargs["specific_day"] = SpecificDay(profile_type)

        try:
            profile = profile_class(ratios=ratios, **kwargs)
        except Exception as e:
            raise ValueError(
                f"Cannot create profile {key=} from {yaml_file=} with {ratios=}"
            ) from e

        profiles.append(profile)

    if len(profiles) == 0:
        logger.warning(f"No profile found in {yaml_file=}")
    return profiles
