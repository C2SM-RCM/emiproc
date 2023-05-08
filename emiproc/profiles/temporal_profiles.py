"""Temporal profiles."""
from __future__ import annotations
import yaml
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from emiproc.profiles.utils import (
    read_profile_csv,
    remove_objects_of_type_from_list,
    type_in_list,
    ratios_to_factors
)

# Constants
N_HOUR_DAY = 24
N_DAY_WEEK = 7
N_MONTH_YEAR = 12
N_DAY_YEAR = 365
N_DAY_LEAPYEAR = 366
N_HOUR_WEEK = N_HOUR_DAY * N_DAY_WEEK
N_HOUR_YEAR = N_DAY_YEAR * N_HOUR_DAY
N_HOUR_LEAPYEAR = N_DAY_LEAPYEAR * N_HOUR_DAY




class SpecificDay(Enum):
    """An enum to define specific day applied of a profile."""

    # Make it automatically assign the value to the name
    def _generate_next_value_(name: str, start, count, last_values):
        return name.lower()

    MONDAY = auto()
    TUESDAY = auto()
    WEDNESDAY = auto()
    THURSDAY = auto()
    FRIDAY = auto()
    SATURDAY = auto()
    SUNDAY = auto()

    # One of the 5 first days
    WEEKDAY = auto()

    # One of the 2 last days
    WEEKEND = auto()


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


AnyTimeProfile = (
    DailyProfile
    | SpecificDayProfile
    | WeeklyProfile
    | MounthsProfile
    | HourOfYearProfile
    | HourOfLeapYearProfile
)


def create_scaling_factors_time_serie(
    start_time: datetime,
    end_time: datetime,
    profiles: list[AnyTimeProfile],
    apply_month_interpolation: bool = True,
    freq: str = "H",
) -> pd.Series:
    """Create a time serie of ratios for the requested time range.

    :arg start_time: The start time of the time serie.
    :arg end_time: The end time of the time serie.
    :arg profiles: The profiles to use to create .
    :arg apply_month_interpolation: If True, apply the month interpolation.
    """

    # Create the time serie
    time_serie = pd.date_range(start_time, end_time, freq=freq)

    # Create the scaling factors
    scaling_factors = np.ones(len(time_serie))

    # Apply the profiles
    for profile in profiles:
        scaling_factors *= profile_to_scaling_factors(
            time_serie, profile, apply_month_interpolation=apply_month_interpolation
        )

    return pd.Series(scaling_factors, index=time_serie)


def profile_to_scaling_factors(
    time_serie: pd.DatetimeIndex,
    profile: AnyTimeProfile,
    apply_month_interpolation: bool = True,
) -> np.ndarray:
    """Convert a temporal profile to a time serie.

    :arg apply_month_interpolation: If True, apply the month interpolation.
        Only applies when the profile is a :py:class: MounthsProfile
        Each mounthly values is assinged to the 15th of the month and then
        interpolation is used to get the values for the other days.

    :return: Scaling factors
        An array by which you can multiply the emission factor using the time serie.

    """

    # Get scaling factors, convert the ratios to scaling factors
    factors = ratios_to_factors(profile.ratios)

    # This will be the output
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
        if apply_month_interpolation:
            # Apply the factor to the 15 th of each month by getting the exact datetime
            # of the 15th of each month
            mid_months = pd.date_range(
                start=time_serie[0],
                end=time_serie[-1],
                freq="MS",
            ) + timedelta(days=14)
            mid_months_factors = np.ones(len(mid_months))
            # Set the value to each month
            for month, factor in enumerate(factors):
                # Months start with 1
                month += 1
                mid_months_factors[mid_months.month == month] *= factor
            # Interpolate the values to the other days
            scaling_factors = np.interp(
                time_serie,
                mid_months,
                mid_months_factors,
            )
        else:
            # Simply apply to each month the scaling factor
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





def read_temporal_profiles(
    profiles_dir: PathLike,
    time_profiles_files_format: str = "timeprofiles-*.csv",
    profile_csv_kwargs: dict[str, Any] = {},
) -> tuple[list[list[AnyTimeProfile]], xr.DataArray]:
    """Read the temporal profiles csv files to the emiproc inventory format.

    The files for the time profiles are csv and must be all in the same directory
    named according to the argument `time_profiles_files_format`.

    Extra arguments depending on the file format can be passed to
    the function :py:func:`temporal_profiles.from_csv`
    that reads the csv files using: `profile_csv_kwargs`

    This returns the time profiles read, and and index xarray matching
    each substance and category to the desired profiles.


    """

    # Note: The logic of this is a bit tricky because it has to handle
    #       the case where the profiles are speciated or not.

    profiles_dir = Path(profiles_dir)

    # List files with the expected format
    files = list(profiles_dir.glob(time_profiles_files_format))
    if not files:
        raise FileNotFoundError(
            f"Cannot find any file matching {time_profiles_files_format=} in {profiles_dir=}"
        )

    categories = []
    substances = []

    speciated_categories = []

    profiles: dict[str | tuple[str, str], list[TemporalProfile]] = {}
    for file in files:
        file_profiles = from_csv(
            file,
            profile_csv_kwargs=profile_csv_kwargs,
        )
        # Add the read profiles
        for key, profile in file_profiles.items():
            # Find if it is a speciated profile
            if isinstance(key, tuple):
                cat, sub = key
                if sub not in substances:
                    substances.append(sub)
                if cat not in speciated_categories:
                    speciated_categories.append(cat)
            elif isinstance(key, str):
                cat = key
                sub = None
            else:
                raise TypeError(f"Unexpected key type {type(key)} from {file=}")

            if cat not in categories:
                categories.append(cat)

            if key not in profiles:
                # Cat sub found
                profiles[key] = []
                if sub is not None and cat in profiles:
                    # If fisrt time we see this speciated, add the general profiles as well
                    profiles[(cat, sub)] = profiles[cat].copy()
                    profiles[(cat, sub)] = remove_objects_of_type_from_list(
                        profile, profiles[(cat, sub)]
                    )

            # simply add the speciated profile
            profiles[key].append(profile)

            # We need to add the general to all the speciated profiles
            if sub is None and cat in speciated_categories:
                for sub_ in substances:
                    if (cat, sub_) in profiles:
                        catsub_profiles = profiles[(cat, sub_)]
                        if type_in_list(profile, catsub_profiles):
                            # Replace the previous nonspeciated profile
                            catsub_profiles = remove_objects_of_type_from_list(
                                profile, catsub_profiles
                            )
                        catsub_profiles.append(profile)
                        profiles[(cat, sub_)] = catsub_profiles

    # Now that we have extracted the profiles for each substance and category, we can fill the array
    array = []
    out_profiles = []
    counter = 0
    for cat in categories:
        cat_counter = counter
        out_profiles.append(profiles[cat])
        counter += 1
        if substances:
            # Speciated cases
            vector = []
            array.append(vector)
        else:
            array.append(cat_counter)
        for sub in substances:
            if (cat, sub) in profiles:
                out_profiles.append(profiles[(cat, sub)])
                vector.append(counter)
                counter += 1
            else:
                vector.append(cat_counter)

    dims = ["category"]
    coords = {"category": categories}
    if substances:
        dims.append("substance")
        coords["substance"] = substances

    indexes = xr.DataArray(
        np.asarray(array, dtype=int),
        dims=dims,
        coords=coords,
    )

    return out_profiles, indexes


def from_csv(
    file: PathLike,
    profile_csv_kwargs: dict[str, Any] = {},
) -> dict[str | tuple[str, str] : AnyTimeProfile]:
    """Create the profile from a csv file.

    Based on the file, guess the correct profile
    to create
    """

    df, cat_header, sub_header = read_profile_csv(file, **profile_csv_kwargs)

    if "Mon" in df.columns:
        # Weekly profile with 3 letters identification
        data_columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        profile = WeeklyProfile
    elif "mon" in df.columns:
        # Weekly profile with 3 letters identification
        data_columns = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        profile = WeeklyProfile
    elif " mon " in df.columns:
        data_columns = [" mon ", " tue ", " wed ", " thu ", " fri ", " sat ", " sun "]
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
    elif " jan " in df.columns:
        data_columns = [
            " jan ",
            " feb ",
            " mar ",
            " apr ",
            " may ",
            " jun ",
            " jul ",
            " aug ",
            " sep ",
            " oct ",
            " nov ",
            " dec ",
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


def to_yaml(profiles: list[AnyTimeProfile], yaml_file: PathLike):
    """Write a list of profiles to a yaml file."""
    yaml_file = Path(yaml_file)
    yaml_file.parent.mkdir(parents=True, exist_ok=True)

    data = {}
    for profile in profiles:
        if isinstance(profile, DailyProfile):
            if isinstance(profile, SpecificDayProfile):
                key = f"diurn_{profile.specific_day.value}"
            else:
                key = "daily"
        elif isinstance(profile, WeeklyProfile):
            key = "weekly"
        elif isinstance(profile, MounthsProfile):
            key = "monthly"
        else:
            raise NotImplementedError(f"Cannot write {profile=}")

        data[key] = profile.ratios.tolist()

    with open(yaml_file, "w") as f:
        yaml.dump(data, f)
