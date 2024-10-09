"""Temporal profiles."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from os import PathLike
from pathlib import Path
from typing import Any, Type, Union

import numpy as np
import pandas as pd
import xarray as xr
import yaml

import emiproc
from emiproc.profiles.utils import (
    get_profiles_indexes,
    load_country_tz,
    merge_indexes,
    ratios_dataarray_to_profiles,
    ratios_to_factors,
    read_profile_csv,
    read_profile_file,
)

logger = logging.getLogger(__name__)

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

    @classmethod
    def from_day_number(cls, day_number: int) -> SpecificDay:
        """Return the day corresponding to the day number.

        Day start at 0 for Monday and ends at 6 for Sunday.
        """
        if not isinstance(day_number, int):
            raise TypeError(f"{day_number=} must be an int.")

        dict_day = {
            0: cls.MONDAY,
            1: cls.TUESDAY,
            2: cls.WEDNESDAY,
            3: cls.THURSDAY,
            4: cls.FRIDAY,
            5: cls.SATURDAY,
            6: cls.SUNDAY,
        }

        if day_number not in dict_day:
            raise ValueError(f"{day_number=} is not a valid day number.")

        return dict_day[day_number]

    def __lt__(self, other: SpecificDay) -> bool:
        """Compare the days by the order of the week."""
        if not isinstance(other, SpecificDay):
            raise TypeError(f"{other=} must be a {SpecificDay}.")
        return self.value < other.value


def days_of_specific_day(specific_day: SpecificDay) -> list[SpecificDay]:
    int_days = get_days_as_ints(specific_day)
    return [SpecificDay.from_day_number(i) for i in int_days]


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


COUNTRY_TZ_DF = None


def get_emep_shift(country_code_iso3: str) -> int:
    """Retunr the time shift form the country code of emep."""
    global COUNTRY_TZ_DF
    # if COUNTRY_TZ_DF is None:
    COUNTRY_TZ_DF = load_country_tz()

    return COUNTRY_TZ_DF.loc[country_code_iso3, "timezone"]


def concatenate_time_profiles(profiles: list[AnyTimeProfile]) -> AnyTimeProfile:
    """Concatenate the time profiles in the list."""
    if not isinstance(profiles, list):
        raise TypeError(f"{profiles=} must be a list.")

    if not profiles:
        raise ValueError(f"{profiles=} must not be empty.")

    if len(profiles) == 1:
        return profiles[0]

    # Get the type of the profiles
    profile_type = type(profiles[0])

    # Check that all the profiles are of the same type
    if not all(isinstance(p, profile_type) for p in profiles):
        raise ValueError(
            f"{profiles=} must all be of the same type, got {profile_type=}."
        )

    # Concatenate the ratios
    ratios = np.concatenate([p.ratios for p in profiles])

    return profile_type(ratios)


@dataclass(eq=False)
class TemporalProfile:
    """Temporal profile.

    Temporal profile defines how the emission is distributed over time.
    """

    ratios: np.ndarray | None = None
    size: int = 1

    def __post_init__(self) -> None:
        # Check the size is an int
        if not isinstance(self.size, int):
            raise TypeError(f"{self.size=} must be an int.")
        if self.size < 1:
            raise ValueError(f"{self.size=} must be positive.")

        if self.ratios is None:
            self.ratios = np.ones((1, self.size)) / self.size
        else:
            if isinstance(self.ratios, list):
                self.ratios = np.array(self.ratios)
            if len(self.ratios.shape) == 1:
                self.ratios = self.ratios.reshape((1, -1))
            # Make sure the size is a int and the array has the correct size
            if self.size != self.ratios.shape[1]:
                raise ValueError(
                    f"{self.ratios.shape[1]=} does not match profile's {self.size=}."
                )

            # Make sure the ratios sum up to 1
            if not np.all(np.isclose(self.ratios.sum(axis=1), 1.0)):
                raise ValueError(f"{self.ratios.sum(axis=1)=} are not all 1.")

    @property
    def n_profiles(self) -> int:
        return self.ratios.shape[0]

    def __getitem__(self, key: int) -> TemporalProfile:
        """Return the profile at the given index."""
        return self.__class__(self.ratios[key])

    def __len__(self) -> int:
        return self.n_profiles

    def __iter__(self):
        for i in range(self.n_profiles):
            yield self[i]

    # Defie the equality of the profiles
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            raise TypeError(f"{other=} must be a {type(self)}.")
        if self.n_profiles != other.n_profiles:
            return False
        return (self.ratios == other.ratios).all()

    # Defined greater smaller by the size attribute
    def __lt__(self, other: AnyTimeProfile) -> bool:
        # All subclasses can be compared
        if not isinstance(other, TemporalProfile):
            raise TypeError(f"{other=} must be a {TemporalProfile}.")
        return self.size < other.size


@dataclass(eq=False)
class DailyProfile(TemporalProfile):
    """Daily profile.

    Daily profile defines how the emission is distributed over the day.
    The profile starts at 00:00.
    """

    size: int = field(default=N_HOUR_DAY, init=False)


@dataclass(eq=False)
class SpecificDayProfile(DailyProfile):
    """Same as DailyProfile but with a specific day of the week."""

    specific_day: SpecificDay | None = None

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.specific_day is None:
            raise ValueError(f"{self.specific_day=} must be defined.")

    def __getitem__(self, key: int) -> SpecificDayProfile:
        """Return the profile at the given index."""
        return SpecificDayProfile(self.ratios[key], specific_day=self.specific_day)

    # Defie the equality of the profiles
    def __eq__(self, other: Any) -> bool:
        super_result = super().__eq__(other)
        if isinstance(other, SpecificDayProfile):
            return super_result and (self.specific_day == other.specific_day)
        else:
            return super_result

    # Defined greater smaller by the size attribute
    def __lt__(self, other: AnyTimeProfile) -> bool:
        # sort by the specific day
        if isinstance(other, SpecificDayProfile):
            return self.specific_day < other.specific_day
        return super().__lt__(other)


@dataclass(eq=False)
class WeeklyProfile(TemporalProfile):
    """Weekly profile.

    Weekly profile defines how the emission is distributed over the week.
    The profile starts on Monday.
    """

    size: int = N_DAY_WEEK


@dataclass(eq=False)
class MounthsProfile(TemporalProfile):
    """Yearly profile.

    Yearly profile defines how the emission is distributed over the year using months.
    """

    size: int = N_MONTH_YEAR


@dataclass(eq=False)
class HourOfWeekProfile(TemporalProfile):
    """Hour of week profile.

    Hour of week profile defines how the emission is distributed over the week using hours.
    This is useful if you want to account for different daily patterns over the
    days of the week (usually for the weekend)

    The profile starts on Monday at 00:00.
    """

    size: int = N_HOUR_WEEK


@dataclass(eq=False)
class HourOfYearProfile(TemporalProfile):
    """Hour of year profile.

    Hour of year profile defines how the emission is distributed over the year using hours.
    """

    size: int = N_HOUR_YEAR


@dataclass(eq=False)
class Hour3OfDay(TemporalProfile):
    """Each 3 hour profile.

    Groups of 3 hours. (0-3, 3-6, 6-9, 9-12, 12-15, 15-18, 18-21, 21-24)
    """

    size: int = int(N_HOUR_DAY / 3)


@dataclass(eq=False)
class Hour3OfDayPerMonth(TemporalProfile):
    """Each 3 hour profile given for each mounth.

    Hour of year profile defines how the emission is distributed over the year using hours.
    """

    size: int = int(N_HOUR_DAY / 3) * N_MONTH_YEAR


@dataclass(eq=False)
class HourOfLeapYearProfile(TemporalProfile):
    """Hour of leap year profile.

    Hour of leap year profile defines how the emission is distributed over the year using hours.
    """

    size: int = N_HOUR_LEAPYEAR


@dataclass(eq=False)
class DayOfYearProfile(TemporalProfile):
    """Day of year profile.

    Day of year profile defines how the emission is distributed over the year using days.
    """

    size: int = N_DAY_YEAR


@dataclass(eq=False)
class DayOfLeapYearProfile(TemporalProfile):
    """Day of leap year profile.

    Day of leap year profile defines how the emission is distributed over the year using days.
    """

    size: int = N_DAY_LEAPYEAR


AnyTimeProfile = Union[
    DailyProfile,
    SpecificDayProfile,
    WeeklyProfile,
    MounthsProfile,
    HourOfYearProfile,
    HourOfLeapYearProfile,
]

# Maps temporal profiles to their corrected version
leap_year_corrected: dict[TemporalProfile, TemporalProfile] = {
    HourOfYearProfile: HourOfLeapYearProfile,
    DayOfYearProfile: DayOfLeapYearProfile,
}


def get_leap_year_or_normal(
    profile_type: type[TemporalProfile], year: int
) -> type[TemporalProfile]:
    """Return the profile type for the given year."""

    if year % 4 == 0:
        return leap_year_corrected.get(profile_type, profile_type)
    return profile_type


class AnyProfiles:
    """SAme a as temporal profiles, but can store any kind of proifles."""

    _profiles: list[AnyTimeProfile]

    def __init__(self, profiles: list[AnyTimeProfile] = None) -> None:
        if profiles is None:
            profiles = []
        self._profiles = profiles

    def __getitem__(self, key: int) -> AnyTimeProfile:
        for profile in self._profiles:
            if key < profile.n_profiles:
                return profile[key]
            key -= profile.n_profiles
        raise IndexError(f"{key=} is out of range.")

    def append(self, profile: AnyTimeProfile) -> None:
        self._profiles.append(profile)

    def __len__(self) -> int:
        return sum(len(profile) for profile in self._profiles)

    # Iteration must smoothly iterate over the profiles
    def __iter__(self):
        for profiles in self._profiles:
            yield from profiles

    @property
    def n_profiles(self) -> int:
        """Return the number of profiles."""
        return len(self)


class CompositeTemporalProfiles:
    """A helper class to handle mixtures of temporal profiles.

    Acts similar to a TemporalProfile

    Stores a dict for each type of profile,
    """

    _profiles: dict[
        type[AnyTimeProfile] | tuple[type[SpecificDayProfile], SpecificDay],
        AnyTimeProfile,
    ]
    # Store for each type, the indexes of the profiles
    _indexes: dict[
        type[AnyTimeProfile] | tuple[type[SpecificDayProfile], SpecificDay] | None,
        np.ndarray[int],
    ]

    def __init__(self, profiles: list[list[AnyTimeProfile]] = []) -> None:
        n = len(profiles)
        self._profiles = {}
        profiles_lists = {}
        self._indexes = {}
        # Get the unique types of profiles
        types = set(
            t if (t := type(p)) != SpecificDayProfile else (t, p.specific_day)
            for profiles_list in profiles
            for p in profiles_list
        )

        if len(types) == 0:
            # Empty profiles
            # only empty lists given
            self._indexes[None] = np.full(n, fill_value=-1, dtype=int)
            return

        # Allocate arrays
        for profile_type in types:
            if not isinstance(profile_type, tuple) and not issubclass(
                profile_type, TemporalProfile
            ):
                raise TypeError(
                    f"Profiles must be subclass of {TemporalProfile}. Not"
                    f" {profile_type=}"
                )
            profiles_lists[profile_type] = []
            self._indexes[profile_type] = np.full(n, fill_value=-1, dtype=int)
        # Construct the list and indexes based on the input
        for i, profiles_list in enumerate(profiles):
            if not isinstance(profiles_list, list):
                raise TypeError(
                    f"{profiles_list=} must be a list of {TemporalProfile}."
                )
            for profile in profiles_list:
                if profile.n_profiles != 1:
                    raise ValueError(
                        "Can only build CompositeTemporalProfiles from profiles with"
                        f" {profile.n_profiles=}, got {profile=}."
                    )
                p_type = type(profile)
                if p_type == SpecificDayProfile:
                    p_type = (p_type, profile.specific_day)
                list_this_type = profiles_lists[p_type]
                if self._indexes[p_type][i] != -1:
                    raise ValueError(
                        f"Cannot add {profile=} to {self=} as it was already added."
                    )
                self._indexes[p_type][i] = len(list_this_type)
                list_this_type.append(profile)
        # Convert the lists to arrays
        for profile_type, profiles_list in profiles_lists.items():
            ratios = np.concatenate([p.ratios for p in profiles_list])
            if isinstance(profile_type, tuple):
                profile = profile_type[0](
                    ratios=ratios,
                    specific_day=profile_type[1],
                )
            else:
                profile = profile_type(ratios=ratios)
            self._profiles[profile_type] = profile

    def __repr__(self) -> str:
        return f"CompositeProfiles({len(self)} profiles from {[t.__name__ for t in self.types]})"

    def __len__(self) -> int:
        indexes_len = [len(indexes) for indexes in self._indexes.values()]
        if not indexes_len:
            return 0
        # Make sure they are all equal
        if len(set(indexes_len)) != 1:
            raise ValueError(
                f"{self=} has different lengths of indexes for each profile type."
                f" {indexes_len=}"
            )
        return indexes_len[0]

    @property
    def n_profiles(self) -> int:
        """Return the number of profiles."""
        return len(self)

    def __getitem__(self, key: int) -> list[AnyTimeProfile]:
        return [
            self._profiles[p_type][index]
            for p_type, indexes in self._indexes.items()
            if p_type is not None and (index := indexes[key]) != -1
        ]

    def __setitem__(self, key: int, value: list[AnyTimeProfile]) -> None:
        self._array[key] = np.array(value, dtype=object)

    @property
    def types(self) -> list[AnyTimeProfile]:
        """Return the types of the profiles."""
        return list(set(self._profiles.keys()))

    @property
    def ratios(self) -> np.ndarray:
        """Return ratios of composite profiles.

        Idea is that we concatenate the ratio of each profile.
        nan values can be used when a profile is not defined for a given index.
        """
        return np.stack(
            [
                np.concatenate(
                    [
                        (
                            self._profiles[pt][index].ratios.reshape(-1)
                            if (index := self._indexes[pt][i]) != -1
                            else np.full(pt.size, np.nan).reshape(-1)
                        )
                        for pt in self.types
                    ]
                )
                for i in range(len(self))
            ],
            # axis=1,
        )

    @property
    def scaling_factors(self) -> np.ndarray:
        """Return the scaling factors of the profiles."""
        return np.stack(
            [
                np.concatenate(
                    [
                        (
                            self._profiles[pt][index].ratios.reshape(-1)
                            * self._profiles[pt][index].size
                            if (index := self._indexes[pt][i]) != -1
                            else np.ones(pt.size).reshape(-1)
                        )
                        for pt in self.types
                    ]
                )
                for i in range(len(self))
            ],
            # axis=1,
        )

    @classmethod
    def from_ratios(
        cls, ratios: np.ndarray, types: list[type], rescale: bool = False
    ) -> CompositeTemporalProfiles:
        """Create a composite profile, directly from the ratios.

        :arg ratios: The ratios of the profiles.
        :arg types: The types of the profiles, as a list of Temporal profiles types.
        :arg rescale: If True, the ratios will be rescaled to sum up to 1.

        """
        for t in types:
            # Check that the type is a subtype of TemporalProfile
            if not issubclass(t, TemporalProfile):
                raise TypeError(f"{t=} must be a {TemporalProfile}.")
        splitters = np.cumsum([0] + [t.size for t in types])
        logger.debug(f"{splitters=}")
        # Create the empty profiles
        profiles = [
            [
                t((r / r.sum(axis=0)) if rescale else r)
                for i, t in enumerate(types)
                if not np.any(
                    np.isnan(r := profile_ratios[splitters[i] : splitters[i + 1]])
                )
            ]
            for profile_ratios in ratios
        ]
        return cls(profiles)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, CompositeTemporalProfiles):
            raise TypeError(f"{__value=} must be a {CompositeTemporalProfiles}.")
        if len(self) != len(__value):
            return False
        return (self.ratios == __value.ratios).all()

    @classmethod
    def join(cls, *profiles: CompositeTemporalProfiles) -> CompositeTemporalProfiles:
        """Join multiple composite profiles."""
        # Get the types of profiles
        _profiles = {}
        types = set(sum((p.types for p in profiles), []))
        profile_lenghts = [len(p) for p in profiles]
        total_len = sum(profile_lenghts)
        _indexes = {t: np.full(total_len, fill_value=-1, dtype=int) for t in types}
        for t in types:
            _this_type_profiles = [p._profiles[t] for p in profiles if t in p.types]
            _this_type_n_profiles = [
                len(p._profiles[t]) if t in p.types else 0 for p in profiles
            ]
            _profiles[t] = concatenate_time_profiles(_this_type_profiles)

            # offset in the indexes indexes
            curr_index = 0
            # offset in the profile indexes
            curr_profile = 0
            for i, n in enumerate(_this_type_n_profiles):
                if n == 0:
                    curr_index += profile_lenghts[i]
                    continue
                indexes = profiles[i]._indexes[t].copy()
                mask_invalid = indexes == -1
                indexes[~mask_invalid] += curr_profile
                _indexes[t][curr_index : curr_index + len(indexes)] = indexes
                curr_index += profile_lenghts[i]
                curr_profile += n

        # Get the indexes
        obj = cls([])
        obj._profiles = _profiles
        obj._indexes = _indexes

        return obj

    def copy(self) -> CompositeTemporalProfiles:
        """Return a copy of the object."""
        return CompositeTemporalProfiles.join(self)

    # define the addition to be the same as if this was a list
    def __add__(self, other: CompositeTemporalProfiles) -> CompositeTemporalProfiles:
        return self.join(self, other)

    def __radd__(self, other: CompositeTemporalProfiles) -> CompositeTemporalProfiles:
        return self.join(other, self)

    def append(self, profiles_list: list[AnyTimeProfile]) -> None:
        """Append a profile list to this."""
        new_len = len(self) + 1
        original_types = self.types
        # expend all the indexes list
        for t in self._indexes.keys():
            self._indexes[t] = np.concatenate(
                (
                    self._indexes[t],
                    np.array([-1], dtype=int),
                )
            )

        for p in profiles_list:
            t = type(p)
            if t == SpecificDayProfile:
                t = (SpecificDayProfile, p.specific_day)
            if t not in self._indexes.keys():
                if isinstance(t, tuple):
                    self._profiles[t] = t[0](ratios=p.ratios, specific_day=t[1])
                else:
                    self._profiles[t] = t(ratios=p.ratios)
                self._indexes[t] = np.full(new_len, fill_value=-1, dtype=int)
                self._indexes[t][-1] = 0
            else:
                self._indexes[t][-1] = len(self._profiles[t])
                self._profiles[t].ratios = np.concatenate(
                    (self._profiles[t].ratios, p.ratios)
                )

    @property
    def size(self) -> int:
        """Return the size of the profiles."""
        return sum(p.size for p in self._profiles.values())

    def broadcast(self, types: list[TemporalProfile]) -> CompositeTemporalProfiles:
        """Create a new composite profile with the given types.

        The non specified profiles will be set to constant profiles.
        """
        all_ratios = []
        for t in types:
            # Get constant ratios
            composite_ratios = np.ones((len(self), t.size)) / t.size
            if t in self.types:
                ratios = self._profiles[t].ratios
                # Need to scale with the indexes
                indexes = self._indexes[t]
                # Fill the ratios with the existing profiles
                mask_valid = indexes != -1
                composite_ratios[mask_valid, :] = ratios[indexes[mask_valid], :]

            all_ratios.append(composite_ratios)

        return CompositeTemporalProfiles.from_ratios(
            np.concatenate(all_ratios, axis=1), types
        )


def make_composite_profiles(
    profiles: AnyProfiles,
    indexes: xr.DataArray,
) -> tuple[CompositeTemporalProfiles, xr.DataArray]:
    """Create a composite temporal profiles from a list of profiles and indexes.

    :arg profiles: The profiles to use.
    :arg indexes: The indexes to use.
        The indexes must have a dim called "profile" with the name of the profile type.

    """

    if not isinstance(profiles, AnyProfiles):
        raise TypeError(f"{profiles=} must be an {AnyProfiles}.")

    logger.debug(f"making composite profiles from {profiles=}, {indexes=}")

    if "profile" not in indexes.dims:
        raise ValueError(f"{indexes=} must have a dim called 'profile'.")
    # If size of the profiles is 1, then we can simply return the profiles
    if indexes.profile.size == 1:
        # It is only one type of profile
        return CompositeTemporalProfiles([[p] for p in profiles]), indexes.squeeze(
            "profile"
        )

    # Stack the arrays to keep only the profiles dimension and the new stacked dim
    dims = list(indexes.dims)
    dims.remove("profile")
    stacked = indexes.stack(z=dims)

    str_array = np.array(
        [
            str(array.values.reshape(-1))
            for lab, array in stacked.groupby(group="z", squeeze=False)
        ]
    )
    logger.debug(f"{str_array=}")
    u, inv = np.unique(str_array, return_inverse=True)

    extracted_profiles = [
        [
            profiles[i]
            # Unpakc the profiles from the str
            for i in np.fromstring(array_str[1:-1], sep=" ", dtype=int)
            if i != -1
        ]
        # Loop over each unique profile found
        for array_str in u
    ]
    logger.debug(f"{extracted_profiles=}")
    new_indexes = xr.DataArray(inv, dims=["z"], coords={"z": stacked.z})

    # Remove the z dimension from the profiles
    out_indexes = new_indexes.unstack("z")

    return CompositeTemporalProfiles(extracted_profiles), out_indexes


def get_index_in_profile(
    profile: Type[TemporalProfile], time_range: pd.DatetimeIndex
) -> pd.Series:
    """Get the index in the profile for each time in the time range.

    :param profile: the profile to use
    :param time_range: the time range to use
    :return: the index in the profile for each time in the time range
    """

    if profile == MounthsProfile:
        indexes = time_range.month - 1
    elif profile in [DayOfYearProfile, DayOfLeapYearProfile]:
        indexes = time_range.day_of_year - 1
    elif profile == DailyProfile:
        indexes = time_range.hour
    elif profile == WeeklyProfile:
        indexes = time_range.day_of_week
    elif profile in [HourOfYearProfile, HourOfLeapYearProfile]:
        indexes = time_range.hour + (time_range.day_of_year - 1) * 24
    elif profile == Hour3OfDayPerMonth:
        indexes = (time_range.hour // 3) + (time_range.month - 1) * 8
    else:
        raise ValueError(f"Profile type {profile} not recognized")

    assert indexes.min() >= 0, f"{profile=}, {time_range=}"
    assert indexes.max() < profile.size, f"{profile=}, {time_range=}"

    return indexes


def get_profile_da(
    profile: TemporalProfile, year: int, use_offset: bool = True
) -> xr.DataArray:
    """Return the profile as a data array.

    The index of the data array is exact timestamp at the middle of the period.
    """
    daterange_kwargs = {
        "start": f"{year}-01-01",
        "end": f"{year+1}-01-01",
        "inclusive": "both",
    }

    # The following will create correct timestamps at which the profile is true
    # An offset is also given, which is half the period

    if isinstance(profile, (DailyProfile, HourOfYearProfile, HourOfLeapYearProfile)):
        ts = pd.date_range(**daterange_kwargs, freq="h")
        offset = pd.Timedelta("30m")
    elif isinstance(profile, (WeeklyProfile, DayOfLeapYearProfile, DayOfYearProfile)):
        ts = pd.date_range(**daterange_kwargs, freq="d")
        offset = pd.Timedelta("12h")
    elif isinstance(profile, MounthsProfile):
        ts = pd.date_range(**daterange_kwargs, freq="MS")
        offset = pd.Timedelta("15d")
    elif isinstance(profile, Hour3OfDayPerMonth):
        ts = pd.date_range(**daterange_kwargs, freq="3h")
        offset = pd.Timedelta("1h30m")
    else:
        raise NotImplementedError(
            f"{type(profile)=} not implemented in `get_profile_da`."
        )

    # Add a first value at the begginning, such that we cover the whole year
    ts = pd.DatetimeIndex([ts[0] - 2 * offset] + list(ts))

    # Correct for non cyclic profiles (day of specific year)
    if isinstance(
        profile,
        (
            HourOfYearProfile,
            HourOfLeapYearProfile,
            DayOfLeapYearProfile,
            DayOfYearProfile,
        ),
    ):
        ts = ts[1:-1]

    da = xr.DataArray(
        profile.ratios[:, get_index_in_profile(type(profile), ts)],
        dims=["profile", "datetime"],
        coords={
            "profile": range(profile.n_profiles),
            "datetime": ts + offset if use_offset else ts,
        },
    )

    return da


def interpolate_profiles_hour_of_year(
    profiles: CompositeTemporalProfiles,
    year: int,
    interpolation_method: str = "linear",
    return_profiles: bool = False,
) -> (
    CompositeTemporalProfiles
    | xr.DataArray
    | tuple[CompositeTemporalProfiles | xr.DataArray, xr.DataArray]
):
    """Interpolate the profiles to create a hour of year profile."""

    serie = pd.date_range(
        f"{year}-01-01", f"{year+1}-01-01", freq="h", inclusive="left"
    )

    das_scaling_factors = []

    ratios = profiles.ratios

    offset = 0
    for t in profiles.types:
        # create an array with the ratios
        t_len = t.size
        this_ratios = ratios[:, offset : offset + t_len]
        this_ratios = np.nan_to_num(this_ratios, nan=1.0 / t_len)
        offset += t_len
        # Create a data array for these ratios and convert to scaling factors
        da_sf = get_profile_da(profile=t(this_ratios), year=year) * t_len
        # Interpolate the data array
        da_interp = da_sf.interp(
            datetime=serie,
            method=interpolation_method,
            assume_sorted=True,
        )

        das_scaling_factors.append(da_interp.expand_dims("profile_type"))

    # Multiply the data arrays
    da = xr.concat(das_scaling_factors, dim="profile_type").prod(dim="profile_type")
    da_ratios = da / da.sum(dim="datetime")
    # Create the profile
    if return_profiles:
        return CompositeTemporalProfiles.from_ratios(
            da_ratios.values, types=[get_leap_year_or_normal(HourOfYearProfile, year)]
        )

    return da_ratios


def resolve_daytype(
    profiles: CompositeTemporalProfiles, profiles_indexes: xr.Dataset, year: int
) -> tuple[CompositeTemporalProfiles, xr.Dataset]:

    time_range = pd.date_range(
        f"{year}-01-01", f"{year+1}-01-01", freq="h", inclusive="left"
    )

    # Few checks on the day types given
    assert (
        "day_type" in profiles_indexes.dims
    ), "The profiles indexes must have a 'day_type' dimension"
    day_types = profiles_indexes.day_type.values
    specific_days = [SpecificDay(day_type) for day_type in day_types]
    days_mapping = {day: get_days_as_ints(day) for day in specific_days}
    all_values = sum([days for days in days_mapping.values()], [])
    if sorted(all_values) != sorted(list(range(7))):
        raise ValueError(
            f"Invalid {day_types=}, must cover all days of the week but they cover {all_values}."
        )

    # Check that the profile given is correct
    expected_profile = get_leap_year_or_normal(HourOfYearProfile, year)
    if not isinstance(profiles, CompositeTemporalProfiles | expected_profile):
        raise TypeError(
            f"{profiles=} must be a {CompositeTemporalProfiles} or {expected_profile}."
        )
    if profiles.types != [expected_profile]:
        raise ValueError(
            f"{profiles=} must contain only {expected_profile} for the given {year=}."
        )

    # Get only the profiles which are not the same on the day types
    dims = list(profiles_indexes.dims)
    dims.remove("day_type")
    stacked_indexes = profiles_indexes.stack(ind=dims)
    mask_differ_over_daytype = ~(
        stacked_indexes == stacked_indexes.isel(day_type=0)
    ).all("day_type")
    require_merge_indexes = stacked_indexes.loc[{"ind": mask_differ_over_daytype}]

    if np.any(require_merge_indexes == -1):
        raise ValueError(
            f"Cannot resolve {profiles_indexes=} as some profiles are missing. \n"
            "Please fill them with constant profiles."
        )

    # Get the ratios of each datetime and time
    da_ratios = xr.DataArray(
        profiles.ratios[require_merge_indexes],
        dims=["day_type", "ind", "time"],
        coords={
            "day_type": require_merge_indexes["day_type"],
            "ind": require_merge_indexes["ind"],
            "time": time_range,
        },
    )

    # Create the output index array
    da_indexes_out = xr.zeros_like(
        mask_differ_over_daytype.loc[mask_differ_over_daytype], dtype=int
    )

    da_profiles_out = da_ratios.isel(day_type=0).drop_vars("day_type")

    for day_type, days in days_mapping.items():
        # Assign the values of each datetime to the correct day
        mask = da_ratios.time.dt.dayofweek.isin(days)
        da_profiles_out.loc[{"time": mask}] = da_ratios.sel(
            day_type=day_type.value
        ).loc[{"time": mask}]

    new_profiles, new_indexes = ratios_dataarray_to_profiles(
        da_profiles_out.rename({"time": "ratio"}).unstack(fill_value=-1)
    )

    new_indexes = new_indexes.stack(ind=dims)
    # Remove the missing
    new_indexes = new_indexes.loc[new_indexes != -1]
    # Set them to the correct value  (as we later append at the end the new profiles)
    new_indexes += profiles.n_profiles

    out_indexes = stacked_indexes.isel(day_type=0).drop_vars("day_type").copy(deep=True)
    out_indexes.loc[{"ind": new_indexes.ind}] = new_indexes

    # Need to rescale as now the ratio might not sum up to 1 exactly
    new_profiles /= new_profiles.sum(axis=1).reshape(-1, 1)

    out_profiles = CompositeTemporalProfiles.from_ratios(
        np.concatenate([profiles.ratios, new_profiles]), types=[expected_profile]
    )

    return out_profiles, out_indexes.unstack("ind", fill_value=-1)


def ensure_specific_days_consistency(
    profiles: list[AnyTimeProfile],
) -> list[AnyTimeProfile]:
    """Make sure that there is not confilct between specific days profiles and normal daily profiles.

    In case there is any conflict, this return a profile for each day of the week.
    """

    if not any(isinstance(p, SpecificDayProfile) for p in profiles):
        return profiles

    # Get the specific days profiles

    daily_profiles = [p for p in profiles if isinstance(p, DailyProfile)]
    non_daily_profiles = [p for p in profiles if not isinstance(p, DailyProfile)]

    weekdays_profiles = {
        SpecificDay.MONDAY: None,
        SpecificDay.TUESDAY: None,
        SpecificDay.WEDNESDAY: None,
        SpecificDay.THURSDAY: None,
        SpecificDay.FRIDAY: None,
        SpecificDay.SATURDAY: None,
        SpecificDay.SUNDAY: None,
    }
    # First assign what we have sepcific
    for p in daily_profiles:
        if isinstance(p, SpecificDayProfile):
            days = days_of_specific_day(p.specific_day)
            if len(days) == 1:
                # This day was concerned specifically
                weekdays_profiles[p.specific_day] = p
            else:
                # This was a weekend or weekday profile
                for day in days:
                    # Only override if not already defined
                    if weekdays_profiles[day] is None:
                        weekdays_profiles[day] = SpecificDayProfile(
                            specific_day=day, ratios=p.ratios
                        )

    general_daily_profiles = [p for p in daily_profiles if type(p) == DailyProfile]

    # Add a constant profile for missing day
    for day, profile in weekdays_profiles.items():
        if profile is not None:
            continue
        if len(general_daily_profiles) == 0:
            p = SpecificDayProfile(specific_day=day)

        elif len(general_daily_profiles) == 1:
            p = SpecificDayProfile(
                specific_day=day, ratios=general_daily_profiles[0].ratios
            )
        else:
            raise ValueError(
                f"Cannot assign {general_daily_profiles=} to {day=}, more than one"
                f" general {type(DailyProfile)} was given."
            )
        weekdays_profiles[day] = p

    return non_daily_profiles + list(weekdays_profiles.values())


def create_scaling_factors_time_serie(
    start_time: datetime,
    end_time: datetime,
    profiles: list[AnyTimeProfile],
    apply_month_interpolation: bool = True,
    freq: str = "h",
    inclusive: str = "both",
    local_tz: str | None = None,
) -> pd.Series:
    """Create a time serie of ratios for the requested time range.

    :arg start_time: The start time of the time serie.
    :arg end_time: The end time of the time serie.
    :arg profiles: The profiles to use to create .
    :arg apply_month_interpolation: If True, apply the month interpolation.
    :arg inclusive: {“both”, “neither”, “left”, “right”}, default “both”
        same as `pd.date_range <https://pandas.pydata.org/docs/reference/api/pandas.date_range.html#pandas-date-range>`_
        Include boundaries; Whether to set each bound as closed or open.
    """

    kwargs = {}
    if local_tz is not None:
        kwargs["tz"] = "UTC"
    # Create the time serie
    time_serie = pd.date_range(
        start_time, end_time, freq=freq, inclusive=inclusive, **kwargs
    )
    if local_tz is not None:
        time_serie = time_serie.tz_convert(local_tz)

    # Create the scaling factors
    scaling_factors = np.ones(len(time_serie))

    # Correct profiles list with specific day profiles
    profiles = ensure_specific_days_consistency(profiles)

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
    if len(profile) != 1:
        raise ValueError(
            f"Cannot apply {profile=} to a time serie, it must have only one profile."
        )

    factors = ratios_to_factors(profile.ratios.reshape(-1))

    # This will be the output
    scaling_factors = np.ones(len(time_serie))

    # Get the profile
    if isinstance(profile, SpecificDayProfile):
        # Find the days corresponding to this factor
        days_allowed = get_days_as_ints(profile.specific_day)
        if len(days_allowed) != 1:
            raise ValueError(
                f"Cannot apply {profile=} to a time serie, it must have only one day."
                "convert the time profiles with `ensure_specific_days_consistency`."
            )
        mask_matching_day = np.isin(time_serie.day_of_week, days_allowed)
        for hour, factor in enumerate(factors):
            # Other days will not have a scaling factor
            scaling_factors[(time_serie.hour == hour) & mask_matching_day] *= factor
    elif isinstance(profile, DailyProfile):
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
    else:
        raise NotImplementedError(
            f"Cannot apply {profile=}, {type(profile)=} is not implemented."
        )

    # Return the scaling factors
    return scaling_factors


_weekdays_long = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
_months_short = [
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
_months_long = [
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


timprofile_colnames = {
    WeeklyProfile: [
        _weekdays_short := [i[:3] for i in _weekdays_long],
        [i.lower() for i in _weekdays_short],
        _weekdays_long,
        [i.lower() for i in _weekdays_long],
        # TNO AVENGERS Format
        [i[:2] for i in _weekdays_short],
    ],
    MounthsProfile: [
        _months_short,
        [i.lower() for i in _months_short],
        _months_long,
        [i.lower() for i in _months_long],
    ],
    DailyProfile: [
        [str(i) for i in range(1, 25)],
        hours := [str(i) for i in range(24)],
        # TNO AVENGERS Format
        [f"H{i}" for i in hours],
    ],
}


def read_temporal_profiles(
    profiles_dir: PathLike,
    time_profiles_files_format: str = "timeprofiles*.csv",
    profile_csv_kwargs: dict[str, Any] = {},
    rtol: float = 1e-5,
) -> tuple[list[list[AnyTimeProfile]] | None, xr.DataArray | None]:
    """Read the temporal profiles csv files to the emiproc inventory format.

    The files for the time profiles are csv and must be all in the same directory
    named according to the argument `time_profiles_files_format`.

    Extra arguments depending on the file format can be passed to
    the function :py:func:`temporal_profiles.from_csv`
    that reads the csv files using: `profile_csv_kwargs`

    This returns the time profiles read, and and index xarray matching
    each substance and category to the desired profiles.

    If no files are found, this returns a warning.

    The format of the file will influence the name of the columns.
    Use the day of the weeks or the month names or the hour of the day to define the profile.

    :

    """

    # Note: The logic of this is a bit tricky because it has to handle
    #       the case where the profiles are speciated or not.

    profiles_dir = Path(profiles_dir)
    logger = logging.getLogger("emiproc.profiles.read_temporal_profiles")

    # List files with the expected format
    if profiles_dir.is_file():
        files = [profiles_dir]
        logger.info(f"File {profiles_dir=} found, will be used for timeprofiles.")
    else:
        if not profiles_dir.is_dir():
            raise ValueError(f"{profiles_dir=} is not a file or a directory.")
        files = list(profiles_dir.glob(time_profiles_files_format))
        if not files:
            logger.warning(
                "Cannot find any temporal profiles matching"
                f" {time_profiles_files_format=} in {profiles_dir=}.\n"
            )
            return None, None
        logger.info(
            f"Found {len(files)} files matching {time_profiles_files_format=} in"
            f" {profiles_dir=}"
        )

    out_profiles = AnyProfiles()
    indexes_list: list[xr.DataArray] = []
    for file in files:
        df = read_profile_file(file, **profile_csv_kwargs)
        possible_matching = {
            profile_type: colnames
            for profile_type, colnames_list in timprofile_colnames.items()
            for colnames in colnames_list
            if all(col in df.columns for col in colnames)
        }
        if not possible_matching:
            raise ValueError(
                f"Cannot find any matching time profile for {file=} with Columns"
                f" {df.columns}."
                "Please check the file format."
                "See more about time profiles file at "
                "https://emiproc.rtfd.io/en/latest/api.html#emiproc.profiles.temporal_profiles.read_temporal_profiles"
            )
        logger.info(f"{possible_matching=}")
        # Generate the profiles objects
        indexes = get_profiles_indexes(df)
        for profile_type, colnames in possible_matching.items():
            try:
                ratios = np.array([df[col] for col in colnames])
                if np.all(np.isclose(ratios.sum(axis=0), 1.0, rtol=rtol)):
                    # Ratios found
                    ratios = ratios
                elif np.all(np.isclose(np.mean(ratios, axis=0), 1.0, rtol=rtol)):
                    # Scaling factors found
                    ratios = ratios / ratios.sum(axis=0)
                else:
                    raise ValueError(
                        "Could not determine if scaling factors or ratios were given"
                        f" in {file=}.\n data:{ratios=} and \n"
                        f" mean:{np.mean(ratios, axis=0)} \n"
                        f" sum:{np.sum(ratios, axis=0)} \n Try to set {rtol=} to a"
                        " higher value if this is due to rounding erros."
                    )
                if isinstance(profile_type, tuple):
                    profiles = profile_type[0](
                        ratios=ratios.T, specific_day=profile_type[1]
                    )
                else:
                    profiles = profile_type(ratios.T)
            except Exception as e:
                raise ValueError(
                    f"Cannot create profile {profile_type=} from {file=} with {ratios=}"
                ) from e
            indexes += len(out_profiles)
            out_profiles.append(profiles)
            # Add a new dim which is the profile type
            indexes_list.append(
                indexes.expand_dims({"profile": [profile_type.__name__]})
            )

    combined_indexes = merge_indexes(indexes_list)

    composite_profiles, out_indexes = make_composite_profiles(
        out_profiles, combined_indexes
    )
    # Drop the profile dim
    if "profile" in out_indexes.dims:
        out_indexes = out_indexes.drop_vars("profile")

    return composite_profiles, out_indexes


@emiproc.deprecated
def from_csv(
    file: PathLike,
    profile_csv_kwargs: dict[str, Any] = {},
) -> dict[str | tuple[str, str] : AnyTimeProfile]:
    """Create the profile from a csv file.

    Based on the file, guess the correct profile
    to create.

    This is now deprectaed. Use :py:func:`read_temporal_profiles` instead.
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
                f"Cannot guess if {ratios=} are ratios or scaling factors for {row=} in"
                f" {file=}."
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
                f"Cannot add {key=} to {yaml_file=} as a {profile_class=} was already"
                " added."
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
