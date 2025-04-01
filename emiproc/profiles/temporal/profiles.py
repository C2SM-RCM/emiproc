from __future__ import annotations
from dataclasses import dataclass, field
import logging
from typing import Any, Union

import numpy as np

from emiproc.profiles.temporal.constants import (
    N_DAY_LEAPYEAR,
    N_DAY_WEEK,
    N_DAY_YEAR,
    N_HOUR_DAY,
    N_HOUR_LEAPYEAR,
    N_HOUR_WEEK,
    N_HOUR_YEAR,
    N_MONTH_YEAR,
)
from emiproc.profiles.temporal.specific_days import SpecificDay


logger = logging.getLogger(__name__)


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
    """Same a as temporal profiles, but can store any kind of proifles."""

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