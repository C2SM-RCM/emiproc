from __future__ import annotations
from enum import Enum, auto


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

    # Only the 4 first weekdays 
    WEEKDAY_4 = auto()

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
    """Return the days corresponding for a specific day.
    
    This agrees with the pandas convention where Monday is 0 and Sunday is 6.
    """

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
    elif specific_day == SpecificDay.WEEKDAY_4:
        return [0, 1, 2, 3]
    elif specific_day == SpecificDay.WEEKEND:
        return [5, 6]
    else:
        raise NotImplementedError(
            f"{specific_day=} is implemented in  {get_days_as_ints}"
        )

