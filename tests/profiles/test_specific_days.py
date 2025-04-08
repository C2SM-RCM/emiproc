"""Test for the specific days profile."""
from __future__ import annotations


import pytest
from emiproc.profiles.temporal.specific_days import SpecificDay, days_of_specific_day
"""Test for the specific days profile."""



def test_specific_day_from_day_number():
    assert SpecificDay.from_day_number(0) == SpecificDay.MONDAY
    assert SpecificDay.from_day_number(1) == SpecificDay.TUESDAY
    assert SpecificDay.from_day_number(2) == SpecificDay.WEDNESDAY
    assert SpecificDay.from_day_number(3) == SpecificDay.THURSDAY
    assert SpecificDay.from_day_number(4) == SpecificDay.FRIDAY
    assert SpecificDay.from_day_number(5) == SpecificDay.SATURDAY
    assert SpecificDay.from_day_number(6) == SpecificDay.SUNDAY

    with pytest.raises(ValueError):
        SpecificDay.from_day_number(7)

    with pytest.raises(TypeError):
        SpecificDay.from_day_number("Monday")


def test_specific_day_comparison():
    assert SpecificDay.MONDAY < SpecificDay.TUESDAY
    assert SpecificDay.FRIDAY < SpecificDay.SATURDAY
    assert not (SpecificDay.SUNDAY < SpecificDay.SATURDAY)

    with pytest.raises(TypeError):
        SpecificDay.MONDAY < "Tuesday"


@pytest.mark.parametrize(
    "specific_day, expected_days",
    [
        (SpecificDay.MONDAY, [SpecificDay.MONDAY]),
        (
            SpecificDay.WEEKDAY,
            [
                SpecificDay.MONDAY,
                SpecificDay.TUESDAY,
                SpecificDay.WEDNESDAY,
                SpecificDay.THURSDAY,
                SpecificDay.FRIDAY,
            ],
        ),
        (
            SpecificDay.WEEKEND,
            [
                SpecificDay.SATURDAY,
                SpecificDay.SUNDAY,
            ],
        ),
        (
            SpecificDay.WEEKDAY_4,
            [
                SpecificDay.MONDAY,
                SpecificDay.TUESDAY,
                SpecificDay.WEDNESDAY,
                SpecificDay.THURSDAY,
            ],
        ),
    ],
)
def test_days_of_specific_day(specific_day, expected_days):
    assert days_of_specific_day(specific_day) == expected_days


def test_days_value():
    assert SpecificDay.MONDAY.value == "monday"
    assert SpecificDay.TUESDAY.value == "tuesday"
    assert SpecificDay.WEEKDAY.value == "weekday"
