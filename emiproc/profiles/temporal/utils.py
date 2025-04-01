import numpy as np
from emiproc.profiles.temporal.profiles import (
    AnyTimeProfile,
    DailyProfile,
    SpecificDayProfile,
)
from emiproc.profiles.temporal.specific_days import SpecificDay, days_of_specific_day


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
