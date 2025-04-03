import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from emiproc.profiles.temporal.constants import N_HOUR_YEAR
from emiproc.profiles.temporal.profiles import (
    AnyTimeProfile,
    DailyProfile,
    HourOfYearProfile,
    HourOfLeapYearProfile,
    MounthsProfile,
    SpecificDayProfile,
    WeeklyProfile,
)
from emiproc.profiles.vertical_profiles import VerticalProfile


def get_x_axis(
    profile: AnyTimeProfile | VerticalProfile,
) -> tuple[str, list[float | str]]:
    match profile:
        case VerticalProfile():
            return "Height", np.concatenate([profile.height, profile.height[-1:]])
        case HourOfYearProfile() | HourOfLeapYearProfile():
            return "Hour of year", np.arange(profile.size + 1)
        case MounthsProfile():
            return "", [
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
                "",
            ]
        case WeeklyProfile():
            return "", [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
                "",
            ]
        case SpecificDayProfile():
            return f"Hour of day on {profile.specific_day.value}", np.arange(25)
        case DailyProfile():
            return "Hour of day", np.arange(25)
        case _:
            raise NotImplementedError(
                f"Cannot determine axis for {type(profile)}. Please implement."
            )


def plot_profile(
    profile: AnyTimeProfile | VerticalProfile,
    ax: plt.Axes | None = None,
    ignore_limit: bool = False,
    profile_number: int | None = None,
    labels: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a profile.

    Any type of profile is accepted.
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if profile.n_profiles > 10 and not ignore_limit and profile_number is None:
        raise ValueError(
            f"Too many profiles {profile.n_profiles}, cannot plot."
            " Set `ignore_limit` to True if you still want to plot."
        )

    # Get the x axis
    x_label, x_axis = get_x_axis(profile)

    # Add a duplicate in the end of the profile to close the loop
    profile_slice = (
        slice(None, None)
        if profile_number is None
        else slice(profile_number, profile_number + 1)
    )
    ratios = profile.ratios[profile_slice, :]
    profiles_to_plot = np.concatenate([ratios, ratios[:, -1:]], axis=1)

    # Plot by steps to show that the whole period is concerned
    ax.step(x_axis, profiles_to_plot.T, where="post")

    if labels:
        ax.set_xlabel(x_label)
        ax.set_ylabel("Ratio [-]")
        # Add rotation
        ax.tick_params(axis="x", rotation=45)
    else:
        # Hide the ticks
        ax.set_xticks([])

    ax.set_ylim(0, None)

    return fig, ax
