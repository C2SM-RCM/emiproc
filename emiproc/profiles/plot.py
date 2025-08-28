from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from emiproc.profiles.temporal.composite import (
    CompositeTemporalProfiles,
    split_composite_profile,
)
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
    if isinstance(profile, VerticalProfile):
        return "Height", np.concatenate([profile.height, profile.height[-1:]])
    elif isinstance(profile, (HourOfYearProfile, HourOfLeapYearProfile)):
        return "Hour of year", np.arange(profile.size + 1)
    elif isinstance(profile, MounthsProfile):
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
    elif isinstance(profile, WeeklyProfile):
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
    elif isinstance(profile, SpecificDayProfile):
        return f"Hour of day on {profile.specific_day.value}", np.arange(25)
    elif isinstance(profile, DailyProfile):
        return "Hour of day", np.arange(25)
    else:
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

    :arg profile: The profile to plot. Can be any kind of profiles supported in emiproc.
    :arg ax: The axes to plot on.
    :arg ignore_limit: Whether to ignore the profile limit, for plotting a 
        lot of profiles.
    :arg profile_number: The profile number to plot. In case you want to plot only
        one of the profiles in the input.
    :arg labels: Whether to add x and y labels.
    """

    if isinstance(profile, CompositeTemporalProfiles):
        profile = split_composite_profile(profile)

    if isinstance(profile, list):
        if len(profile) == 0:
            raise ValueError("Cannot pass an empty list of profiles.")
        elif len(profile) == 1:
            profile = profile[0]
        else:
            # Plot all profiles in the list
            return _plot_profiles(
                profile,
                ignore_limit=ignore_limit,
                labels=labels,
                profile_number=profile_number,
                share_ax=True if ax else False,
                ax=ax,
            )

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
    ax.step(x_axis, profiles_to_plot.T, where="post", label=profile.label)

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


def _plot_profiles(
    profiles: list[VerticalProfile | AnyTimeProfile],
    ignore_limit: bool = False,
    labels: bool = True,
    profile_number: int | None = None,
    share_ax: bool = False,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a list of profiles.

    Same as `plot_profile`, but for a list of profiles.
    Hidden so that one can use it in the main function.
    """

    assert len(profiles) > 1, "Should not be called for a single profile."

    if ax is None:
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(profiles) if not share_ax else 1,
            figsize=(5 * len(profiles), 5),
            sharey=True,
            squeeze=True,
        )
    else:
        fig = ax.get_figure()
        axes = ax
        share_ax = True

    if share_ax:
        axes = np.array([axes] * len(profiles))

    for i, profile in enumerate(profiles):
        ax = axes[i]

        plot_profile(
            profile,
            ax=ax,
            ignore_limit=ignore_limit,
            profile_number=profile_number,
            labels=labels,
        )
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2)
    fig.suptitle("Profiles", fontsize=16, y=1.05)

    # Set the ylim after ward
    max_y = max(np.max(p.ratios) for p in profiles) * 1.1
    min_y = min(np.min(p.ratios) for p in profiles) * 0.9

    axes[0].set_ylim(min_y, max_y)

    return fig, axes if not share_ax else ax
