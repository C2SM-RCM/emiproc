"""Plotting functions for VPRM profiles."""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from emiproc.profiles.vprm import VPRM_Model, urban_vprm_models


def plot_vprm_params_per_veg_type(
    df: pd.DataFrame,
    df_vprm: pd.DataFrame,
    veg_types: list[str] | None = None,
    model: VPRM_Model | str = VPRM_Model.standard,
    plots: list[str] = ["meteo", "indices", "emissions", "scaling"],
    group_by: str | None = None,
):
    """Plot the VPRM parameters per vegetation type.

    Each column is a vegetation type, and each row is a different plot.

    1. Temperature and radiation
    2. Vegetation indices
    3. Emissions
    4. Scaling parameters (Tscale, Wscale, Pscale) 

    :param df: Dataframe with the observations.
        Output from :py:func:`~emiproc.profiles.vprm.calculate_vprm_emissions`.
    :param df_vprm: Dataframe with the VPRM parameters per vegetation type.
        Same input as the `df_vprm` parameter in
        :py:func:`~emiproc.profiles.vprm.calculate_vprm_emissions`.
    :param veg_types: List of vegetation types to plot.
        If None, all vegetation types in the dataframe will be plotted.
    :param model: VPRM model to use. This is used to determine which parameters to plot.
    :param plots: List of plots to create. 
    :param group_by: If provided, the dataframe will be grouped
        by this temporal frequency before plotting.
        e.g. "%m%H" to get daily profiles for each month.
    """

    model = VPRM_Model(model)

    if veg_types is None:
        veg_types_indices = df_vprm.index.tolist()
        veg_type_timeseries = df.columns.get_level_values(0).unique().tolist()
        veg_types = list(set(veg_types_indices) & set(veg_type_timeseries))
        if not veg_types:
            raise ValueError(
                "No vegetation types found in the data. "
                "Please provide a list of vegetation types."
            )

    fig, axes_grid = plt.subplots(
        nrows=len(plots),
        ncols=len(veg_types),
        sharex=True,
        sharey=False,
        gridspec_kw={"hspace": 0.1, "wspace": 0.01},
        figsize=(5 * len(veg_types), 3 * len(plots)),
    )

    if group_by is not None:
        df["group"] = df.index.strftime(group_by)
        df = df.set_index("group", append=True)
        # Only aggregate numeric columns for performance and avoid dtype issues
        df_grouped = df.groupby(level="group", sort=False).mean(numeric_only=True)
        # Use the group as the new index for plotting
        df_grouped.index = df_grouped.index.get_level_values("group")
        df = df_grouped

        x = np.arange(len(df.index))
    else:
        x = df.index

    y_labels = {
        "meteo": "Temperature [C]",
        "indices": "Vegetation indices",
        "emissions": "Emissions [umoles/m2/s]",
        "scaling": "Scaling parameters",
    }

    for vegetation_type, axes in zip(veg_types, axes_grid.T):
        is_left_col = vegetation_type == veg_types[0]
        is_right_col = vegetation_type == veg_types[-1]

        axes_iter = iter(axes)
        axes_dict = {}
        if "meteo" in plots:

            ax_T = next(axes_iter)
            axes_dict["meteo"] = ax_T
            ax_T.set_title(vegetation_type)
            l_tg = ax_T.plot(
                x,
                df[("T", "global")],
                label="T global  averaged",
                alpha=0.5,
            )
            if model in urban_vprm_models:
                l_tu = ax_T.plot(
                    x,
                    df[("T", "urban")],
                    label="T urban  averaged",
                    alpha=0.5,
                )

            vprm_params = [
                "Tmin",
                "Tmax",
                "Topt",
                "Tlow",
            ]

            ax_RAD = ax_T.twinx()
            l_r = ax_RAD.plot(
                x,
                df[("RAD", "global")],
                label="Radiation",
                color="orange",
                alpha=0.5,
            )
            if is_right_col:
                ax_RAD.set_ylabel("Radiation [W/m2]")
                lines = l_tg + l_r
                if model in urban_vprm_models:
                    lines += l_tu
                ax_RAD.legend(lines, [l.get_label() for l in lines])

            # Make horizontal lines for the parameters
            ax_T.hlines(
                df_vprm.loc[vegetation_type, vprm_params],
                x[0],
                x[-1],
                color="k",
                linestyles="dashed",
            )
            for param in vprm_params:
                # Add text for the parameters
                ax_T.text(
                    x[0],
                    df_vprm.loc[vegetation_type, param],
                    param,
                    color="k",
                    fontsize=8,
                )
            if not is_right_col:
                ax_RAD.set_yticks([], [])

        # Plot the vegetation indices
        if "indices" in plots:
            ax_inds = next(axes_iter)
            axes_dict["indices"] = ax_inds
            indices = {
                "evi": "green",
                "ndvi": "orange",
                "lswi": "red",
                "evi_ref": "blue",
            }
            for index, color in indices.items():
                if index == "evi_ref" and model not in urban_vprm_models:
                    continue
                if (vegetation_type, index) not in df.columns:
                    continue
                ax_inds.plot(
                    x,
                    df[(vegetation_type, index)],
                    label=index,
                    color=color,
                    alpha=0.8,
                )
                if group_by is None:
                    mask = df[(vegetation_type, index + "_mask")]
                    ax_inds.scatter(
                        x[mask],
                        df.loc[mask, (vegetation_type, index + "_extracted")],
                        color=color,
                        marker="x",
                    )
            evi_ref_col = (vegetation_type, "evi_ref")
            if model in urban_vprm_models and evi_ref_col in df.columns:
                min_evi_ref = min(df[evi_ref_col])
                ax_inds.hlines(
                    min_evi_ref,
                    x[0],
                    x[-1],
                    color="blue",
                    linestyles="dashed",
                    alpha=0.5,
                    label="minimum evi_ref",
                )

            ax_inds.set_ylim(-0.5, 1.1)

        # plot the emissions
        if "emissions" in plots:
            ax_emi = next(axes_iter)
            axes_dict["emissions"] = ax_emi

            ax_emi.plot(x, df[(vegetation_type, "resp")], label="resp", alpha=0.7)
            ax_emi.plot(x, df[(vegetation_type, "gee")], label="gee", alpha=0.7)
            ax_emi.plot(x, df[(vegetation_type, "nee")], label="nee", alpha=0.7)
            # Horizontal line at 0
            ax_emi.plot([x[0], x[-1]], [0, 0], "k")

        # Plot the scaling params

        if "scaling" in plots:
            ax_scale = next(axes_iter)
            axes_dict["scaling"] = ax_scale

            for param in ["Tscale", "Wscale", "Pscale"]:
                ax_scale.plot(x, df[(vegetation_type, param)], label=param)

        if is_left_col:
            for title, ax in axes_dict.items():
                ax.set_ylabel(y_labels[title])

        if is_right_col:
            for title, ax in axes_dict.items():
                if title == "meteo":
                    continue
                ax.legend()

        if not is_left_col:

            for ax in axes:
                # Hide y ticks
                ax.set_yticks([], [])

        # Set x ticks rotation
        axes[-1].tick_params(
            "x",
            # rotation=45,
            direction="out",
        )

    if group_by is not None:

        # Remove the highest frequency from the group by
        months = [
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

        group_by_clean = group_by.replace(":", "").replace("-", "").replace(" ", "")
        ticks = {
            "%m%H": (np.arange(0, 12 * 24, 24), months),
            "%m%d": (np.arange(0, 12 * 31, 31), months),
        }.get(group_by_clean, None)
        if ticks is None:
            raise ValueError(
                f"Cannot set x ticks for cleaned group_by: {group_by_clean}"
            )

        axes_grid[-1, 0].set_xticks(*ticks)

    fig.align_ylabels(axes_grid[:, 0])
