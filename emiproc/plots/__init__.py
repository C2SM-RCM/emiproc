"""Few plot functions for the emiproc package."""

from __future__ import annotations

import itertools
import logging
from os import PathLike
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, SymLogNorm

from emiproc.inventories import Inventory
from emiproc.plots import nclcmaps
from emiproc.regrid import get_weights_mapping, weights_remap
from emiproc.utilities import get_natural_earth


def explore_multilevel(gdf: gpd.GeoDataFrame, colum: Any, logscale: bool = False):
    """Explore a multilevel GeodataFrame.

    There is a bug with multilevel datframes that makes them impossible
    to call with the gpd.explore method.
    You can use this instead.
    """
    col_name = str(colum)
    # Deep copy to avoid changing anything
    data = gdf[colum].copy(deep=True)
    if logscale:
        data.loc[data == 0] = np.nan
        data = np.log(data)
    gdf_plot = gpd.GeoDataFrame({col_name: data}, geometry=gdf.geometry)
    return gdf_plot.explore(gdf[colum])


def explore_inventory(
    inv: Inventory, category: None | str = None, substance: None | str = None
):
    """Explore the emission of an inventory."""
    # First check if the data is available
    if inv.gdf is None:
        if category is not None and substance is not None:
            on_others_gdfs = category in inv.gdfs and substance in inv.gdfs[category]
            return inv.gdfs[category].explore()
        # TODO: implement some checks
        raise NotImplementedError()
    elif (
        category is not None
        and category not in inv.gdfs
        and category not in inv.gdf.columns
    ):
        raise IndexError(f"Category '{category}' not in inventory '{inv}'")
    elif (
        substance is not None
        and all((substance not in gdf for gdf in inv.gdfs))
        and substance not in inv.gdf.columns.swaplevel(0, 1)
    ):
        raise IndexError(f"Substance '{substance}' not in inventory '{inv}'")
    elif (
        substance is not None
        and category is not None
        and (category, substance) not in inv.gdf
        and (category not in inv.gdfs or substance not in inv.gdfs[category])
    ):
        raise IndexError(
            f"Substance '{substance}' for Category '{category}' not in inventory '{inv}'"
        )
    else:
        # Valid choice
        pass

    if category is None and substance is None:
        gdf = gpd.GeoDataFrame(
            geometry=pd.concat(
                [inv.geometry, *(gdf.geometry for gdf in inv.gdfs.values())]
            ),
            crs=inv.crs,
        )
        return gdf.explore()
    elif category is not None and substance is None:

        gdf = gpd.GeoDataFrame(
            geometry=pd.concat(
                ([inv.geometry] if category in inv.gdf.columns else [])
                + ([inv.gdfs[category].geometry] if category in inv.gdfs else [])
            ),
            crs=inv.crs,
        )
        return gdf.explore()
    elif category is not None and substance is not None:
        on_main_grid = (category, substance) in inv.gdf.columns
        on_others_gdfs = category in inv.gdfs and substance in inv.gdfs[category]
        gdf = gpd.GeoDataFrame(
            {
                str((category, substance)): pd.concat(
                    ([inv.gdf[(category, substance)]] if on_main_grid else [])
                    + ([inv.gdfs[category][substance]] if on_others_gdfs else [])
                )
            },
            geometry=pd.concat(
                ([inv.geometry] if on_main_grid else [])
                + ([inv.gdfs[category].geometry] if on_others_gdfs else [])
            ),
            crs=inv.crs,
        )
        return gdf.explore(gdf[str((category, substance))])
    else:
        raise NotImplementedError()


def plot_inventory(
    inv: Inventory,
    figsize=(16, 9),
    q=0.001,
    vmin: None | float = None,
    vmax: None | float = None,
    cmap=nclcmaps.cmap("WhViBlGrYeOrRe"),
    symcmap="RdBu_r",
    spec_lims: None | tuple[float] = None,
    out_dir: PathLike | None = None,
    axis_formatter: str | None = None,
    x_label="lon [°]",
    y_label="lat [°]",
    add_country_borders: bool = False,
    total_only: bool = False,
    reverse_y: bool = False,
):
    """Plot an inventory.

    Will plot all the combination of substnaces/category of the inventory.
    Will also plot the total emission for each substance.

    :arg axis_formatter: for example "{x:6.0f}" will show 6 number and 0
        after the . , which is useful for swiss coordinates.
    """

    logger = logging.getLogger(__name__)

    grid = inv.grid
    grid_shape = (grid.nx, grid.ny)

    def get_vmax(data: np.ndarray) -> float:
        if vmax is not None:
            return vmax
        return np.quantile(data, 1 - q)

    def get_vmin(data: np.ndarray) -> float:
        if vmin is not None:
            return vmin
        return np.quantile(data, q)

    def get_norm_and_cmap(data: np.ndarray) -> tuple[mpl.colors.Normalize, str]:
        if np.any(data < 0):
            abs_values = np.abs(data)
            vmax_ = get_vmax(abs_values)
            vmin_ = get_vmin(abs_values)
            # Use symlog instead
            return SymLogNorm(linthresh=vmin_, vmin=-vmax_, vmax=vmax_), symcmap
        else:
            return LogNorm(vmin=get_vmin(data), vmax=get_vmax(data)), cmap

    if len(inv.categories) == 1:
        logger.info("Only one category, will plot only the total emissions")
        total_only = True

    lon_range = grid.lon_range if hasattr(grid, "lon_range") else np.arange(grid.nx)
    lat_range = grid.lat_range if hasattr(grid, "lat_range") else np.arange(grid.ny)

    x_min, x_max = lon_range[0], lon_range[-1]
    y_min, y_max = lat_range[0], lat_range[-1]

    if add_country_borders and (
        not hasattr(grid, "lat_range") or hasattr(grid, "lon_range")
    ):
        raise ValueError(
            "Cannot add country borders without grid lat_range and lon_range"
        )
    elif add_country_borders:
        gdf_countries = get_natural_earth(
            resolution="10m", category="cultural", name="admin_0_countries"
        )
        # Crop the countries to the grid
        gdf_countries = gdf_countries.cx[x_min:x_max, y_min:y_max].clip_by_rect(
            x_min, y_min, x_max, y_max
        )

        def add_country_borders(ax: mpl.axes.Axes):
            gdf_countries.boundary.plot(ax=ax, color="black", linewidth=0.5)

    else:
        add_country_borders = lambda ax: None

    def add_ax_info(ax: mpl.axes.Axes):
        if axis_formatter is not None:
            ax.xaxis.set_major_formatter(axis_formatter)
            ax.yaxis.set_major_formatter(axis_formatter)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if spec_lims:
            ax.set_xlim(spec_lims[0], spec_lims[1])
            ax.set_ylim(spec_lims[2], spec_lims[3])

    if out_dir is not None:
        plt.ioff()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    per_substances_per_sector_emissions = {}
    for sub in inv.substances:
        per_sector_emissions = {}
        per_substances_per_sector_emissions[sub] = per_sector_emissions
        total_sub_emissions = np.zeros(grid_shape).T
        for cat in inv.categories:
            if (cat, sub) not in inv.gdf:
                # TODO: this will miss point sources for the total_sub_emissions
                # And also miss point sources for that category
                print(f"passsed {sub},{cat} no data")
                continue
            emissions = inv.gdf[(cat, sub)].copy(deep=True).to_numpy()
            if cat in inv.gdfs and sub in inv.gdfs[cat]:
                if out_dir:
                    weights_file = out_dir / f".emiproc_weights_{inv.name}_gdfs_{cat}"
                else:
                    weights_file = None
                weights_mapping = get_weights_mapping(
                    weights_file, inv.gdfs[cat], inv.gdf, loop_over_inv_objects=True
                )

                gdfs_emissions = weights_remap(
                    weights_mapping, inv.gdfs[cat][sub], len(emissions)
                )
                # TODO: could check and plot only the point sources as well
                emissions += gdfs_emissions

            per_sector_emissions[cat] = np.sum(emissions)

            # from ha to m2
            emissions /= inv.cell_areas

            y_slice = slice(None, None, 1 if reverse_y else -1)
            emissions = emissions.reshape(grid_shape).T[y_slice, :]

            total_sub_emissions += emissions

            if not np.any(emissions):
                print(f"passsed {sub},{cat} no emissions")
                continue

            if total_only:
                continue

            fig, ax = plt.subplots(
                figsize=figsize,
                # gridspec_kw={"right": 0.9, "wspace": 0},
            )

            # Check if there are values below 0
            emission_non_zero_values = emissions[
                (emissions != 0) & (~np.isnan(emissions))
            ]
            if len(emission_non_zero_values) == 0:
                print(f"passsed {sub},{cat} no emissions")
                continue

            norm, this_cmap = get_norm_and_cmap(emission_non_zero_values)

            im = ax.imshow(
                emissions,
                norm=norm,
                cmap=this_cmap,
                extent=[x_min, x_max, y_min, y_max],
            )
            add_ax_info(ax)
            ax.set_title(f"{sub} - {cat}: " f"{per_sector_emissions[cat]:.2} " f"kg/y")
            fig.colorbar(
                im,
                label="kg/y/m2",
                extend="both",
                extendfrac=0.02,
                # cax=cax,
            )

            add_country_borders(ax)

            fig.tight_layout()

            if out_dir:
                file_name = Path(out_dir) / f"raster_{sub}_{cat}"

                fig.savefig(file_name.with_suffix(".png"))
                # fig.savefig(file_name.with_suffix(".pdf"))
                fig.clear()
            else:
                plt.show()

            plt.close(fig)

        if not np.any(total_sub_emissions):
            print(f"passsed {sub},total_emissions, no emissions")
            continue

        fig, ax = plt.subplots(figsize=figsize)

        emission_non_zero_values = total_sub_emissions[
            (total_sub_emissions > 0) & (~np.isnan(total_sub_emissions))
        ]
        if len(emission_non_zero_values) == 0:
            print(f"passsed {sub},total_emissions, no emissions")
            continue

        norm, this_cmap = get_norm_and_cmap(emission_non_zero_values)

        im = ax.imshow(
            total_sub_emissions,
            norm=norm,
            cmap=this_cmap,
            extent=[x_min, x_max, y_min, y_max],
        )
        ax.set_title(
            f"Total {sub}: " f"{sum(per_sector_emissions.values()):.2} " f"kg/y"
        )
        fig.colorbar(
            im,
            label="kg/y/m2",
            extend="both",
            extendfrac=0.02,
            # cax=cax,
        )

        add_ax_info(ax)
        add_country_borders(ax)
        fig.tight_layout()

        if out_dir:
            file_name = Path(out_dir) / f"raster_total_{sub}"

            fig.savefig(file_name.with_suffix(".png"))
            # fig.savefig(file_name.with_suffix(".pdf"))
            fig.clear()
        else:
            plt.show()

        plt.close(fig)

    # A bar plot of the total emissions for each substances and each category
    sorted_categories = sorted(inv.categories)
    n_substances = len(inv.substances)
    fig, axes = plt.subplots(
        figsize=(len(inv.categories) * 0.5, n_substances),
        nrows=n_substances,
        sharex=True,
    )

    color_iter = itertools.cycle(plt.rcParams["axes.prop_cycle"])
    colors_of_categories = {cat: next(color_iter)["color"] for cat in sorted_categories}
    for i, sub in enumerate(inv.substances):
        if n_substances > 1:
            ax = axes[i]
        else:
            ax = axes
        for j, cat in enumerate(sorted_categories):
            ax.bar(
                j,
                per_substances_per_sector_emissions[sub].get(cat, 0),
                color=colors_of_categories.get(cat, "black"),
            )
        ax.set_ylabel(f"{sub} [kg/y]")

    # Add ticks on the last ax (the one at the bottom)
    ax.set_xticks(range(len(sorted_categories)))
    ax.set_xticklabels(sorted_categories, rotation=45, ha="right")

    if out_dir:
        file_name = Path(out_dir) / f"barplot_total_emissions"
        fig.savefig(file_name.with_suffix(".png"))
        fig.clear()
    else:
        plt.show()

    plt.close(fig)
