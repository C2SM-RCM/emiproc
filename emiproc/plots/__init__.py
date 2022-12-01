from os import PathLike
from pathlib import Path
import geopandas as gpd
from typing import Any
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


from emiproc.plots import nclcmaps
from emiproc.inventories import Inventory
from emiproc.regrid import get_weights_mapping, weights_remap

mpl.style.use("default")


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
            )
        )
        return gdf.explore()
    elif category is not None and substance is None:

        gdf = gpd.GeoDataFrame(
            geometry=pd.concat(
                ([inv.geometry] if category in inv.gdf.columns else [])
                + ([inv.gdfs[category].geometry] if category in inv.gdfs else [])
            )
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
        )
        return gdf.explore(gdf[str((category, substance))])
    raise NotImplementedError()


def plot_inventory(
    inv: Inventory,
    figsize=(16, 9),
    q=0.001,
    cmap=nclcmaps.cmap("WhViBlGrYeOrRe"),
    spec_lims: None | tuple[float] = None,
    out_dir: PathLike | None = None,
    axis_formatter: str | None = None,
    x_label="lon [°]",
    y_label="lat [°]",
):
    """Plot an inventory.

    Will plot all the combination of substnaces/category of the inventory.
    Will also plot the total emission for each substance.

    :arg axis_formatter: for example "{x:6.0f}" will show 6 number and 0
        after the . , which is useful for swiss coordinates.
    """

    grid = inv.grid
    grid_shape = (grid.nx, grid.ny)
    x_min = grid.lon_range()[0]
    x_max = grid.lon_range()[-1]
    y_min = grid.lat_range()[0]
    y_max = grid.lat_range()[-1]

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

    for sub in inv.substances:
        total_sub_emissions = np.zeros(grid_shape).T
        total_emission_all_sectors = 0
        for cat in inv.categories:
            if (cat, sub) not in inv.gdf:
                # TODO: this will miss point sources for the total_sub_emissions
                # And also miss point sources for that category
                print(f"passsed {sub},{cat} no data")
                continue
            emissions = inv.gdf[(cat, sub)].to_numpy()
            if cat in inv.gdfs and sub in inv.gdfs[cat]:
                weights_file = out_dir / f".emiproc_weights_{inv.name}_gdfs_{cat}"
                weights_mapping = get_weights_mapping(
                    weights_file, inv.gdfs[cat], inv.gdf, loop_over_inv_objects=True
                )

                gdfs_emissions = weights_remap(
                    weights_mapping, inv.gdfs[cat][sub], len(emissions)
                )
                # TODO: could check and plot only the point sources as well
                emissions += gdfs_emissions

            total_emissions_per_sector = np.sum(emissions)
            total_emission_all_sectors += total_emissions_per_sector

            # from ha to m2
            emissions /= inv.cell_areas
            emissions = emissions.reshape(grid_shape).T[::-1, :]

            total_sub_emissions += emissions

            if not np.any(emissions):
                print(f"passsed {sub},{cat} no emissions")
                continue

            fig, ax = plt.subplots(
                figsize=figsize,
                # figsize=(38.40, 21.60),
                # gridspec_kw={"right": 0.9, "wspace": 0},
            )
            # cax = fig.add_axes([0.87, 0.15, 0.02, 0.7])

            emission_non_zero_values = emissions[emissions > 0]

            norm = LogNorm(
                vmin=np.quantile(emission_non_zero_values, q),
                vmax=np.quantile(emission_non_zero_values, 1 - q),
            )

            im = ax.imshow(
                emissions,
                norm=norm,
                cmap=cmap,
                extent=[x_min, x_max, y_min, y_max],
            )
            add_ax_info(ax)
            ax.set_title(f"{sub} - {cat}: " f"{total_emissions_per_sector:.2} " f"kg/y")
            fig.colorbar(
                im,
                label="kg/y/m2",
                extend="both",
                extendfrac=0.02
                # cax=cax,
            )

            fig.tight_layout()

            if out_dir:
                file_name = Path(out_dir) / f"raster_{sub}_{cat}"

                fig.savefig(file_name.with_suffix(".png"))
                # fig.savefig(file_name.with_suffix(".pdf"))
                fig.clear()
            else:
                plt.show()

        if not np.any(total_sub_emissions):
            print(f"passsed {sub},total_emissions, no emissions")
            continue

        fig, ax = plt.subplots(figsize=figsize)

        emission_non_zero_values = total_sub_emissions[total_sub_emissions > 0]
        norm = LogNorm(
            vmin=np.quantile(emission_non_zero_values, q),
            vmax=np.quantile(emission_non_zero_values, 1 - q),
        )

        im = ax.imshow(
            total_sub_emissions,
            norm=norm,
            cmap=cmap,
            extent=[x_min, x_max, y_min, y_max],
        )
        ax.set_title(f"Total {sub}: " f"{total_emission_all_sectors:.2} " f"kg/y")
        fig.colorbar(
            im,
            label="kg/y/m2",
            extend="both",
            extendfrac=0.02
            # cax=cax,
        )

        add_ax_info(ax)
        fig.tight_layout()

        if out_dir:
            file_name = Path(out_dir) / f"raster_total_{sub}"

            fig.savefig(file_name.with_suffix(".png"))
            # fig.savefig(file_name.with_suffix(".pdf"))
            fig.clear()
        else:
            plt.show()
