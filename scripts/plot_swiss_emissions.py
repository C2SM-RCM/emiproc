# %% Select the path with my data
from pathlib import Path

from emiproc.inventories.swiss import SwissRasters
from emiproc.regrid import get_weights_mapping, remap_inventory, weights_remap

# Whether to plot only the place over zurich
OVER_ZH = True
year = 2020
data_path = Path(r"C:\Users\coli\Documents\ZH-CH-emission\Data\CHEmissionen")
plots_path = Path(r"C:\Users\coli\Pictures") / ( "ch_inv_over_zh" if OVER_ZH else "ch_rasters") / str(year)
weights_path = Path(".emiproc_weights_swiss_2_icon")
weights_path.mkdir(parents=True, exist_ok=True)
plots_path.mkdir(parents=True, exist_ok=True)


# %% Create the inventory object
inv_ch = SwissRasters(
    data_path=data_path,
    rasters_dir=data_path / "ekat_gridascii",
    rasters_str_dir=data_path / "ekat_str_gridascii",
    requires_grid=False,
    year=year,
)
inv_ch.gdf

#%%
import matplotlib.pyplot as plt
from emiproc.plots import nclcmaps

cmap = nclcmaps.cmap("WhViBlGrYeOrRe")
import matplotlib as mpl

mpl.style.use("default")
from matplotlib.colors import LogNorm
import numpy as np

plt.ioff()

# %%
grid = inv_ch.grid

x_min = grid.lon_range()[0]
x_max = grid.lon_range()[-1]
y_min = grid.lat_range()[0]
y_max = grid.lat_range()[-1]

# %% plot all susbstances and categories
for substance in inv_ch.substances:
    for cat in inv_ch.categories:
        if (cat, substance) not in inv_ch.gdf:
            continue
        emissions = inv_ch.gdf[(cat, substance)].to_numpy()
        emissions = emissions.reshape((grid.ny, grid.nx))
        total_emissions_per_sector = np.sum(emissions)

        if not np.any(emissions):
            print(f"passsed {substance},{cat} no emissions")
            continue

        fig, ax = plt.subplots(
            figsize=(16, 9),
            # figsize=(38.40, 21.60),
            # gridspec_kw={"right": 0.9, "wspace": 0},
        )
        # cax = fig.add_axes([0.87, 0.15, 0.02, 0.7])

        emission_non_zero_values = emissions[emissions > 0]
        #q = 0.005
        q = 0.001
        norm = LogNorm(
            vmin=np.quantile(emission_non_zero_values, q),
            vmax=np.quantile(emission_non_zero_values, 1 - q),
        )

        im = ax.imshow(
            emissions,
            norm=norm,
            cmap=cmap,
            # cmap="inferno",
            # extent=[wgs_84_extents[1],wgs_84_extents[3], wgs_84_extents[0],wgs_84_extents[2] ]
            extent=[x_min, x_max, y_min, y_max],
        )
        ax.xaxis.set_major_formatter("{x:6.0f}")
        ax.yaxis.set_major_formatter("{x:6.0f}")
        ax.set_title(
            f"{substance} - {cat}: " f"{total_emissions_per_sector:.2} " f"kg/y"
        )
        fig.colorbar(
            im,
            label="kg/y",
            extend="both",
            extendfrac=0.02
            # cax=cax,
        )
        if OVER_ZH:
            ax.set_xlim(2675000,2690000)
            ax.set_ylim(1242000, 1255000)
        fig.tight_layout()
        file_name = plots_path / f"raster_{substance}_{cat}"
        fig.savefig(file_name.with_suffix(".png"))
        # fig.savefig(file_name.with_suffix(".pdf"))
        fig.clear()
# %% sum all categories
for substance in inv_ch.substances:

    emissions = sum(
        [
            inv_ch.gdf[(cat, substance)].to_numpy()
            for cat in inv_ch.categories
            if (cat, substance) in inv_ch.gdf
        ]
    )
    # Add the point sources 
    w_mapping = get_weights_mapping(
        data_path / 'weights_eipwp', inv_ch.gdfs['eipwp'].geometry, inv_ch.gdf.geometry, loop_over_inv_objects=True
    )
    emissions += weights_remap(w_mapping, inv_ch.gdfs['eipwp'][substance], len(inv_ch.gdf))
    

    total_emissions = np.sum(emissions)
    emissions = emissions.reshape((grid.ny, grid.nx))

    # from ha to m2
    emissions /=  10000

    if not np.any(emissions):
        print(f"passsed {substance},{cat} no emissions")
        continue

    fig, ax = plt.subplots(
        figsize=(16, 9),
        # figsize=(38.40, 21.60),
        # gridspec_kw={"right": 0.9, "wspace": 0},
    )
    # cax = fig.add_axes([0.87, 0.15, 0.02, 0.7])

    emission_non_zero_values = emissions[emissions > 0]
    q = 0.001
    norm = LogNorm(
        vmin=np.quantile(emission_non_zero_values, q),
        vmax=np.quantile(emission_non_zero_values, 1 - q),
    )



    im = ax.imshow(
        emissions,
        norm=norm,
        cmap=cmap,
        # cmap="inferno",
        # extent=[wgs_84_extents[1],wgs_84_extents[3], wgs_84_extents[0],wgs_84_extents[2] ]
        extent=[x_min, x_max, y_min, y_max],
    )
    ax.xaxis.set_major_formatter("{x:6.0f}")
    ax.yaxis.set_major_formatter("{x:6.0f}")
    ax.set_title(f"Total {substance}: " f"{total_emissions:.2} " f"kg/y")
    fig.colorbar(
        im,
        label="kg/y/m2",
        extend="both",
        extendfrac=0.02
        # cax=cax,
    )
    if OVER_ZH:
        ax.set_xlim(2675000,2690000)
        ax.set_ylim(1242000, 1255000)
    fig.tight_layout()
    file_name = plots_path / f"raster_total_{substance}"

    fig.savefig(file_name.with_suffix(".png"))
    # fig.savefig(file_name.with_suffix(".pdf"))
    fig.clear()

# %%
