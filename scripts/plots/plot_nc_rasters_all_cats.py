"""plots netcdf raster files"""

# %% Plot the total emissions
from pathlib import Path
import matplotlib as mpl
from emiproc.plots import nclcmaps
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# %%

cmap = nclcmaps.cmap("WhViBlGrYeOrRe")
mpl.style.use("default")
from matplotlib.colors import LogNorm, SymLogNorm


# %%
file = Path(
    r"C:\Users\coli\Documents\ZH-CH-emission\output_files\mapluft_rasters\zurich_cropped_Fsplit_100x100_mapLuft_2022_v2024_v1.7.nc"
)
ds = xr.load_dataset(file)
x_min, x_max, y_min, y_max = min(ds["x"]), max(ds["x"]), min(ds["y"]), max(ds["y"])
ds
images_dir = file.with_suffix(".plots")
images_dir.mkdir(exist_ok=True)

substances = [
    "CO2_bio",
    "CO2_fos",
    "CO",
    "BC",
    # "PM10ex",
    # "PM10non",
    # "PM25ex",
    # "PM25non",
    "SO2",
    "NOx",
    "N2O",
    #"NH3",
    "CH4",
    # "VOC",
    # "benzene",
]

categories = list(ds["source_category"].values)

# %% Make plots with the total emissions

fig, ax = plt.subplots()

colors_of_category = {
    "GNFR_A": "olive",
    "GNFR_B": "brown",
    "GNFR_C": "tomato",
    "GNFR_D": "cyan",
    "GNFR_E": "blue",
    "GNFR_F-cars": "orange",
    "GNFR_F-heavy_duty": "darkorange",
    "GNFR_F-light_duty": "peru",
    "GNFR_F-two_wheels": "sandybrown",
    "GNFR_G": "hotpink",
    "GNFR_I": "purple",
    "GNFR_J": "gray",
    "GNFR_K": "teal",
    "GNFR_L": "maroon",
    "GNFR_O": "green",
    "GNFR_R": "lightgreen",
}
co2_substances = ["CO2_bio", "CO2_fos"]
for i, substance in enumerate(substances):

    total_emissions_per_sector = ds[f"emi_{substance}_total"]
    if substance in co2_substances:
        # Sum over all the CO2
        total_sum = np.sum([ds[f"emi_{s}_all_sectors"].sum().values for s in co2_substances])
    else:
        total_sum = total_emissions_per_sector.sum().values
    # Stack the emissions from each category
    cumsum = 0

    for cat in sorted(categories):
        da = total_emissions_per_sector.sel(source_category=cat)
        emissions = da.values
        emissions = np.nan_to_num(emissions, nan=0.0)
        # Remove the nan values
        kwargs = {}
        if i == 0:
            kwargs["label"] = cat
        ax.bar(
            x=i,
            bottom=cumsum / total_sum,
            height=emissions / total_sum,
            width=0.8,
            color=colors_of_category[cat],
            **kwargs,
        )
        cumsum += emissions

ax.set_xticks(range(len(substances)))
ax.set_xticklabels(substances, rotation=45)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.00), ncol=3)
ax.set_ylabel("Fraction of total emissions")

fig.tight_layout()
file_name = images_dir / f"total_emissions_fractions_{file.stem}"
fig.savefig(file_name.with_suffix(".png"))


# %% plots all substances


q_max = 0.001
q_min = 0.2
for substance in substances:

    for cat in categories + ["__total__"]:
        if cat == "__total__":
            da = ds[f"emi_{substance}_all_sectors"]
            cat_sub_emissions = da.to_numpy()
        else:
            da = ds[f"emi_{substance}"]
            cat_sub_emissions = da.sel(source_category=cat).to_numpy()

        # cax = fig.add_axes([0.87, 0.15, 0.02, 0.7])

        emission_non_zero_values = cat_sub_emissions[cat_sub_emissions > 0]
        if not np.any(emission_non_zero_values):
            # No emissions at all
            continue

        fig, axes = plt.subplots(
            figsize=(10, 8),
            ncols=2,
            # Set the colorbar to a smaller size
            gridspec_kw={"width_ratios": [1, 0.05]},
        )
        ax, cax = axes
        norm = LogNorm(
            # vmin=1e-1,
            vmin=np.quantile(emission_non_zero_values, q_min),
            # vmax=1e4,
            vmax=np.quantile(emission_non_zero_values, 1 - q_max),
        )

        ax.clear()
        im = ax.imshow(
            cat_sub_emissions,
            norm=norm,
            cmap=cmap,
            # cmap="inferno",
            # extent=[wgs_84_extents[1],wgs_84_extents[3], wgs_84_extents[0],wgs_84_extents[2] ]
            extent=[x_min, x_max, y_min, y_max],
        )
        ax.xaxis.set_major_formatter("{x:6.0f}")
        ax.yaxis.set_major_formatter("{x:6.0f}")
        ax.set_title(
            f"Total {substance} {cat} from Zurich: "
            f"{cat_sub_emissions.sum():.2} "
            f"{da.units}"
        )
        cax.clear()

        fig.colorbar(
            im,
            extend="both",
            extendfrac=0.02,
            label=da.units,
            cax=cax,
        )
        fig.tight_layout()
        file_name = images_dir / f"raster_{substance}_{cat}_{file.stem}"
        fig.savefig(file_name.with_suffix(".png"))
        # fig.savefig(file_name.with_suffix(".pdf"))

        # close
        plt.close(fig)

# %%
