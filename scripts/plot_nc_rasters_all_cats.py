"""plots netcdf raster files"""
#%% Plot the total emissions
from pathlib import Path
import matplotlib as mpl
from emiproc.plots import nclcmaps
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

#%%
images_dir = Path(r"C:\Users\coli\Pictures\zh_rasters_in_ch_F_split")
images_dir.mkdir(exist_ok=True)
cmap = nclcmaps.cmap("WhViBlGrYeOrRe")
mpl.style.use("default")
from matplotlib.colors import LogNorm, SymLogNorm


#%%
file = Path(r"C:\Users\coli\Documents\ZH-CH-emission\output_files\mapluft_rasters\zurich_inside_swiss_Fsplit_100x100_mapLuft_2020_v1.4.nc")
ds = xr.load_dataset(file)
x_min, x_max, y_min, y_max = min(ds["x"]), max(ds["x"]), min(ds["y"]), max(ds["y"])
ds


# %% plots all substances

for substance in [
    "CO2",
    "CO",
    "PM10ex",
    "PM10non",
    "PM25ex",
    "PM25non",
    "SO2",
    "NOx",
    "N2O",
    "NH3",
    "CH4",
    "VOC",
    "benzene",
]:
    for cat in ds['source_category']:
        da = ds[f"emi_{substance}"]
        cat_sub_emissions = da.sel(source_category=cat).to_numpy()

        fig, ax = plt.subplots(
            figsize=(10, 8),
            # gridspec_kw={"right": 0.9, "wspace": 0},
        )
        # cax = fig.add_axes([0.87, 0.15, 0.02, 0.7])

        emission_non_zero_values = cat_sub_emissions[cat_sub_emissions > 0]
        if not np.any(emission_non_zero_values):
            # No emissions at all
            continue
        q_max = 0.001
        q_min = 0.2
        norm = LogNorm(
            #vmin=1e-1,
            vmin=np.quantile(emission_non_zero_values, q_min),
            #vmax=1e4,
            vmax=np.quantile(emission_non_zero_values, 1 - q_max),
        )

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
            f"Total {substance} {cat.values} from Zurich: "
            f"{cat_sub_emissions.sum():.2} "
            f"{da.units}"
        )
        fig.colorbar(
            im,
            label=da.units,
            extend='both',
            extendfrac=0.02
            #cax=cax,
        )
        fig.tight_layout()
        file_name = images_dir / f"raster_{substance}_{cat.values}_{file.stem}"
        fig.savefig(file_name.with_suffix(".png"))
        #fig.savefig(file_name.with_suffix(".pdf"))

# %%
