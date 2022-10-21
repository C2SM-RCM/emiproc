"""plots netcdf raster files"""
#%% Plot the total emissions
from pathlib import Path
import matplotlib as mpl
import matplotlib.style
import numpy as np
from emiproc.plots import nclcmaps
import contextily as cx
import xarray as xr
import matplotlib.pyplot as plt
from emiproc.utilities import SEC_PER_YR

#%%
images_dir = Path(r"C:\Users\coli\Pictures\zh_rasters")

cmap = nclcmaps.cmap("WhViBlGrYeOrRe")
mpl.style.use("default")
from matplotlib.colors import LogNorm, SymLogNorm

#%%
file = Path(r"C:\Users\coli\Documents\ZH-CH-emission\output_files\mapluft_rasters\zurich_100x100_mapLuft_2020_v1.0.nc")
#file = Path(r"C:\Users\coli\Documents\ZH-CH-emission\output_files\mapluft_rasters\zurich_cropped_100x100_mapLuft_2020_v1.0.nc")
ds = xr.load_dataset(file)
x_min, x_max, y_min, y_max = min(ds["x"]), max(ds["x"]), min(ds["y"]), max(ds["y"])
ds
# %% PLot a single category
category = ds["source_category"].to_numpy()[1]
for category in ds["source_category"].to_numpy():
    sub = "CO2"
    emissions = ds[f"emi_{sub}"].loc[{"source_category": category}]
    total_emission = ds[f"emi_{sub}_total"].loc[{"source_category": category}]

    if np.all(np.isnan(total_emission) | (total_emission == 0)):
        continue
    fig, ax = plt.subplots(
        figsize=(10, 8),
    )
    cax = fig.add_axes([0.85, 0.3, 0.05, 0.4])

    # Add a map in the background
    # gdf_zh_shape = gpd.GeoDataFrame(geometry=[zh_shape], crs=SWISS_CRS).to_crs(WGS84)
    # cx.add_basemap(ax, crs=gdf_zh_shape.crs)
    # wgs_84_extents =[*gdf_zh_shape.bounds.iloc[0]]

    image = ax.imshow(
        emissions,
        norm=LogNorm(
            vmin=1e-6,
            vmax=1e6,
        ),
        cmap=cmap,
        # cmap="inferno",
        # extent=[wgs_84_extents[1],wgs_84_extents[3], wgs_84_extents[0],wgs_84_extents[2] ]
        extent=[x_min, x_max, y_min, y_max],
    )
    ax.xaxis.set_major_formatter("{x:6.0f}")
    ax.yaxis.set_major_formatter("{x:6.0f}")
    fig.colorbar(image, label=ds["emi_CO2_all_sectors"].units, cax=cax)
    ax.set_title(
        f"Emissions of {sub}: {category}, total: {total_emission.variable.values:.2}"
    )
    plt.show()

#%% Plot the comparison with zh inventory

# cmap = ListedColormap(['white', 'violet', 'blue', 'green', 'yellow', 'orange', 'red'], N=255)

total_emissions = ds["emi_CO2_all_sectors"]
fig, ax = plt.subplots(
    figsize=(10, 8),
    ncols=2,
    sharex=True,
    sharey=True,
    gridspec_kw={"right": 0.9, "wspace": 0},
)
cax = fig.add_axes([0.92, 0.3, 0.02, 0.4])

# Add a map in the background
# gdf_zh_shape = gpd.GeoDataFrame(geometry=[zh_shape], crs=SWISS_CRS).to_crs(WGS84)
# cx.add_basemap(ax, crs=gdf_zh_shape.crs)
# wgs_84_extents =[*gdf_zh_shape.bounds.iloc[0]]
norm = LogNorm(
    vmin=1e-1,
    vmax=1e4,
)

image = ax[0].imshow(
    total_emissions,
    norm=norm,
    cmap=cmap,
    # cmap="inferno",
    # extent=[wgs_84_extents[1],wgs_84_extents[3], wgs_84_extents[0],wgs_84_extents[2] ]
    extent=[x_min, x_max, y_min, y_max],
)
ax[0].xaxis.set_major_formatter("{x:6.0f}")
ax[0].yaxis.set_major_formatter("{x:6.0f}")
ax[0].set_title("Total CO2 from my raster")


import rasterio

src = rasterio.open(
    r"C:\Users\coli\Documents\e214415c-3038-11ed-8896-005056b0ce82\data\ha_co2_2020.tif"
)
zh_inventar = src.read(1) * 1e9 / SEC_PER_YR / (100 * 100)
ax[1].imshow(
    zh_inventar,
    norm=norm,
    cmap=cmap,
    extent=[x_min, x_max, y_min, y_max],
)

ax[1].set_title("From ZH maps")

fig.colorbar(image, label=ds["emi_CO2_all_sectors"].units, cax=cax)
fig.show()
# %%



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
    "Benzol",
]:
    total_emissions = ds[f"emi_{substance}_all_sectors"].to_numpy()
    total_emissions_per_sector = ds[f"emi_{substance}_total"]

    fig, ax = plt.subplots(
        figsize=(20, 16),
        # gridspec_kw={"right": 0.9, "wspace": 0},
    )
    # cax = fig.add_axes([0.87, 0.15, 0.02, 0.7])

    emission_non_zero_values = total_emissions[total_emissions > 0]
    q = 0.05
    norm = LogNorm(
        vmin=np.quantile(emission_non_zero_values, q),
        vmax=np.quantile(emission_non_zero_values, 1 - q),
    )

    im = ax.imshow(
        total_emissions,
        norm=norm,
        cmap=cmap,
        # cmap="inferno",
        # extent=[wgs_84_extents[1],wgs_84_extents[3], wgs_84_extents[0],wgs_84_extents[2] ]
        extent=[x_min, x_max, y_min, y_max],
    )
    ax.xaxis.set_major_formatter("{x:6.0f}")
    ax.yaxis.set_major_formatter("{x:6.0f}")
    ax.set_title(
        f"Total {substance} from Zurich: "
        f"{total_emissions_per_sector.sum().values:.2} "
        f"{total_emissions_per_sector.units}"
    )
    fig.colorbar(
        im,
        label=ds["emi_CO2_all_sectors"].units,
        extend='both',
        extendfrac=0.02
        #cax=cax,
    )
    fig.tight_layout()
    file_name = images_dir / f"raster_{substance}_{file.stem}"
    fig.savefig(file_name.with_suffix(".png"))
    #fig.savefig(file_name.with_suffix(".pdf"))

# %%
