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
#file = Path(r"C:\Users\coli\Documents\ZH-CH-emission\output_files\mapluft_rasters\zurich_cropped_100x100_mapLuft_2020_v1.3.nc")
ds = xr.load_dataset(file)
x_min, x_max, y_min, y_max = min(ds["x"]), max(ds["x"]), min(ds["y"]), max(ds["y"])
ds


#%% Plot the comparison with zh inventory


# total_emissions = ds["emi_CO2_all_sectors"]
# fig, ax = plt.subplots(
#     figsize=(10, 8),
#     ncols=2,
#     sharex=True,
#     sharey=True,
#     gridspec_kw={"right": 0.9, "wspace": 0},
# )
# cax = fig.add_axes([0.92, 0.3, 0.02, 0.4])# 

# # Add a map in the background
# # gdf_zh_shape = gpd.GeoDataFrame(geometry=[zh_shape], crs=SWISS_CRS).to_crs(WGS84)
# # cx.add_basemap(ax, crs=gdf_zh_shape.crs)
# # wgs_84_extents =[*gdf_zh_shape.bounds.iloc[0]]
# norm = LogNorm(
#     vmin=1e-1,
#     vmax=1e4,
# )# 

# image = ax[0].imshow(
#     total_emissions,
#     norm=norm,
#     cmap=cmap,
#     # cmap="inferno",
#     # extent=[wgs_84_extents[1],wgs_84_extents[3], wgs_84_extents[0],wgs_84_extents[2] ]
#     extent=[x_min, x_max, y_min, y_max],
# )
# ax[0].xaxis.set_major_formatter("{x:6.0f}")
# ax[0].yaxis.set_major_formatter("{x:6.0f}")
# ax[0].set_title("Total CO2 from my raster")# 
# 

# import rasterio# 

# src = rasterio.open(
#     r"C:\Users\coli\Documents\e214415c-3038-11ed-8896-005056b0ce82\data\ha_co2_2020.tif"
# )
# zh_inventar = src.read(1) * 1e9 / SEC_PER_YR / (100 * 100)
# ax[1].imshow(
#     zh_inventar,
#     norm=norm,
#     cmap=cmap,
#     extent=[x_min, x_max, y_min, y_max],
# )# 

# ax[1].set_title("From ZH maps")# 

# fig.colorbar(image, label=ds["emi_CO2_all_sectors"].units, cax=cax)
# fig.show()

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
    total_emissions = ds[f"emi_{substance}_all_sectors"].to_numpy()
    total_emissions_per_sector = ds[f"emi_{substance}_total"]

    fig, ax = plt.subplots(
        figsize=(10, 8),
        # gridspec_kw={"right": 0.9, "wspace": 0},
    )
    # cax = fig.add_axes([0.87, 0.15, 0.02, 0.7])

    emission_non_zero_values = total_emissions[total_emissions > 0]
    q_max = 0.001
    q_min = 0.2
    norm = LogNorm(
        #vmin=1e-1,
        vmin=np.quantile(emission_non_zero_values, q_min),
        #vmax=1e4,
        vmax=np.quantile(emission_non_zero_values, 1 - q_max),
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
