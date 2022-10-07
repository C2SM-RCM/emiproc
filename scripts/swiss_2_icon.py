"""Maps the swiss inventory to Icon."""
# %% Imports
import collections
import itertools
from os import PathLike
from pathlib import Path
import pandas as pd
from emiproc.inventories import Inventory
from emiproc.inventories.zurich import MapLuftZurich
from emiproc.inventories.swiss import SwissRasters
from emiproc.inventories.utils import (
    add_inventories,
    crop_with_shape,
    group_categories,
    load_category,
)
from emiproc.plots import explore_inventory, explore_multilevel
from emiproc.grids import LV95, WGS84, Grid, ICONGrid
from shapely.geometry import Polygon, Point
from emiproc.regrid import remap_inventory
from emiproc.inventories.categories_groups import CH_2_GNFR, ZH_2_GNFR

import geopandas as gpd
import numpy as np

# %% Select the path with my data
data_path = Path(r"C:\Users\coli\Documents\ZH-CH-emission\Data\CHEmissionen")
weights_path = Path(".emiproc_weights_swiss_2_icon")
weights_path.mkdir(parents=True, exist_ok=True)




# %% Create the inventory object
inv_ch = SwissRasters(
    data_path=data_path,
    rasters_dir=data_path / "ekat_gridascii",
    rasters_str_dir=data_path / "ekat_str_gridascii",
    requires_grid=False,
)
inv_ch.gdf

#%% load mapluft
inv_zh = MapLuftZurich(
    Path(r"H:\ZurichEmissions\Data\mapLuft_2020_v2021\mapLuft_2020_v2021.gdb")
)

#%% Load the icon grid
grid_file = Path(
    r"C:\Users\coli\Documents\ZH-CH-emission\icon_Zurich_R19B9_wide_DOM01.nc"
)
icon_grid = ICONGrid(grid_file)
#%%
def load_zurich_shape(
    zh_raw_file=r"C:\Users\coli\Documents\ZH-CH-emission\Data\Zurich_borders.txt",
    crs_file: int = WGS84,
    crs_out: int = LV95,
) -> Polygon:
    with open(zh_raw_file, "r") as f:
        points_list = eval(f.read())
        zh_poly = Polygon(points_list[0])
        zh_poly_df = gpd.GeoDataFrame(geometry=[zh_poly], crs=crs_file).to_crs(crs_out)
        zh_poly = zh_poly_df.geometry.iloc[0]
        return zh_poly


zh_poly = load_zurich_shape()
# %% crop using zurich shape


cropped_ch = crop_with_shape(
    inv_ch, zh_poly, keep_outside=True, weight_file=weights_path / "ch_out_zh"
)
cropped_zh = crop_with_shape(
    inv_zh, zh_poly, keep_outside=False, weight_file=weights_path / "zh_in_zh"
)


# %% group the categories
groupped_ch = group_categories(cropped_ch, CH_2_GNFR)
groupped_zh = group_categories(cropped_zh, ZH_2_GNFR)

# %%

remaped_ch = remap_inventory(groupped_ch, icon_grid, weights_path / "remap_ch2icon")
remaped_zh = remap_inventory(groupped_zh, icon_grid, weights_path / "remap_zh2icon")
#%%
combined = add_inventories(remaped_ch, remaped_zh)

# %%
import importlib
import emiproc.inventories.exports

importlib.reload(emiproc.inventories.exports)
from emiproc.inventories.exports import export_icon_oem

export_icon_oem(combined, grid_file, weights_path / f"{grid_file.stem}_zh_ch_combined.nc", ZH_2_GNFR | CH_2_GNFR)
# %%
# View the data
col = ("GNFR_B", "CO2")
# mask = groupped_inv.gdf[col] > 0
# explore_multilevel(groupped_inv.gdf.loc[mask].iloc[10000:50000], col, logscale=True)
# %%
#%matplotlib qt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import matplotlib.style

matplotlib.style.use("default")

plt.imshow(
    groupped_inv.gdf[col]
    .to_numpy()
    .reshape((groupped_inv.grid.ny, groupped_inv.grid.nx)),
    norm=LogNorm(),
)
plt.show()

# %%
