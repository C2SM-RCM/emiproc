"""Maps the swiss inventory to Icon."""
# %% Imports

from pathlib import Path
from emiproc.inventories.zurich import MapLuftZurich
from emiproc.inventories.tno import TNO_Inventory
from emiproc.inventories.swiss import SwissRasters
from emiproc.inventories.utils import (
    add_inventories,
    crop_with_shape,
    group_categories,
)
from emiproc.grids import LV95, WGS84, ICONGrid
from shapely.geometry import Polygon, Point
from emiproc.regrid import remap_inventory
from emiproc.inventories.categories_groups import CH_2_GNFR, TNO_2_GNFR, ZH_2_GNFR
from emiproc.plots import explore_inventory, explore_multilevel, plot_inventory
import geopandas as gpd
import cartopy.io.shapereader as shpreader
from emiproc.inventories.exports import export_icon_oem

# %% Select the path with my data
data_path = Path(r"C:\Users\coli\Documents\ZH-CH-emission\Data\CHEmissionen")
weights_path = Path(r"C:\Users\coli\Documents\emiproc\scripts\.emiproc_weights_swiss_2_icon")
weights_path.mkdir(parents=True, exist_ok=True)


#%%
inv_tno = TNO_Inventory(
    r"C:\Users\coli\Documents\emiproc\files\TNO_6x6_GHGco_v4_0\TNO_GHGco_v4_0_year2018.nc"
)
inv_tno.to_crs(LV95)
# %% Create the inventory object
inv_ch = SwissRasters(
    data_path=data_path,
    rasters_dir=data_path / "ekat_gridascii",
    rasters_str_dir=data_path / "ekat_str_gridascii",
    requires_grid=True,
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


def load_swiss_shape(
    crs_out: int = LV95,
) -> Polygon:
    crs_file = WGS84
    shpfilename = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )
    countries = list(shpreader.Reader(shpfilename).records())
    wgs_serie = gpd.GeoSeries([countries[91].geometry], crs=crs_file)
    return wgs_serie.to_crs(crs_out).iloc[0]


zh_poly = load_zurich_shape()
ch_poly = load_swiss_shape()
# %% crop using zurich shape


cropped_ch = crop_with_shape(
    inv_ch, zh_poly, keep_outside=True, modify_grid=True
)
cropped_zh = crop_with_shape(
    inv_zh, zh_poly, keep_outside=False,
)
cropped_tno = crop_with_shape(
    inv_tno, ch_poly, keep_outside=True, modify_grid=True
)


# %% group the categories
groupped_ch = group_categories(cropped_ch, CH_2_GNFR)
groupped_zh = group_categories(cropped_zh, ZH_2_GNFR)
groupped_tno = group_categories(cropped_tno, TNO_2_GNFR)

# Merge the groups
groups = {
    key: sum([ZH_2_GNFR.get(key, []), CH_2_GNFR.get(key, []), TNO_2_GNFR.get(key, [])], [])
    for key in (ZH_2_GNFR | CH_2_GNFR | TNO_2_GNFR).keys()
}

# %%

remaped_ch = remap_inventory(groupped_ch, icon_grid, weights_path / "remap_ch2icon")
remaped_zh = remap_inventory(groupped_zh, icon_grid, weights_path / "remap_zh2icon")
remaped_tno = remap_inventory(groupped_tno, icon_grid, weights_path / "remap_tno2icon")
#%%
combined = add_inventories(remaped_ch, remaped_zh)
combined = add_inventories(combined, remaped_tno)

# %%


export_icon_oem(
    combined,
    grid_file,
    weights_path / f"{grid_file.stem}_zh_ch_tno_combined.nc",
    group_dict=groups
)

#%% Test what it looks like only swiss
remaped_cropped_ch = remap_inventory(cropped_ch, icon_grid, weights_path / "remap_ch_cropped2icon")
export_icon_oem(
    group_categories(remaped_cropped_ch, CH_2_GNFR),
    grid_file,
    weights_path / f"{grid_file.stem}_ch_cropped.nc",
)
# %%
# View the data
col = ("GNFR_B", "CO2")
explore_inventory(remaped_tno, "GNFR_J", "CO2")
# mask = groupped_inv.gdf[col] > 0
# explore_multilevel(groupped_inv.gdf.loc[mask].iloc[10000:50000], col, logscale=True)
# %%
col = ('A', 'CO')
explore_multilevel(cropped_tno.gdf.iloc[300000:340000], col, logscale=False)

# %%
