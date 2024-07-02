"""Maps the swiss inventory to Icon.

This example is for a simulation over the city of zuirch.
However the domains goes far beyond the city limits, and even touches germany.

So here we will do a triple nesting:
- We put the zurich inventory in the city limits
- We put the swiss inventory in the country limits but we remove the zurich part
- We put the tno inventory outside the country limits

"""
# %% Imports
from pathlib import Path
from emiproc.inventories.zurich import MapLuftZurich
from emiproc.inventories.tno import TNO_Inventory
from emiproc.inventories.swiss import SwissRasters
from emiproc.inventories.utils import (
    add_inventories,
    crop_with_shape,
    group_categories,
    drop,
)
from emiproc.speciation import merge_substances
from emiproc.grids import LV95, WGS84, ICONGrid
from shapely.geometry import Polygon, Point
from emiproc.regrid import remap_inventory
from emiproc.inventories.categories_groups import CH_2_GNFR, TNO_2_GNFR
from emiproc.inventories.zurich.gnrf_groups import ZH_2_GNFR
import geopandas as gpd
from emiproc.exports.icon import TemporalProfilesTypes, export_icon_oem
from emiproc.utilities import get_natural_earth
from emiproc.profiles.temporal_profiles import from_yaml
from emiproc.profiles.vertical_profiles import VerticalProfile
import numpy as np
from emiproc import FILES_DIR

# %% Select the path with my data
data_path = Path(r"C:\Users\coli\Documents\ZH-CH-emission\Data\CHEmissionen")


# %%
tno_path = Path(r"C:\Users\coli\Documents\Data\TNO\AVENGERS_GHGs_year2021.nc")
inv_tno = TNO_Inventory(
    tno_path,
    substances_mapping = { "co2": "CO2"}
)
groupping_tno = {}
for cat in inv_tno.categories:
    # We need only the first letter for the categories
    gnfr_cat = f"GNFR_{cat[0]}"
    if gnfr_cat not in groupping_tno:
        groupping_tno[gnfr_cat] = []
    groupping_tno[gnfr_cat].append(cat)
inv_tno.to_crs(LV95)
# %% Create the inventory object
inv_ch = SwissRasters(
    data_path=data_path / "CH_Emissions_2015_2020_2022_CO2_CO2biog_CH4_N2O_BC_AP.xlsx",
    rasters_dir=data_path / "ekat_gridascii",
    rasters_str_dir=data_path / "ekat_str_gridascii_v_swiss2icon",
    requires_grid=True,
    year=2022,
)
inv_ch = drop(inv_ch, substances=["CO2", "CO2_biog"], keep_instead_of_drop=True)
inv_ch = merge_substances(inv_ch, {"CO2": ["CO2", "CO2_biog"]}, inplace=True)
inv_ch

# %% load mapluft
zh_path = Path(
    r"C:\Users\coli\Documents\Data\mapluft\mapLuft_2022_v2024.gdb"
)
inv_zh = MapLuftZurich(zh_path,substances=["CO2"])


# %% Load the icon grid
grid_file = Path(
    r"C:\Users\coli\Documents\ZH-CH-emission\for_nikolai\icon_Zurich_R19B9_beo_DOM01_v2.nc"
)
weights_path = grid_file.with_suffix("")
weights_path.mkdir(parents=True, exist_ok=True)
icon_grid = ICONGrid(grid_file)


# %%
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

gdf = get_natural_earth(resolution="10m", category="cultural", name="admin_0_countries")
gdf = gdf.to_crs(LV95)
ch_poly = gdf.set_index("SOVEREIGNT").loc["Switzerland"].geometry

# %% crop using zurich and swiss shapes


cropped_ch = crop_with_shape(inv_ch, zh_poly, keep_outside=True, modify_grid=True)
cropped_zh = crop_with_shape(
    inv_zh,
    zh_poly,
    keep_outside=False,
)
cropped_tno = crop_with_shape(inv_tno, ch_poly, keep_outside=True, modify_grid=True)


# %% group the categories
groupped_ch = group_categories(cropped_ch, CH_2_GNFR, ignore_missing=True)
groupped_zh = group_categories(cropped_zh, ZH_2_GNFR, ignore_missing=True)
groupped_tno = group_categories(
    cropped_tno,
    groupping_tno,
)

# Merge the groups
groups = {
    key: sum(
        [ZH_2_GNFR.get(key, []), CH_2_GNFR.get(key, []), groupping_tno.get(key, [])], []
    )
    for key in (ZH_2_GNFR | CH_2_GNFR | groupping_tno).keys()
}

# %%

remaped_ch = remap_inventory(groupped_ch, icon_grid, weights_path / "remap_ch2icon")
remaped_zh = remap_inventory(groupped_zh, icon_grid, weights_path / f"remap_zh2icon_{zh_path.stem}")
remaped_tno = remap_inventory(groupped_tno, icon_grid, weights_path / f"remap_tno2icon_{tno_path.stem}")
# %%
combined = add_inventories(remaped_ch, remaped_zh)
combined = add_inventories(remaped_tno, combined)

# %%
groupped_tno.t_profiles_indexes
# %% Add custom profiles


ships_profiles = from_yaml(FILES_DIR / "profiles" / "yamls" / "ship.yaml")

combined.set_profile(
    profile=ships_profiles,
    category="GNFR_G",
)
# R is the others category and not profile is defined in TNO
combined.set_profile(
    profile=[],
    category="GNFR_R",
)
combined.set_profile(
    # Ground emission (1) at 0m (ground level)
    profile=VerticalProfile(ratios=np.array([1.0, 0.0]), height=np.array([1.0, 2.0])),
    category="GNFR_R",
)

# %%


for profile in [TemporalProfilesTypes.HOUR_OF_YEAR, TemporalProfilesTypes.THREE_CYCLES]:
    export_icon_oem(
        inv=combined,
        icon_grid_file=grid_file,
        output_dir=grid_file.parent / f"{grid_file.stem}_zh_ch_tno_combined",
        group_dict=groups,
        substances=["CO2"],
        year=2020,
        temporal_profiles_type=profile,
    )


# %%
