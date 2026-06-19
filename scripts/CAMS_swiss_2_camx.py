"""Maps the swiss inventory to Icon."""
# %% Imports
from pathlib import Path
from emiproc.inventories.tno import TNO_Inventory
from emiproc.inventories.swiss import SwissRasters
from emiproc.inventories.utils import (
    add_inventories,
    crop_with_shape,
    group_categories,
)
from emiproc.grids import LV95, WGS84
from emiproc.regrid import remap_inventory
from emiproc.inventories.categories_groups import CH_2_GNFR_v3, TNO_2_GNFR_f1_f2_f3_f4
import geopandas as gpd
import pandas as pd
from emiproc.utilities import get_natural_earth


# %% Select the path with my data
data_path_ch = Path(r"/capstor/store/cscs/userlab/lp171/ckeller/Emissions/Inventories_scenarios/2050/BAFU/CH_emissions_2050_WAM_scenario.csv")
data_path_prtr = Path(r"/capstor/store/cscs/userlab/lp171/ckeller/Emissions/Inventories_scenarios/Point_sources/SwissPRTR-Daten_2007-2022.xlsx")
path_rasters = Path(r"/capstor/store/cscs/userlab/lp171/ckeller/Emissions/Rasters_and_profiles/CH")

data_path_tno = r"/capstor/store/cscs/userlab/lp171/ckeller/Emissions/Inventories_scenarios/2050/CAMS_TNO/CAMS-REG-v8_1_scen_MFR_emissions_year2050.nc"

weights_path = Path(r"/users/ckeller/emiproc/scripts/.emiproc_weights_CAMS_swiss_2_icon")
weights_path.mkdir(parents=True, exist_ok=True)

year = 2050
scenario = "MFR"

# %% Load the inventory to an object
inv_tno = TNO_Inventory(data_path_tno, 
            substances_mapping = {
            "co": "CO",
            "nox": "NOx",
            "ch4": "CH4",
            "nmvoc": "VOC",
            "nh3": "NH3",
            "sox": "SO2",
            "pm2_5": "PM25",
            "pm10":"PM10"},
            profiles_dir=r"/capstor/store/cscs/userlab/lp171/ckeller/Emissions/Rasters_and_profiles/TNO"
)
inv_tno.to_crs(LV95)

# %% Create the inventory object
inv_ch = SwissRasters(
    filepath_csv_totals=data_path_ch,
    filepath_point_sources=data_path_prtr,
    #rasters_dir=data_path / "ekat_gridascii",
    rasters_dir=path_rasters / "ekat_gridascii_PSI",
    #rasters_str_dir=data_path / "ekat_str_gridascii_SCENE",
    rasters_str_dir=path_rasters / "ekat_str_gridascii_PSI",
    requires_grid=True,
    year = year,
    scenario = scenario,
)

#%% Load the swiss polygon shape for cropping
gdf = get_natural_earth(resolution="10m", category="cultural", name="admin_0_countries")
gdf = gdf.to_crs(LV95)
ch_poly = gdf.set_index('SOVEREIGNT').loc['Switzerland'].geometry

# %% crop using Swiss shape
cropped_ch = crop_with_shape(
    inv_ch, ch_poly, keep_outside=False, modify_grid=True
)

cropped_tno = crop_with_shape(
    inv_tno, ch_poly, keep_outside=True, modify_grid=True
)

# %% group the categories
grouped_ch = group_categories(
    inv=cropped_ch, 
    categories_group=CH_2_GNFR_v3, 
    ignore_missing=True
)

grouped_tno = group_categories(
    inv=cropped_tno,    
    categories_group=TNO_2_GNFR_f1_f2_f3_f4,
    ignore_missing=True 
)
#%%
remapped_ch = remap_inventory(
    grouped_ch,
    inv_tno.geometry,
    weights_file=weights_path / "remap_ch2camx", 
    keep_gdfs=True
    )

remapped_tno = remap_inventory(
    grouped_tno,    
    inv_tno.geometry,
    weights_file=weights_path / "remap_tno2camx",
    keep_gdfs=True
)

# %%
combined = add_inventories(remapped_ch, remapped_tno)

# %%
name_2_report = {
    "GNFR_A": "A",
    "GNFR_B": "B",
    "GNFR_C": "C",
    "GNFR_D": "D",
    "GNFR_E": "E",
    "GNFR_F1": "F1",
    "GNFR_F2": "F2",
    "GNFR_F3": "F3",
    "GNFR_F4": "F4",
    "GNFR_G": "G",
    "GNFR_H": "H",
    "GNFR_I": "I",
    "GNFR_J": "J",
    "GNFR_K": "K",
    "GNFR_L": "L",
    "GNFR_N": "N",
    "GNFR_O": "O",
    "GNFR_P": "P",
    # We used R for others
    "GNFR_R": "M",
}

# %%
combined.categories

# %%
combined.to_crs(inv_tno.grid.crs)

# %%
from emiproc.utilities import get_country_mask
ctry_codes_as = get_country_mask(inv_tno.geometry)
dfs = []
for cat in combined.categories:
    dfs.append(
        pd.DataFrame(
            {
                "Lon_rounded": inv_tno.grid.centers.x.round(2),
                "Lat_rounded": inv_tno.grid.centers.y.round(3),
                "ISO3": ctry_codes_as,
                "Year": year,
                "GNFR_Sector": name_2_report[cat],
                "SourceType": "A",
            }
            | {
                sub: combined.gdf[(cat, sub)] 
                if (cat, sub) in combined.gdf.columns
                # Use 0 in case the cat sub is not present
                else 0.0
                for sub in ["CH4", "CO", "NH3", "VOC", "NOx", "PM10", "PM25", "SO2"]
            }
        )
    )

# %%
# Add the point sources 
for cat, gdf in combined.gdfs.items():
    #gdf = gdf.to_crs(WGS84)
    geom_ps = gpd.GeoSeries(gdf.geometry)
    ctry_codes_ps = get_country_mask(geom_ps)
    dfs.append(
        pd.DataFrame(
            {
                    "Lon_rounded": gdf.geometry.x.round(2),
                    "Lat_rounded": gdf.geometry.y.round(3),
                    "ISO3": ctry_codes_ps,
                    "Year": year,
                    "GNFR_Sector": name_2_report[cat],
                    "SourceType": "P",
                }
                |  {
                sub: gdf[sub] 
                if sub in gdf.columns
                # Use 0 in case the cat sub is not present
                else 0.0
                for sub in ["CH4", "CO", "NH3", "VOC", "NOx", "PM10", "PM25", "SO2"]
            }
        ))
# %%
results = pd.concat(dfs)

#results_ch = results[results["ISO3"] == "CHE"]

# %%
mask_all_0 = results[["CH4", "CO", "NH3", "VOC", "NOx", "PM10", "PM25", "SO2"]].sum(axis=1) == 0.0
results.loc[~mask_all_0].to_csv(
    "/capstor/store/cscs/userlab/lp171/ckeller/Emissions/Inventories_scenarios/2050/CAMx/TNO_BAFU_2050_MFR_CAMx_full_grid.csv", index=False, sep=","
)

# %%
