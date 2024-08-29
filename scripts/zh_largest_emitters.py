"""Gets the largest point source in the inventory."""

# %%
from pathlib import Path

import pandas as pd

from emiproc.inventories.zurich import MapLuftZurich

# %% define some parameters for the output


YEAR = 2022

INCLUDE_SWISS_OUTSIDE = False
swiss_data_path = Path(
    r"C:\Users\coli\Documents\ZH-CH-emission\Data\CHEmissionen\CH_Emissions_2015_2020_2022_CO2_CO2biog_CH4_N2O_BC_AP.xlsx"
)
outdir = Path(r"C:\Users\coli\Documents\ZH-CH-emission\output_files\mapluft_rasters")
mapluft_dir = Path(r"C:\Users\coli\Documents\Data\mapluft_emissionnen_kanton")
mapluf_file = mapluft_dir / f"mapLuft_{YEAR}_v2024.gdb"


# %% load the zurich inventory
inv = MapLuftZurich(mapluf_file, substances=["CO2"])
# %%
gdf_biggest_emissions = []
for cat, gdf in inv.gdfs.items():
    # Check if the geometry is a point
    if gdf.geometry.type[0] != "Point":
        continue

    gdf_max = gdf.sort_values("CO2").tail(100)
    gdf_max["category"] = cat
    gdf_biggest_emissions.append(gdf_max)

gdf_biggest_emissions = pd.concat(gdf_biggest_emissions)
biggest_emitters = gdf_biggest_emissions.sort_values("CO2", ascending=False)
biggest_emitters
# %%
biggest_emitters["x"] = biggest_emitters.geometry.x
biggest_emitters["y"] = biggest_emitters.geometry.y
biggest_emitters.iloc[:100].to_csv("biggest_point_sources.csv")
# %%
