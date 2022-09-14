"""Maps the swiss inventory to Icon."""
# %% Imports
from pathlib import Path
import pandas as pd
from emiproc.inventories import SwissRasters
from emiproc.inventories.utils import load_category


# %% Select the path with my data
data_path = Path(r"C:\Users\coli\Documents\ZH-CH-emission\Data\CHEmissionen")


# %% Load the file with the point sources
df_eipwp = load_category(
    data_path / "ekat_ch_basisraster.gdb" / "ekat_ch_basisraster.gdb", "eipwp" + "_2015"
)
df_eipwp = df_eipwp.rename(
    columns={
        "CO2_15": "CO2",
        "CH4_15": "CH4",
        "N2O_15": "N2O",
        "NOx_15": "NOx",
        "CO_15": "CO",
        "NMVOC_15": "NMVOC",
        "SO2_15": "SO2",
        "NH3_15": "NH3",
    }
)
df_eipwp["F-Gase"] = 0.0




# %% Load the excel sheet with the total emissions
df_emissions = pd.read_excel(
   data_path / "Emissionen-2015-je-Emittentengruppe.xlsx",
    header=2,
    index_col="Basisraster",
    usecols=[5, 6, 7, 8, 10, 11, 12, 13, 14, 16],
)
df_emissions = df_emissions.rename(columns={"CO2 foss/geog": "CO2"})
df_emissions = df_emissions.loc[~pd.isna(df_emissions.index)]



# %% Create the inventory object
inv = SwissRasters(
    rasters_dir=data_path / "ekat_gridascii",
    rasters_str_dir=data_path / "ekat_str_gridascii",
    df_eipwp = df_eipwp,
    df_emission=df_emissions
)

#%%
inv.gdf.geometry
