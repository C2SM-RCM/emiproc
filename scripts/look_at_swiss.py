"""Comparison of Swiss emissions in 2020 and 2022."""

# %% Imports
from pathlib import Path

from emiproc.inventories.swiss import SwissRasters
from emiproc.inventories.utils import drop
from emiproc.speciation import merge_substances

# %% Select the path with my data
data_path = Path(r"C:\Users\coli\Documents\ZH-CH-emission\Data\CHEmissionen")


# %% Create the inventory object
ch_22 = SwissRasters(
    data_path=data_path / "CH_Emissions_2015_2020_2022_CO2_CO2biog_CH4_N2O_BC_AP.xlsx",
    rasters_dir=data_path / "ekat_gridascii",
    rasters_str_dir=data_path / "ekat_str_gridascii_v_swiss2icon",
    requires_grid=False,
    year=2022,
)
ch_20 = SwissRasters(
    data_path=data_path / "CH_Emissions_2015_2020_2022_CO2_CO2biog_CH4_N2O_BC_AP.xlsx",
    rasters_dir=data_path / "ekat_gridascii",
    rasters_str_dir=data_path / "ekat_str_gridascii_v_swiss2icon",
    requires_grid=False,
    year=2020,
)
ch_20 = drop(ch_20, substances=["CO2", "CO2_biog"], keep_instead_of_drop=True)
ch_20 = merge_substances(ch_20, {"CO2": ["CO2", "CO2_biog"]}, inplace=True)
ch_22 = drop(ch_22, substances=["CO2", "CO2_biog"], keep_instead_of_drop=True)
ch_22 = merge_substances(ch_22, {"CO2": ["CO2", "CO2_biog"]}, inplace=True)
