"""Prepare Zurich city CO2 inventory for GRAL

The script combines the following inventories:
- MapLuft Zurich inventory (the City's inventory)
- Swiss inventory (from the Swiss Federal Office for the Environment)
- Human respiration inventory (from the Quartieranalyse data)

and writes it out to GRAL format with point, line and area sources grouped
by source groups according to Ivo Suter's UGZ report.

The GRAL grid definition is read from a GRAL rundir using the pygg package, 
which is specified in the GRAL_GRID variable.
"""
# %%
# autoreload modules in interactive python
%load_ext autoreload
%autoreload 2
# %%

from pathlib import Path
import logging
from typing import cast

import geopandas as gpd
from shapely.geometry import Polygon

from emiproc.exports.gral import export_to_gral
from emiproc.grids import LV95, WGS84
from emiproc.inventories.swiss import SwissRasters
from emiproc.inventories.utils import (
    add_inventories,
    crop_with_shape,
    clip_box,
    group_categories,
    validate_group,
    drop,
)
from emiproc.human_respiration import (
    load_data_from_quartieranalyse,
    people_to_emissions,
    EmissionFactor,
)
from emiproc.inventories.zurich import MapLuftZurich
from emiproc.inventories.zurich.gral_groups \
    import ZH_CO2_Groups, ZH_CO2_DUCK_Groups, CH_CO2_Groups

#from emiproc.regrid import remap_inventory
from emiproc.speciation import merge_substances
from emiproc.utilities import Units
from emiproc.inventories.zurich.duck import DuckDBInventory
from pygg.grids import GralGrid

# %% define some parameters for the output

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# define year for which to generate emissions
YEAR = 2024

# Whether to include the Swiss inventory outside of Zurich in the output
INCLUDE_SWISS_OUTSIDE = True

# Whether to add the human respiration
ADD_HUMAN_RESPIRATION = True

# Substances to include (so far only CO2)
substances = ["CO2"]

# Paths to data
emis_dir = Path("/cluster/home/dbrunner/data/emissions")
swiss_dir = emis_dir / "bafu"
out_dir = emis_dir / "exports"
mapluft_dir = emis_dir / "mapluft"  
duckdbs_dir = emis_dir / "mapluft" / "duckdbs"

# Choose here which data file to use
# Duck db is the new version, it contains multiple years in one file
# (our current version only 2023 and 2024)
#inv_file = mapluft_dir / f"mapLuft_{YEAR}_v2024.gdb"
inv_file = duckdbs_dir / f"emikat_v2026a.db"

# GRAL grid to which inventory is exported
gral_dir = Path("/cluster/scratch/dbrunner/gg_run_dir/GRAL_00001")
GRAL_GRID = GralGrid.from_gral_rundir(gral_dir)

# CRS of the output, can be WGS84 or LV95
OUTPUT_CRS = GRAL_GRID.crs

# Edge of the raster cells (in meters) for our GRAL raster
RASTER_DX = GRAL_GRID.dx
RASTER_DY = GRAL_GRID.dy

VERSION = "v3.2"

# File with the data required for the human respiration
quartier_anlyse_dir = emis_dir / "quartieranalyse" / "v202605"
quartier_anlyse_file = quartier_anlyse_dir / "Quartieranalyse_-OGD.gpkg"

# the following is the unit of the output, should be kg/h
output_unit = Units.KG_PER_HOUR

# %% Check some parameters and create the output directory
weights_dir = out_dir / f"weights_files_gralgrid_{RASTER_DX}_{RASTER_DY}_{YEAR}_{VERSION}_crs{OUTPUT_CRS}"
weights_dir.mkdir(exist_ok=True, parents=True)

# %% load the zurich inventory
if inv_file.suffix == ".gdb":
    inv_zh = MapLuftZurich(inv_file,substances=substances)

    ignore_categories = ['c5801_BrandFeuerschaeden_Emissionen',
                         'c5601_Feuerwerke_Emissionen',
                         'c5701_Tabakwaren_Emissionen']

else:
    # for duckdb, we need to provide the substances in lower case, 
    # as they are stored in lower case in the database
    inv_zh = DuckDBInventory(inv_file, year=YEAR,
                            substances=[str(sub).lower() for sub in substances])

    ignore_categories = ['brandfeuerschaden',
                         'feuerwerk',
                         'tabakwaren']

    ZH_CO2_Groups = ZH_CO2_DUCK_Groups

# remove the ignore categories from the inventory
inv_zh = drop(inv_zh,categories=ignore_categories)

# check if all categories except for ignore_categories are matched with ZH_CO2_Group
#check_categories = [category for category in inv.categories 
#                    if category not in ignore_categories]

validate_group(ZH_CO2_Groups, inv_zh.categories)


# %% add the swiss inventory outside Zurich

def load_zurich_shape(
    zh_raw_file= emis_dir / "Zurich_borders.txt",
    crs_file: int = WGS84,
    crs_out: int = LV95,
) -> Polygon:
    with open(zh_raw_file, "r") as f:
        points_list = eval(f.read())
        zh_poly = Polygon(points_list[0])
        zh_poly_df = gpd.GeoDataFrame(geometry=[zh_poly], crs=crs_file).to_crs(crs_out)
        zh_poly = zh_poly_df.geometry.iloc[0]
        return zh_poly


swiss_dir = emis_dir / "bafu"
# The csv file contains the total emissions for all categories and substances,
# used to calculate the scaling factors for the Swiss inventory outside of zurich
# The file was generated with Corina's script dict_process_grid.py in a version that
# allows splitting traffic into 4 subcategories (f1: cars, f2: light duty, f3: heavy duty, f4: two wheels)
filepath_csv_totals = swiss_dir / "CH_emissions_2020_2022_2024_CO2.csv"
inv_ch = SwissRasters(
    filepath_csv_totals=filepath_csv_totals,
    filepath_point_sources=swiss_dir / "swissprtr-daten-2007-2024.xlsx",
    rasters_dir=swiss_dir / "ekat_gridascii",
    rasters_str_dir=swiss_dir / "ekat_str_gridascii",
    requires_grid=True,
    year=YEAR,
    substances=["CO2", "CO2_biog"],
)
# crop inventory to GRAL domain to speed-up subsequent processing
inv_ch = clip_box(inv_ch, GRAL_GRID.xmin, GRAL_GRID.ymin, 
                  GRAL_GRID.xmax, GRAL_GRID.ymax)

print('CH inventory successfully loaded')

# since we do not differentiate between biogenic and anthropogenic CO2
# at this stage, we merge the biogenic and fossi contributions
#merge_substances(inv_ch, {"CO2_bio": ["CO2_biog"]}, inplace=True)
#merge_substances(inv_ch, {"CO2_fos": ["CO2"]}, inplace=True)

merge_substances(inv_ch, {"CO2": ["CO2", "CO2_biog"]}, inplace=True)

# check whether all required groups are available and apply grouping
validate_group(CH_CO2_Groups, inv_ch.categories)

ch_outside_zh = crop_with_shape(
    inv_ch,
    load_zurich_shape(),
    keep_outside=True,
    modify_grid=False,
    #weight_file=weights_dir / "ch_out_zh",
)

# add the scaled Swiss inventory to the Zurich inventory
inv_combined = add_inventories(inv_zh, ch_outside_zh)

# %% Add the human respiration
if ADD_HUMAN_RESPIRATION:

    # Load the data. It is available for the whole canton of Zurich,
    # which covers the whole grid of the output
    df_quartier = load_data_from_quartieranalyse(quartier_anlyse_file,
                                                 GRAL_GRID)

    # Load into an emiproc Inventory
    raw_resp_inv = people_to_emissions(
        df_quartier,
        # Assumes people spend 60% of their time at home and 40% at work
        time_ratios={"people_living": 0.6, "people_working": 0.4},
        emission_factor={
            ("people_living", "CO2"): EmissionFactor.ROUGH_ESTIMATON,
            ("people_working", "CO2"): EmissionFactor.ROUGH_ESTIMATON
        },
    )

    # Group the categories
    resp_inv = raw_resp_inv.copy()
    resp_inv = group_categories(
            resp_inv,
            {
                "AtmungZuhause": ["people_living"], 
                "AtmungArbeit": ["people_working"],
            },
        )
    
    # We may crop human respiration to the zurich shape but we don't
    # resp_inv = crop_with_shape(resp_inv, load_zurich_shape())

    #resp_inv = gdf_to_gdfs(resp_inv)

    #inv_combined = add_inventories(inv_combined, resp_inv)

# %% check point
from emiproc.inventories.zurich.categories_info import ZURICH_DUCKDB_SOURCES, ZURICH_CH_SOURCES

# combined
ZH_COMBINED_SOURCES = {**ZURICH_DUCKDB_SOURCES, **ZURICH_CH_SOURCES}
ZH_RESP= {"AtmungZuhause": ["AtmungZuhause"], 
        "AtmungArbeit": ["AtmungArbeit"]}
ZH_COMBINED_GROUPS = {**ZH_CO2_DUCK_Groups, **CH_CO2_Groups, **ZH_RESP}

def add_sg_info(cat_info, cat_groups):

    # source group numbers according to UGZ report
    source_groups = {
        "KHKW": 14,
        "Industrie": 15,
        "FeuerungenFossil": 11,
        "FeuerungenBio": 12,
        "Strassenverkehr": 6,
        "Schwerverkehr": 7,
        "OeffentlicherVerkehr": 9,
        "Schiffahrt": 1,
        "FahrzeugeMaschinen": 17,
        "Umschwung": 18,
        "FeuerungenFossil_CH": 35,
        "FeuerungenBio_CH": 36,
        "Verkehr_CH": 34,
        "Rest_CH": 37,
        "AtmungZuhause": 61,
        "AtmungArbeit": 62,
    }

    new_cat = {}

    for cat in cat_info:
        for grp, members in cat_groups.items():
            if cat in members:
                new_cat[cat] = cat_info[cat]
                new_cat[cat].source_group = source_groups[grp]
                break

    return new_cat

inv_zh.emission_infos = add_sg_info(ZH_COMBINED_SOURCES,ZH_COMBINED_GROUPS)

# %%

export_to_gral(
    inv_zh,
    GRAL_GRID,
    out_dir / 'test'
)
# %%
