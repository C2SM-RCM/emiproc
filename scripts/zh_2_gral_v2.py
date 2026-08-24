"""Prepare Zurich city CO2 inventory for GRAL

The script combines the following inventories:
- MapLuft Zurich inventory (the City's inventory)
- Swiss inventory (from the Swiss Federal Office for the Environment)
- Human respiration inventory (from the Quartieranalyse data)

and writes it out to GRAL format with point, line and area sources.

The GRAL grid definition is read from a GRAL rundir using the pygg package, 
which is specified in the GRAL_GRID variable.
"""
# %%
# autoreload modules in interactive python
#%load_ext autoreload
#%autoreload 2
# %%

from pathlib import Path
import logging

import geopandas as gpd
from shapely.geometry import Polygon

from emiproc.exports.gral import export_to_gral
from emiproc.grids import LV95, WGS84
from emiproc.inventories.swiss import SwissRasters
from emiproc.inventories.utils import (
    add_inventories,
    crop_with_shape,
    clip_box,
    get_total_emissions,
    scale_inventory,
    group_categories,
)
from emiproc.human_respiration import (
    load_data_from_quartieranalyse,
    people_to_emissions,
    EmissionFactor,
)
from emiproc.inventories.zurich import MapLuftZurich
from emiproc.inventories.zurich.gral_groups import ZH_CO2_Groups

from emiproc.regrid import remap_inventory
from emiproc.speciation import merge_substances, speciate
from emiproc.utilities import Units
from emiproc.inventories.zurich.duck import DuckDBInventory
from pygg.grids import GralGrid

# %% define some parameters for the output

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# define year for which to generate emissions
# Note that the Swiss inventory may not yet be available for this year, 
# in which case the last available year will be used
YEAR = 2024

# Whether to include the Swiss inventory outside of Zurich in the output
INCLUDE_SWISS_OUTSIDE = True

# Whether to split the biogenic and antropogenic CO2
SPLIT_BIOGENIC_CO2 = False

# Whether to add the human respiration
ADD_HUMAN_RESPIRATION = True

# Paths to data
emis_dir = Path("/cluster/home/dbrunner/data/emissions")
swiss_dir = emis_dir / "bafu"
outdir = emis_dir / "exports"
mapluft_dir = emis_dir / "mapluft"  
duckdbs_dir = emis_dir / "mapluft" / "duckdbs"

# Choose here which data file to use
# Duck db is the new version, it contains all years in one file
#inv_file = mapluft_dir / f"mapLuft_{YEAR}_v2024.gdb"
inv_file = duckdbs_dir / f"emikat_v2026a.db"


# GRAL grid to which inventory is exported
gral_dir = Path("/cluster/scratch/dbrunner/gg_run_dir/GRAL_00001")
GRAL_GRID = GralGrid.from_gral_rundir(gral_dir)

# CRS of the output, can be WGS84 or LV95
#OUTPUT_CRS = LV95
OUTPUT_CRS = GRAL_GRID.crs

# edge of the raster cells (in meters) for our GRAL raster
RASTER_DX = GRAL_GRID.dx
RASTER_DY = GRAL_GRID.dy

VERSION = "v3.2"

# File with the data required for the human respiration
quartier_anlyse_dir = emis_dir / "quartieranalyse" / "v202605"
quartier_anlyse_file = quartier_anlyse_dir / "Quartieranalyse_-OGD.gpkg"

# the following is the unit of the output, should be kg/h
output_unit = Units.KG_PER_HOUR

# Whether to group categories to the GNRF categories
USE_GNRF = True

# Whether to split the F category of the GNRF into 4 subcategories for accounting
# for the different vehicle types (cars, light duty, heavy duty, two wheels)
SPLIT_GNRF_ROAD_TRANSPORT = True

# %% Check some parameters and create the output directory
weights_dir = outdir / f"weights_files_gralgrid_{RASTER_DX}_{RASTER_DY}_{YEAR}_{VERSION}_crs{OUTPUT_CRS}"
weights_dir.mkdir(exist_ok=True, parents=True)

if SPLIT_GNRF_ROAD_TRANSPORT and not USE_GNRF:
    raise ValueError("Cannot split GNRF if not using GNRF")

if INCLUDE_SWISS_OUTSIDE:
    # Swiss inventory code works only if the raster is the same as the swiss raster (100 m )
    #assert RASTER_EDGE == 100
    # Need to have the same categories between swiss and zurich
    assert USE_GNRF

# %% load the zurich inventory
if inv_file.suffix == ".gdb":
    inv = MapLuftZurich(inv_file,substances=["CO2"])
else:
    # for duckdb, we need to provide the substances in lower case, 
    # as they are stored in lower case in the database
    inv = DuckDBInventory(inv_file, year=YEAR,substances=['co2'])
    # Convert substance names to upper case
    rename_dict: dict[str, list[str]] = {
        str(sub).upper(): [str(sub)]
        for sub in inv.substances
        if str(sub).lower() != "nox"
    }
    #rename_dict["NOx"] = ["nox"]
    inv = merge_substances(inv, rename_dict)

# %%
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

# %% Split the biogenic CO2

if SPLIT_BIOGENIC_CO2:
    from emiproc.inventories.zurich.speciation_co2_bio import ZH_CO2_BIO_RATIOS
    inv = speciate(inv, "CO2", ZH_CO2_BIO_RATIOS, drop=True)

# %% do the actual remapping of zurich to rasters

# crop Zurich inventory to the Zurich domain borders (why necessary?)
zh_cropped = crop_with_shape(inv, load_zurich_shape())
zh_cropped.to_crs(OUTPUT_CRS)


# %% change the categories
if USE_GNRF:

    from emiproc.inventories.zurich.gnrf_groups import ZH_2_GNFR, ZH_DUCK_2_GNFR 

    if isinstance(inv, DuckDBInventory):
        ZH_2_GNFR = ZH_DUCK_2_GNFR

    if SPLIT_GNRF_ROAD_TRANSPORT:

        # Remove the road transport from the GNRF
        ZH_2_GNFR = ZH_2_GNFR.copy()
        ZH_2_GNFR.pop("GNFR_F")
        if isinstance(inv, DuckDBInventory):
            # Split the road transport into 4 subcategories
            splitted_cats = {
                "GNFR_F-cars": [
                    "personenwagen",
                    "startstoptankatmung",
                ],
                "GNFR_F-light_duty": [
                    "lieferwagen",
                ],
                "GNFR_F-heavy_duty": [
                    "lastwagen",
                    "linienbus",
                    "trolleybus",
                    "reisebus",
                ],
                "GNFR_F-two_wheels": [
                    "motorräder",
                ],
            }
        else:
            # Split the road transport into 4 subcategories
            splitted_cats = {
                "GNFR_F-cars": [
                    "c1301_Personenwagen_Emissionen",
                    "c1306_StartStopTankatmung_Emissionen",
                ],
                "GNFR_F-light_duty": [
                    "c1307_Lieferwagen_Emissionen",
                    "c1309_Kleinbusse_Emissionen",
                ],
                "GNFR_F-heavy_duty": [
                    "c1302_Lastwagen_Emissionen",
                    "c1304_Linienbusse_Emissionen",
                    "c1305_Trolleybusse_Emissionen",
                    "c1308_Reisebusse_Emissionen",
                ],
                "GNFR_F-two_wheels": [
                    "c1303_Motorraeder_Emissionen",
                    "c1310_Motorraeder_Emissionen",
                ],
            }
        # add this to the mapping
        ZH_2_GNFR |= splitted_cats

    zh_cropped = group_categories(zh_cropped, ZH_2_GNFR, ignore_missing=True)

# %% add the swiss inventory when needed

if INCLUDE_SWISS_OUTSIDE:
    swiss_dir = emis_dir / "bafu"
    # The following file contains the total emissions for all categories and substances,
    # which is used to calculate the scaling factors for the swiss inventory outside of zurich
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

    bounds_polygon = Polygon([
        (GRAL_GRID.bounds[0], GRAL_GRID.bounds[1]),
        (GRAL_GRID.bounds[2], GRAL_GRID.bounds[1]),
        (GRAL_GRID.bounds[2], GRAL_GRID.bounds[3]),
        (GRAL_GRID.bounds[0], GRAL_GRID.bounds[3]),
    ])
    inv_ch = crop_with_shape(inv_ch, bounds_polygon, 
                             keep_outside=False,
                              modify_grid=True)
    
    
    #inv_ch = clip_box(inv_ch, GRAL_GRID.xmin, GRAL_GRID.ymin, 
    #                  GRAL_GRID.xmax, GRAL_GRID.ymax)

    print('CH inventory successfully loaded')

    merge_substances(inv_ch, {"CO2_bio": ["CO2_biog"]}, inplace=True)
    merge_substances(inv_ch, {"CO2_fos": ["CO2"]}, inplace=True)

    if not SPLIT_BIOGENIC_CO2:
        merge_substances(inv_ch, {"CO2": ["CO2_fos", "CO2_bio"]}, inplace=True)

    #inv_ch.history.append(
    #    "the map of CO2 for evstr was used for BC and CO2-bio as they did not exist"
    #)

    from emiproc.inventories.categories_groups import CH_2_GNFR

    categories_available = set(inv_ch.categories)
    # Remove the missing categories
    our_CH_2_GNFR = {
        new_cat: [c for c in cats if c in categories_available]
        for new_cat, cats in CH_2_GNFR.items()
    }

    if SPLIT_GNRF_ROAD_TRANSPORT:
        # Split the road transport into 4 subcategories
        # Remove the road transport from the GNRF
        our_CH_2_GNFR.pop("GNFR_F")
        splitted_cats = {
                "GNFR_F-cars": ["evstrf1","evzon"],
                "GNFR_F-light_duty": ["evstrf2"],
                "GNFR_F-heavy_duty": ["evstrf3"],
                "GNFR_F-two_wheels": ["evstrf4"],
        }
        # add this to the mapping
        our_CH_2_GNFR |= splitted_cats

    grouped_ch = group_categories(inv_ch, our_CH_2_GNFR)

    ch_outside_zh = crop_with_shape(
        grouped_ch,
        load_zurich_shape(),
        keep_outside=True,
        modify_grid=False,
        #weight_file=weights_dir / "ch_out_zh",
    )
    ch_inside_zh = crop_with_shape(
        grouped_ch,
        load_zurich_shape(),
        keep_outside=False,
        modify_grid=False,
        #weight_file=weights_dir / "ch_in_zh",
    )

# %% Rescale the swiss inventory and add it, the scaling is made such that the
# mapluft inventory is not changed and the total swiss inventory is also not changed
# so we only scale the region outside of zurich to compensate
if INCLUDE_SWISS_OUTSIDE:
    # get the total inside zurich from mapluft
    mapluft_total = get_total_emissions(zh_cropped)
    # get the total inside zurich from swiss inv
    swiss_out_total = get_total_emissions(ch_outside_zh)
    swiss_total = get_total_emissions(grouped_ch)
    # calculates scalings
    scaling_factors = {}
    for sub, cat_dic in swiss_total.items():
        if sub not in mapluft_total.keys():
            continue
        scaling_factors[sub] = {}
        for cat, total in cat_dic.items():
            if cat == "__total__" or cat not in mapluft_total[sub]:
                continue
            # we want scaling_factor * swiss_out + mapluft = swiss_total
            scaling_factor = (total - mapluft_total[sub][cat]) / swiss_out_total[sub][
                cat
            ]
            scaling_factors[sub][cat] = scaling_factor
    # rescale inventory
    rescaled_ch_outside = scale_inventory(ch_outside_zh, scaling_factors)

    # add the scaled Swiss inventory to the Zurich inventory
    inv_zh = add_inventories(zh_cropped, rescaled_ch_outside)
else:
    inv_zh = zh_cropped

# %% Add the human respiration
if ADD_HUMAN_RESPIRATION:

    # Load the data. It is available for the whole Kanton of zurich,
    # which covers the whole grid of the output
    df_quartier = load_data_from_quartieranalyse(quartier_anlyse_file)

    # Load into an emiproc Inventory
    co2_hr_name = "CO2_bio" if SPLIT_BIOGENIC_CO2 else "CO2"
    raw_resp_inv = people_to_emissions(
        df_quartier,
        # Assumes people spend 60% of their time at home and 40% at work
        time_ratios={"people_living": 0.6, "people_working": 0.4},
        emission_factor={
            ("people_living", co2_hr_name): EmissionFactor.ROUGH_ESTIMATON,
            ("people_working", co2_hr_name): EmissionFactor.ROUGH_ESTIMATON,
            ("people_living", "N2O"): EmissionFactor.N2O_MITSUI_ET_ALL,
            ("people_working", "N2O"): EmissionFactor.N2O_MITSUI_ET_ALL,
            ("people_living", "CH4"): EmissionFactor.CH4_POLAG_KEPPLER,
            ("people_working", "CH4"): EmissionFactor.CH4_POLAG_KEPPLER,
        },
    )

    # Group the categories
    resp_inv = raw_resp_inv.copy()
    if USE_GNRF:
        resp_inv = group_categories(
            resp_inv,
            {
                "GNFR_O": ["people_living", "people_working"],
            },
        )
    # If keep inside, crop the inventory to the zurich shape
    if not INCLUDE_SWISS_OUTSIDE:
        resp_inv = crop_with_shape(resp_inv, zh_shape)

    inv_zh = add_inventories(inv_zh, resp_inv)


# %% check point

validate_group(ZH_CO2_Groups, zh_inv.categories)

# TODO: group should also group emission infos
#zh_grouped = group_categories(zh_cropped, ZH_CO2_Groups, ignore_missing=False)

#%%

export_to_gral(
    inv_zh,
    GRAL_GRID,
    outdir,
    polygon_raster_size = 10
)
# %%
