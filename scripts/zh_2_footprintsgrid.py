"""Script used for creating raster data of zurich for the ICOS-cities project.

The idea of this scripts is to produce raster files for zurich.

It is possible put the rasters inside the swiss inventory as well.
"""

# %%
from datetime import datetime
from enum import Enum
from math import floor
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point, Polygon

from emiproc.exports.netcdf import nc_cf_attributes
from emiproc.grids import LV95, WGS84, RegularGrid, SwissGrid
from emiproc.inventories.swiss import SwissRasters
from emiproc.inventories.utils import (
    add_inventories,
    crop_with_shape,
    drop,
    get_total_emissions,
    scale_inventory,
    group_categories,
)
from emiproc.inventories.zurich import MapLuftZurich
from emiproc.regrid import remap_inventory
from emiproc.speciation import merge_substances, speciate, speciate_inventory
from emiproc.utilities import SEC_PER_YR
from emiproc.exports.rasters import export_raster_netcdf
from emiproc.exports.netcdf import nc_cf_attributes
from emiproc.plots import plot_inventory

# %% define some parameters for the output


YEAR = 2022

INCLUDE_SWISS_OUTSIDE = True
swiss_data_path = Path(
    r"C:\Users\coli\Documents\ZH-CH-emission\Data\CHEmissionen\CH_Emissions_2015_2020_2022_CO2_CO2biog_CH4_N2O_BC_AP.xlsx"
)
footprints_dir = Path(r"C:\Users\coli\Documents\Data\footprints")
mapluft_dir = Path(r"C:\Users\coli\Documents\Data\mapluft_emissionnen_kanton")
mapluf_file = mapluft_dir / f"mapLuft_{YEAR}_v2024.gdb"

# edge of the raster cells
VERSION = "v4"

# Whether to split the biogenic CO2 and the antoropogenic CO2
SPLIT_BIOGENIC_CO2 = True

# Whether to add the human respiration
ADD_HUMAN_RESPIRATION = True
# File with the data required for the human respiration
quartier_anlyse_file = r"C:\Users\coli\Documents\emiproc_cases\cases\parks_polygons\Quartieranalyse_-OGD\Quartieranalyse_-OGD.gpkg"


# Whether to group categories to the GNRF categories
USE_GNRF = True


# Whether to split the F category of the GNRF into 4 subcategories for accounting
# for the different vehicle types (cars, light duty, heavy duty, two wheels)
SPLIT_GNRF_ROAD_TRANSPORT = True


# %% Check some parameters and create the output directory
weights_dir = footprints_dir / f"weights_files_{YEAR}_{VERSION}"
weights_dir.mkdir(exist_ok=True)

if SPLIT_GNRF_ROAD_TRANSPORT and not USE_GNRF:
    raise ValueError("Cannot split GNRF if not using GNRF")

if INCLUDE_SWISS_OUTSIDE:
    # Need to have the same categories between swiss and zurich
    assert USE_GNRF


# %% load the zurich inventory
inv = MapLuftZurich(mapluf_file)


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


zh_shape = load_zurich_shape()

x_min, y_min, x_max, y_max = zh_shape.bounds

# %% load the grid

footprint_file = footprints_dir / "zurich_footprint_220808_correct.nc"
ds = xr.open_dataset(footprint_file)

measurement_coordinates = 2680911.322, 1248390.798

# Rename the indexes
ds["x_lv95"] = ds.x + measurement_coordinates[0]
ds["y_lv95"] = ds.y + measurement_coordinates[1]
# Assing x and y as dimensions
ds = ds.set_coords(["x_lv95", "y_lv95"])

# Create a new dataset with the new coordinates

# Fromat the integer timestep to a datetime object
# Format is yymmddhhMM
datetime = pd.to_datetime(ds["timestep"].values, format="%y%m%d%H%M")
ds = ds.assign_coords(datetime=("timestep", datetime))
# Add georeference to the dataset
# ds = ds.rio.write_crs("LV95").rio.set_spatial_dims(x_dim="x", y_dim="y")


# Get only the cells of the grid where footprint is greater than 0
x_coords = ds.x_lv95.values
y_coords = ds.y_lv95.values
d2 = 5.0  # meters, half the size of the cells
d_out = 10.0  # meters, size of the cells of the output grid
grid = RegularGrid(
    xmin=x_coords.min() - d2,
    xmax=x_coords.max() + d2,
    ymin=y_coords.min() - d2,
    ymax=y_coords.max() + d2,
    dx=d_out,
    dy=d_out,
    crs="LV95",
)


# %% Split the biogenic CO2

if SPLIT_BIOGENIC_CO2:
    from emiproc.inventories.zurich.speciation_co2_bio import ZH_CO2_BIO_RATIOS

    inv = speciate(inv, "CO2", ZH_CO2_BIO_RATIOS, drop=True)


# %% change the categories
if USE_GNRF:

    from emiproc.inventories.zurich.gnrf_groups import ZH_2_GNFR

    if SPLIT_GNRF_ROAD_TRANSPORT:
        ZH_2_GNFR = ZH_2_GNFR.copy()
        # Remove the road transport from the GNRF
        ZH_2_GNFR.pop("GNFR_F")
        splitted_cats = {
            "GNFR_F-cars": [
                "c1301_Personenwagen_Emissionen_Kanton",
                "c1306_StartStopTankatmung_Emissionen_Kanton",
            ],
            "GNFR_F-light_duty": [
                "c1307_Lieferwagen_Emissionen_Kanton",
            ],
            "GNFR_F-heavy_duty": [
                "c1302_Lastwagen_Emissionen_Kanton",
                "c1304_Linienbusse_Emissionen_Kanton",
                "c1305_Trolleybusse_Emissionen_Kanton",
                "c1308_Reisebusse_Emissionen_Kanton",
            ],
            "GNFR_F-two_wheels": [
                "c1303_Motorraeder_Emissionen_Kanton",
            ],
        }
        # add this to the mapping
        ZH_2_GNFR |= splitted_cats

    inv = group_categories(inv, ZH_2_GNFR)

# %% add the swiss inventory when needed
if INCLUDE_SWISS_OUTSIDE:
    inv_ch = SwissRasters(
        filepath_csv_totals=r"C:\Users\coli\Documents\emissions_preprocessing\output\CH_emissions_EMIS-Daten_1990-2050_Submission_2024_N2O_PM25_NH3_NOx_SO2_PM10_CH4_CO2_biog_CO2_CO.csv",
        filepath_point_sources=r"C:\Users\coli\Documents\emissions_preprocessing\input\SwissPRTR-Daten_2007-2022.xlsx",
        rasters_dir=swiss_data_path.parent / "ekat_gridascii_v_swiss2icon",
        rasters_str_dir=swiss_data_path.parent / "ekat_str_gridascii_v_footprints",
        requires_grid=True,
        # requires_grid=False,
        year=YEAR,
    )
    inv_ch = drop(inv_ch, categories=["na"])
    merge_substances(inv_ch, {"CO2_bio": ["CO2_biog"]}, inplace=True)
    merge_substances(inv_ch, {"CO2_fos": ["CO2"]}, inplace=True)
    inv_ch.history.append(
        "the map of CO2 for evstr was used for BC and CO2-bio as they did not exist"
    )

    from emiproc.inventories.categories_groups import CH_2_GNFR

    # These categories are not in the invenotry here, because we don't care about them
    missing_cats = ["eilgk", "evklm", "evtrk", "enwal", "eipwp"]
    # Remove the missing categories
    our_CH_2_GNFR = {
        new_cat: [c for c in cats if c not in missing_cats]
        for new_cat, cats in CH_2_GNFR.items()
    }
    groupped_ch = group_categories(inv_ch, our_CH_2_GNFR)

    if SPLIT_GNRF_ROAD_TRANSPORT:
        # Calculate splitting ratios in zurich
        total_emisson = get_total_emissions(inv)
        speciation_dict = {}
        for sub, cat_dic in total_emisson.items():
            # Get the categories of the GNRF-F
            f_cat_dict = {
                cat: val for cat, val in cat_dic.items() if cat.startswith("GNFR_F")
            }
            # Get the total of the GNRF-F
            f_total = sum(f_cat_dict.values())
            # Calculate the ratios
            catsub = ("GNFR_F", sub)
            if catsub in groupped_ch._gdf_columns:
                speciation_dict[catsub] = {
                    (cat, sub): val / f_total for cat, val in f_cat_dict.items()
                }

        groupped_ch = speciate_inventory(groupped_ch, speciation_dict, drop=True)

    ch_outside_zh = crop_with_shape(
        groupped_ch,
        zh_shape,
        keep_outside=True,
        modify_grid=False,
        weight_file=weights_dir / "ch_out_zh",
    )
    ch_inside_zh = crop_with_shape(
        groupped_ch,
        zh_shape,
        keep_outside=False,
        modify_grid=False,
        weight_file=weights_dir / "ch_in_zh",
    )


# %%
if INCLUDE_SWISS_OUTSIDE:
    remapped_ch_out = remap_inventory(
        ch_outside_zh,
        grid,
        weights_file=(weights_dir / f"swiss_around_zh_{d_out}x{d_out}"),
    )
# %% do the actual remapping of zurich to rasters

rasters_inv = remap_inventory(
    crop_with_shape(inv, zh_shape),
    grid,
    weights_file=weights_dir / f"{mapluf_file.stem}_weights_{d_out}x{d_out}",
)

# %% Rescale the swiss and add it, the scaling is made such that the
# mapluft inventory is not changed and the total swiss inventory is also not changed
# so we only scale the region outside of zurich to compensate
if INCLUDE_SWISS_OUTSIDE:
    # get the total inside zurich from mapluft
    mapluft_total = get_total_emissions(rasters_inv)
    # get the total inside zurich from swiss inv
    swiss_out_total = get_total_emissions(ch_outside_zh)
    swiss_total = get_total_emissions(groupped_ch)
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
    rescaled_ch = scale_inventory(remapped_ch_out, scaling_factors)
    # add the inventories
    rasters_inv = add_inventories(rasters_inv, rescaled_ch)

# %% Add the human respiration
if ADD_HUMAN_RESPIRATION:

    from emiproc.human_respiration import (
        load_data_from_quartieranalyse,
        people_to_emissions,
        EmissionFactor,
    )

    # Load the data. It is available for the whole Kanton of zurich,
    # which covers the whole grid of the output
    df_quariter = load_data_from_quartieranalyse(quartier_anlyse_file)
    # Load into an emiproc Inventory
    raw_resp_inv = people_to_emissions(
        df_quariter,
        # Assumes people spend 60% of their time at home and 40% at work
        time_ratios={"people_living": 0.6, "people_working": 0.4},
        emission_factor={
            ("people_living", "CO2_bio"): EmissionFactor.ROUGH_ESTIMATON,
            ("people_working", "CO2_bio"): EmissionFactor.ROUGH_ESTIMATON,
            ("people_living", "N2O"): EmissionFactor.N2O_MITSUI_ET_ALL,
            ("people_working", "N2O"): EmissionFactor.N2O_MITSUI_ET_ALL,
            ("people_living", "CH4"): EmissionFactor.CH4_POLAG_KEPPLER,
            ("people_working", "CH4"): EmissionFactor.CH4_POLAG_KEPPLER,
        },
    )

    # Group the categories
    resp_inv = group_categories(
        raw_resp_inv,
        {
            "GNFR_O-home": ["people_living"],
            "GNFR_O-work": ["people_working"],
        },
    )
    # If keep inside, crop the inventory to the zurich shape
    if not INCLUDE_SWISS_OUTSIDE:
        resp_inv = crop_with_shape(resp_inv, zh_shape)

    # Remap the inventory to the raster
    remapped_resp = remap_inventory(
        resp_inv,
        grid,
        weights_file=weights_dir
        / f"resp_weights_{INCLUDE_SWISS_OUTSIDE}_{d_out}x{d_out}",
    )

    rasters_inv = add_inventories(rasters_inv, remapped_resp)
# %% Populate the dataframe of the output


export_raster_netcdf(
    rasters_inv,
    footprints_dir / f"inventory_for_footprints_{VERSION}.nc",
    grid=grid,
    netcdf_attributes=nc_cf_attributes(
        author="Lionel Constantin",
        contact="lionel.constantin@empa.ch",
        title=f"Zurich {YEAR} inventory combined for footprint analysis",
        source="MapLuft and Swiss inventory",
        additional_attributes={"version": VERSION},
    ),
)


# %%
plot_inventory(
    rasters_inv,
    # out_dir=footprints_dir / "plots",
    total_only=True,
)
# %%
