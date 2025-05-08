"""Script used for creating raster data of zurich for the ICOS-cities project.

The idea of this scripts is to produce raster files for zurich.

It is possible put the rasters inside the swiss inventory as well.
"""

# %%
# autoreload modules in interactive python
%load_ext autoreload
%autoreload 2
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
from shapely.ops import transform
from shapely.geometry import Point, Polygon, LineString
from pyproj import Transformer
import matplotlib.pyplot as plt

from emiproc.plots import plot_inventory
from emiproc.exports.netcdf import nc_cf_attributes
from emiproc.grids import LV95, WGS84, SwissGrid, RegularGrid
from emiproc.inventories.swiss import SwissRasters
from emiproc.inventories.utils import (
    add_inventories,
    crop_with_shape,
    drop,
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
from emiproc.regrid import remap_inventory
from emiproc.speciation import merge_substances, speciate, speciate_inventory
from emiproc.utilities import SEC_PER_YR
from emiproc.exports.netcdf import nc_cf_attributes
from emiproc.exports.rasters import export_raster_netcdf
from emiproc.utilities import Units

# %% define some parameters for the output


YEAR = 2020


INCLUDE_SWISS_OUTSIDE = True
swiss_data_path = Path(
    r"C:\Users\coli\Documents\ZH-CH-emission\Data\CHEmissionen\CH_Emissions_2015_2020_2022_CO2_CO2biog_CH4_N2O_BC_AP.xlsx"
)
outdir = Path(r"C:\Users\coli\Documents\ZH-CH-emission\output_files\mapluft_rasters")
mapluft_dir = Path(r"C:\Users\coli\Documents\Data\mapluft_emissionnen_kanton")
mapluf_file = mapluft_dir / f"mapLuft_{YEAR}_v2024.gdb"

# CRS of the output, can be WGS84 or LV95
OUTPUT_CRS = WGS84
# edge of the raster cells (in meters)
RASTER_EDGE = 100


VERSION = "v2.2"

# Whether to split the biogenic CO2 and the antoropogenic CO2
SPLIT_BIOGENIC_CO2 = False

# Whether to add the human respiration
ADD_HUMAN_RESPIRATION = True
# File with the data required for the human respiration
quartier_anlyse_dir = Path(
    r"C:\Users\coli\Documents\Data\Quartieranalyse_zurich"
) / str(YEAR)
quartier_anlyse_file = quartier_anlyse_dir / "Quartieranalyse_-OGD.gpkg"


output_unit = Units.KG_PER_YEAR

# Whether to group categories to the GNRF categories
USE_GNRF = True


# Whether to split the F category of the GNRF into 4 subcategories for accounting
# for the different vehicle types (cars, light duty, heavy duty, two wheels)
SPLIT_GNRF_ROAD_TRANSPORT = True


# %% Check some parameters and create the output directory
weights_dir = outdir / f"weights_files_{RASTER_EDGE}_{YEAR}_{VERSION}_crs{OUTPUT_CRS}"

if SPLIT_GNRF_ROAD_TRANSPORT and not USE_GNRF:
    raise ValueError("Cannot split GNRF if not using GNRF")

if INCLUDE_SWISS_OUTSIDE:
    # Swiss inventory code works only if the raster is the same as the swiss raster (100 m )
    assert RASTER_EDGE == 100
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


# %% create the zurich swiss grid

OUTPUT_CRS = WGS84
zh_shape = load_zurich_shape()
x_min, y_min, x_max, y_max = zh_shape.bounds

if OUTPUT_CRS == LV95:
    dx, dy = RASTER_EDGE, RASTER_EDGE

elif OUTPUT_CRS == WGS84:
    transformer = Transformer.from_crs(LV95, WGS84, always_xy=True)
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    line = LineString([(x_mid, y_mid), (x_mid + RASTER_EDGE, y_mid + RASTER_EDGE)])
    coords = transform(transformer.transform, line).xy
    # First calcualate the edges in WGS84
    dx = abs(coords[0][1] - coords[0][0])
    dy = abs(coords[1][0] - coords[1][1])

    dx = round(dx, 5)
    dy = round(dy, 5)

    # But the zurich border also
    zh_shape = transform(transformer.transform, zh_shape)
    x_min, y_min, x_max, y_max = zh_shape.bounds

else:
    raise ValueError("Output CRS not supported")

# Round the min to be a multiple of the dx
x_min = dx * floor(x_min / dx)
y_min = dy * floor(y_min / dy)
grid = RegularGrid(
    xmin=x_min,
    ymin=y_min,
    xmax=x_max,
    ymax=y_max,
    dx=dx,
    dy=dy,
    crs=OUTPUT_CRS,
    name="Zurich",
)
grid
# %% Split the biogenic CO2

if SPLIT_BIOGENIC_CO2:
    from emiproc.inventories.zurich.speciation_co2_bio import ZH_CO2_BIO_RATIOS

    inv = speciate(inv, "CO2", ZH_CO2_BIO_RATIOS, drop=True)

# %% do the actual remapping of zurich to rasters

zh_shape = load_zurich_shape()
zh_cropped = crop_with_shape(inv, zh_shape)
zh_cropped.to_crs(OUTPUT_CRS)
rasters_inv = remap_inventory(
    zh_cropped,
    grid,
    weights_file=weights_dir / f"{mapluf_file.stem}_weights",
)


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

    rasters_inv = group_categories(rasters_inv, ZH_2_GNFR)

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

    if not SPLIT_BIOGENIC_CO2:
        merge_substances(inv_ch, {"CO2": ["CO2_fos", "CO2_bio"]}, inplace=True)

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
        total_emisson = get_total_emissions(rasters_inv)
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
    ch_outside_zh.to_crs(OUTPUT_CRS)
    remapped_ch_out = remap_inventory(
        ch_outside_zh,
        grid,
        # weights_file=(weights_dir / f"swiss_around_zh_2_{RASTER_EDGE}x{RASTER_EDGE}"),
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
    inv_zh = rasters_inv
    rasters_inv = add_inventories(rasters_inv, rescaled_ch)

# %% Add the human respiration
if ADD_HUMAN_RESPIRATION:

    # Load the data. It is available for the whole Kanton of zurich,
    # which covers the whole grid of the output
    df_quariter = load_data_from_quartieranalyse(quartier_anlyse_file)
    # Load into an emiproc Inventory
    co2_hr_name = "CO2_bio" if SPLIT_BIOGENIC_CO2 else "CO2"
    raw_resp_inv = people_to_emissions(
        df_quariter,
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
    resp_inv = group_categories(
        raw_resp_inv,
        {
            "GNFR_O": ["people_living", "people_working"],
        },
    )
    # If keep inside, crop the inventory to the zurich shape
    if not INCLUDE_SWISS_OUTSIDE:
        resp_inv = crop_with_shape(resp_inv, zh_shape)

    # Remap the inventory to the raster
    resp_inv.to_crs(OUTPUT_CRS)
    remapped_resp = remap_inventory(
        resp_inv,
        grid,
        weights_file=weights_dir / f"resp_weights_{INCLUDE_SWISS_OUTSIDE}",
    )

    rasters_inv = add_inventories(rasters_inv, remapped_resp)
# %% Populate the dataframe of the output

rasters_inv.year = YEAR
out_path = export_raster_netcdf(
    rasters_inv,
    outdir
    /  "_".join(
        [
            "zurich",
            "inside_swiss" if INCLUDE_SWISS_OUTSIDE else "cropped",
            "Fsplit" if SPLIT_GNRF_ROAD_TRANSPORT else "",
            f"{RASTER_EDGE}x{RASTER_EDGE}",
            mapluf_file.stem,
            VERSION,
            f"crs{OUTPUT_CRS}",
            "rasters.nc",
        ]
    ) ,
    unit=output_unit,
    group_categories=True,
    netcdf_attributes=nc_cf_attributes(
        author="Lionel Constantin, Empa",
        contact="dominik.brunner@empa.ch",
        title=(
            "Annual mean emissions of CO2 of the city of Zurich (only emissions within"
            " the political borders of the city)"
        ),
        source="https://www.stadt-zuerich.ch/gud/de/index/umwelt_energie/luftqualitaet/schadstoffquellen/emissionskataster.html",
        comment="Created for use in the EU project ICOS-Cities",
        history=(
            "Created from original GIS inventory mapLuft of the city of Zurich by"
            " rasterizing all point, line and area sources"
        ),
        additional_attributes={
            "swiss_coordinate_system_lv95": "https://www.swisstopo.admin.ch/en/knowledge-facts/surveying-geodesy/coordinates/swiss-coordinates.html",
            "comment_lv95": (
                "In original LV95 system, x denote northings and y eastings. They have"
                " been exchanged here for better compatibility with lon/lat."
            ),
            "copyright_notice": "",
            "script_version": VERSION,
            "emiproc_history": str(rasters_inv.history),
        },
    ),
    categories_description={
        "GNFR_A": "Public Power",
        "GNFR_B": "Industry",
        "GNFR_C": "Other Stationary Combustion",
        "GNFR_D": "Fugitives",
        "GNFR_E": "Solvents",
        "GNFR_F": "Road Transport",
        "GNFR_F-cars": "Road Transport - Cars",
        "GNFR_F-light_duty": "Road Transport - Light Duty Vehicules",
        "GNFR_F-heavy_duty": "Road Transport - Heavy Duty Vehicules",
        "GNFR_F-two_wheels": "Road Transport - Two Wheels Vehicles",
        "GNFR_G": "Shipping",
        "GNFR_H": "Aviation",
        "GNFR_I": "OffRoad",
        "GNFR_J": "Waste",
        "GNFR_K": "Agriculture Livestock",
        "GNFR_L": "Agriculture Other",
        "GNFR_O": "Human Respiration",
        "GNFR_R": "Others",
    }
)
print(f"Output written to {out_path}")

# %%
plt.style.use("default")

plots_dir = Path(r"C:\Users\coli\Pictures\emiproc\zurich_rasters_for_presentations")

for inv, name in zip([rescaled_ch, inv_zh, rasters_inv], ["ch", "zh", "combined"]):
    out_dir = plots_dir / weights_dir.name / name
    out_dir.mkdir(exist_ok=True, parents=True)
    plot_inventory(inv, vmin=0.01, vmax=100, out_dir=out_dir, total_only=True)

# %%
