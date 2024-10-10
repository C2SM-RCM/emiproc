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
from shapely.geometry import Point, Polygon

from emiproc.exports.netcdf import nc_cf_attributes
from emiproc.grids import LV95, WGS84, SwissGrid
from emiproc.inventories.swiss import SwissRasters
from emiproc.inventories.utils import (
    add_inventories,
    crop_with_shape,
    get_total_emissions,
    scale_inventory,
    group_categories
)
from emiproc.inventories.zurich import MapLuftZurich
from emiproc.regrid import remap_inventory
from emiproc.speciation import speciate, speciate_inventory
from emiproc.utilities import SEC_PER_YR

# %% define some parameters for the output


YEAR = 2020

INCLUDE_SWISS_OUTSIDE = True
swiss_data_path = Path(
    r"C:\Users\coli\Documents\ZH-CH-emission\Data\CHEmissionen\CH_Emissions_2015_2020_2022_CO2_CO2biog_CH4_N2O_BC_AP.xlsx"
)
outdir = Path(r"C:\Users\coli\Documents\ZH-CH-emission\output_files\mapluft_rasters")
mapluft_dir = Path(r"C:\Users\coli\Documents\Data\mapluft_emissionnen_kanton")
mapluf_file = mapluft_dir / f"mapLuft_{YEAR}_v2024.gdb"

# edge of the raster cells
RASTER_EDGE = 100
VERSION = "v1.8"

# Whether to split the biogenic CO2 and the antoropogenic CO2
SPLIT_BIOGENIC_CO2 = True

# Whether to add the human respiration
ADD_HUMAN_RESPIRATION = True
# File with the data required for the human respiration
quartier_anlyse_dir = Path(r"C:\Users\coli\Documents\Data\Quartieranalyse_zurich") / str(YEAR)
quartier_anlyse_file = quartier_anlyse_dir / "Quartieranalyse_-OGD.gpkg"


# Make a Enum class for output unit choices
class Unit(Enum):
    """Enum class for units"""

    kg_yr = "kg yr-1 cell-1"
    ug_m2_s = "μg m-2 s-1"


output_unit = Unit.kg_yr

# Whether to group categories to the GNRF categories
USE_GNRF = True


# Whether to split the F category of the GNRF into 4 subcategories for accounting
# for the different vehicle types (cars, light duty, heavy duty, two wheels)
SPLIT_GNRF_ROAD_TRANSPORT = True


# %% Check some parameters and create the output directory
weights_dir = outdir / f"weights_files_{RASTER_EDGE}_{YEAR}_{VERSION}"

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


zh_shape = load_zurich_shape()

x_min, y_min, x_max, y_max = zh_shape.bounds

# %% create the zurich swiss grid

d = RASTER_EDGE
xmin, ymin = floor(x_min) // d * d, floor(y_min) // d * d
nx, ny = int(x_max - xmin) // d, int(y_max - ymin) // d
xs = np.arange(xmin, xmin + nx * d, step=d)
ys = np.arange(ymin, ymin + ny * d, step=d)
point_x = np.repeat(xs, ny)
point_y = np.tile(ys, nx)

polys = [
    Polygon(((x, y), (x + d, y), (x + d, y + d), (x, y + d)))
    for y in reversed(ys)
    for x in xs
]
centers = [Point((x + d / 2, y + d / 2)) for y in reversed(ys) for x in xs]
zh_gdf = gpd.GeoDataFrame(geometry=polys, crs=LV95)
gdf_centers = gpd.GeoDataFrame(geometry=centers, crs=LV95)
# %% prepare the output on WGS84
WGS84_gdf = zh_gdf.to_crs(WGS84)
for i in range(4):
    WGS84_gdf[f"coord_{i}"] = WGS84_gdf.geometry.apply(
        lambda poly: poly.exterior.coords[i]
    )

# %% Split the biogenic CO2

if SPLIT_BIOGENIC_CO2:
    from emiproc.inventories.zurich.speciation_co2_bio import ZH_CO2_BIO_RATIOS

    inv = speciate(inv, "CO2", ZH_CO2_BIO_RATIOS, drop=True)

# %% do the actual remapping of zurich to rasters

rasters_inv = remap_inventory(
    crop_with_shape(inv, zh_shape),
    zh_gdf.geometry,
    weigths_file=weights_dir
    / f"{mapluf_file.stem}_weights",
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
        data_path=swiss_data_path,
        rasters_dir=swiss_data_path.parent / "ekat_gridascii",
        rasters_str_dir=swiss_data_path.parent / "ekat_str_gridascii_v17",
        requires_grid=True,
        # requires_grid=False,
        year=YEAR,
        # Specify the compound in the inventory and how they should be named in the output
        dict_spec={
            "NOX": "NOx",
            "PM2.5": "PM25",
            "F-Gase": "F-gases",
            "CO2": "CO2_fos",
            "CO2_biog": "CO2_bio",
        },
    )

    inv_ch.history.append(
        "the map of CO2 for evstr was used for BC and CO2-bio as they did not exist"
    )

    from emiproc.inventories.categories_groups import CH_2_GNFR
    
    # These categories are not in the invenotry here, because we don't care about them
    missing_cats = ['eilgk', 'evklm', 'evtrk', 'enwal']
    # Remove the missing categories
    our_CH_2_GNFR = {
        new_cat: [c for c in cats if c not in missing_cats] for new_cat, cats in CH_2_GNFR.items() 
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
    remapped_ch_out = remap_inventory(
        ch_outside_zh,
        zh_gdf.geometry,
        # weigths_file=(weights_dir / f"swiss_around_zh_2_{RASTER_EDGE}x{RASTER_EDGE}"),
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
            "GNFR_O": ["people_living", "people_working"],
        },
    )
    # If keep inside, crop the inventory to the zurich shape
    if not INCLUDE_SWISS_OUTSIDE:
        resp_inv = crop_with_shape(resp_inv, zh_shape)

    # Remap the inventory to the raster
    remapped_resp = remap_inventory(
        resp_inv,
        zh_gdf.geometry,
        weigths_file=weights_dir / f"resp_weights_{INCLUDE_SWISS_OUTSIDE}",
    )

    rasters_inv = add_inventories(rasters_inv, remapped_resp)
# %% Populate the dataframe of the output


ds_out = xr.Dataset(
    coords={
        "x": (
            "x",
            xs + d / 2,
            {
                "standard_name": "easting",
                "long_name": "easting",
                "units": "m",
                "comment": "center_of_cell",
                "projection": "Swiss coordinate system LV95",
            },
        ),
        "y": (
            "y",
            (ys + d / 2)[::-1],
            {
                "standard_name": "northing",
                "long_name": "northing",
                "units": "m",
                "comment": "center_of_cell",
                "projection": "Swiss coordinate system LV95",
            },
        ),
        "lon": (
            ("y", "x"),
            gdf_centers.to_crs(WGS84)
            .geometry.apply(lambda point: point.coords[0][0])
            .to_numpy()
            .reshape((ny, nx)),
            {
                "standard_name": "longitude",
                "long_name": "longitude",
                "units": "degrees_east",
                "comment": "center_of_cell",
                "bounds": "lon_bnds",
                "projection": "WGS84",
            },
        ),
        "lat": (
            ("y", "x"),
            gdf_centers.to_crs(WGS84)
            .geometry.apply(lambda point: point.coords[0][1])
            .to_numpy()
            .reshape((ny, nx)),
            {
                "standard_name": "latitude",
                "long_name": "latitude",
                "units": "degrees_north",
                "comment": "center_of_cell",
                "bounds": "lat_bnds",
                "projection": "WGS84",
            },
        ),
        "lon_bnds": (
            ("nv", "y", "x"),
            np.array(
                [
                    WGS84_gdf[f"coord_{i}"]
                    .apply(lambda p: p[0])
                    .to_numpy()
                    .reshape((ny, nx))
                    for i in range(4)
                ]
            ),
            {
                "comment": "cell boundaries, anticlockwise",
            },
        ),
        "lat_bnds": (
            ("nv", "y", "x"),
            np.array(
                [
                    WGS84_gdf[f"coord_{i}"]
                    .apply(lambda p: p[1])
                    .to_numpy()
                    .reshape((ny, nx))
                    for i in range(4)
                ]
            ),
            {
                "comment": "cell boundaries, anticlockwise",
            },
        ),
        "source_category": ("source_category", rasters_inv.categories, {}),
    },
    data_vars={
        f"emi_{sub}": (
            ("source_category", "y", "x"),
            np.full((len(rasters_inv.categories), ny, nx), np.nan),
            {
                "standard_name": (
                    f"tendency_of_atmosphere_mass_content_of_{sub}_due_to_emission"
                ),
                "long_name": f"Emissions of {sub}",
                "units": output_unit.value,
                "comment": "annual mean emission rate",
            },
        )
        for sub in rasters_inv.substances
    }
    | {
        f"emi_{sub}_total": (
            "source_category",
            np.full(len(rasters_inv.categories), np.nan),
            {
                "long_name": f"Total Emissions of {sub}",
                "units": "kg yr-1",
            },
        )
        for sub in rasters_inv.substances
    }
    | {
        f"emi_{sub}_all_sectors": (
            ("y", "x"),
            np.zeros((ny, nx)),
            {
                "standard_name": (
                    f"tendency_of_atmosphere_mass_content_of_{sub}_due_to_emission"
                ),
                "long_name": f"Aggregated Emissions of {sub} from all sectors",
                "units": output_unit.value,
                "comment": "annual mean emission rate",
            },
        )
        for sub in rasters_inv.substances
    },
    attrs=nc_cf_attributes(
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
)


ds_out
# %%
for category, sub in rasters_inv._gdf_columns:
    if (category, sub) not in rasters_inv._gdf_columns:
        continue
    emissions = rasters_inv.gdf[(category, sub)].to_numpy().reshape((ny, nx))
    # convert from kg/y to μg m-2 s-1
    # Get the desired unit
    if output_unit == Unit.ug_m2_s:
        rescaled = emissions * 1e9 / SEC_PER_YR / (RASTER_EDGE * RASTER_EDGE)
    elif output_unit == Unit.kg_yr:
        rescaled = emissions
    else:
        raise ValueError(f"Unknown unit {output_unit}")

    # Assign emissions of this category
    ds_out[f"emi_{sub}"].loc[dict(source_category=category)] = rescaled
    # Add the the categories aggregated emission
    ds_out[f"emi_{sub}_all_sectors"] += rescaled
    # Add to the total emission value
    ds_out[f"emi_{sub}_total"].loc[dict(source_category=category)] = np.sum(emissions)

# %%
ds_out.to_netcdf(
    outdir
    / f"zurich_{'inside_swiss' if INCLUDE_SWISS_OUTSIDE else 'cropped'}_{'Fsplit' if SPLIT_GNRF_ROAD_TRANSPORT else ''}_{RASTER_EDGE}x{RASTER_EDGE}_{mapluf_file.stem}_{VERSION}.nc"
)

# %%

print(f"Output written to {outdir}")

# %%
