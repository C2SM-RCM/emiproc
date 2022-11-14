"""Script used for creating raster data of zurich."""
#%%
from math import floor
from pathlib import Path


from emiproc.grids import SwissGrid, LV95, WGS84
from emiproc.inventories.zurich import MapLuftZurich
from emiproc.regrid import remap_inventory, weights_remap, get_weights_mapping
from emiproc.utilities import SEC_PER_YR
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, Point

import matplotlib.pyplot as plt
import xarray as xr

#%%

outdir = Path(r"C:\Users\coli\Documents\ZH-CH-emission\output_files\mapluft_rasters")
mapluf_file = Path(
    r"C:\Users\coli\Documents\ZH-CH-emission\Data\mapLuft_2020_v2021\mapLuft_2020_v2021.gdb"
)

inv = MapLuftZurich(mapluf_file)
# %% Out dataset

import xarray as xr
from datetime import datetime


def nc_cf_attributes():
    return {
        "Conventions": "CF-1.5",
        "title": f"Annual mean emissions of CO2 of the city of Zurich (only emissions within the political borders of the city)",
        "comment": "Created for use in the EU project ICOS-Cities",
        "swiss_coordinate_system_lv95": "https://www.swisstopo.admin.ch/en/knowledge-facts/surveying-geodesy/coordinates/swiss-coordinates.html",
        "comment_lv95": "In original LV95 system, x denote northings and y eastings. They have been exchanged here for better compatibility with lon/lat.",
        "source": "https://www.stadt-zuerich.ch/gud/de/index/umwelt_energie/luftqualitaet/schadstoffquellen/emissionskataster.html",
        "history": "Created from original GIS inventory mapLuft of the city of Zurich by rasterizing all point, line and area sources",
        "references": "Produced by a emiproc script TODO: add link for reading the script",
        "copyright_notice": "",
        "institution": "Empa, Swiss Federal Laboratories for Materials Science and Technology",
        "author": "Lionel Constantin, Empa",
        "contact": "dominik.brunner@empa.ch",
        "created": "Lionel Constantin, Empa",
        "creation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


#%%
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

#%% create the zurich swiss grid
raster_edge = 100
d = raster_edge
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
#%%
WGS84_gdf = zh_gdf.to_crs(WGS84)
for i in range(4):
    WGS84_gdf[f"coord_{i}"] = WGS84_gdf.geometry.apply(
        lambda poly: poly.exterior.coords[i]
    )

WGS84_gdf
# %% to see if it is clockwise or not
# for i in range(4):
#     plt.scatter(*WGS84_gdf.loc[0][f'coord_{i}'], label=f"{i}")
# plt.legend()
#%%
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
        "source_category": ("source_category", inv.categories, {}),
    },
    data_vars={
        f"emi_{sub}": (
            ("source_category", "y", "x"),
            np.full((len(inv.categories), ny, nx), np.nan),
            {
                "standard_name": f"tendency_of_atmosphere_mass_content_of_{sub}_due_to_emission",
                "long_name": f"Emissions of {sub}",
                "units": "μg m-2 s-1",
                "comment": "annual mean emission rate",
            },
        )
        for sub in inv.substances
    }
    | {
        f"emi_{sub}_total": (
            ("source_category"),
            np.full(len(inv.categories), np.nan),
            {
                "long_name": f"Total Emissions of {sub}",
                "units": "kg yr-1",
            },
        )
        for sub in inv.substances
    }
    | {
        f"emi_{sub}_all_sectors": (
            ("y", "x"),
            np.zeros((ny, nx)),
            {
                "standard_name": f"tendency_of_atmosphere_mass_content_of_{sub}_due_to_emission",
                "long_name": f"Aggregated Emissions of {sub} from all sectors",
                "units": "μg m-2 s-1",
                "comment": "annual mean emission rate",
            },
        )
        for sub in inv.substances
    },
    attrs=nc_cf_attributes(),
)


ds_out
#%% do the actual remapping 
import importlib
import emiproc.regrid
importlib.reload(emiproc.regrid)
import emiproc.regrid
from emiproc.regrid import remap_inventory
from emiproc.inventories.utils import crop_with_shape


rasters_inv = remap_inventory(
    crop_with_shape(inv, zh_shape),
    zh_gdf.geometry,
    weigths_file=  (
        outdir
        / "weights_files"
        / f"{mapluf_file.stem}_2_{raster_edge}x{raster_edge}"
    )
)
#%%
for category, sub in rasters_inv._gdf_columns:
    if (category, sub) not in rasters_inv._gdf_columns:
        continue
    emissions = rasters_inv.gdf[(category, sub)].to_numpy().reshape((ny, nx ))
    # convert from kg/y to μg m-2 s-1
    rescaled = emissions * 1e9 / SEC_PER_YR / (raster_edge * raster_edge)
    # Assign emissions of this category
    ds_out[f"emi_{sub}"].loc[
        dict(source_category=category)
    ] = rescaled
    # Add the the categories aggregated emission
    ds_out[
        f"emi_{sub}_all_sectors"
    ] += rescaled
    # Add to the total emission value
    ds_out[f"emi_{sub}_total"].loc[
        dict(source_category=category)
    ] = np.sum(emissions)

# %%
ds_out.to_netcdf(outdir / f"zurich_{'cropped'}_{raster_edge}x{raster_edge}_{mapluf_file.stem[:-6]}_v1.0.nc")


# %%
from emiproc.plots import explore_inventory,plot_inventory
explore_inventory(rasters_inv,'c1101_Linienschiffe_Emissionen_Kanton', 'CO2' )
# %%
plot_inventory(
    rasters_inv,
    figsize=(20,9),
    out_dir=r"C:\Users\coli\Pictures\zh_rasters"
)

# %%
