"""Maps the swiss inventory to Icon."""
# %% Imports
import collections
import itertools
from os import PathLike
from pathlib import Path
import pandas as pd
from emiproc.inventories import Inventory
from emiproc.inventories.swiss import SwissRasters
from emiproc.inventories.utils import crop_with_shape, load_category
from emiproc.plots import explore_inventory, explore_multilevel
from emiproc.grids import LV95, WGS84, Grid, ICONGrid
from shapely.geometry import Polygon, Point
from emiproc.regrid import geoserie_intersection, get_weights_mapping, weights_remap
import geopandas as gpd
import numpy as np

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
        "NOx_15": "NOX",
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
    df_eipwp=df_eipwp[
        ["CO2", "CH4", "N2O", "NOX", "CO", "NMVOC", "SO2", "NH3", "F-Gase", "geometry"]
    ],
    df_emission=df_emissions,
    # requires_grid=False,
)
inv.gdf

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


zh_poly = load_zurich_shape()
# %% cropp the outside of zurich


# return gpd.GeoDataFrame(
#    {
#        col: inv.gdf.loc[mask, col] * weights[mask]
#        for col in inv.gdf.columns
#        if col != "geometry"
#    }
#    | {"_weights": weights[mask]},
#    geometry=intersection_shapes[mask],
#    crs=inv.gdf.crs,
# )


out_inv = crop_with_shape(inv, zh_poly, keep_outside=True)
# %%

from emiproc.inventories.utils import validate_group
from emiproc.inventories.categories_groups import CH_2_GNFR

validate_group(CH_2_GNFR, inv.categories)

#%%
def group_categories(
    inv: Inventory, catergories_group: dict[str, list[str]]
) -> Inventory:
    """Group the categories of an inventory in new categories.

    :arg inv: The Inventory to group.
    :arg categories_group: A mapping of which groups should be greated
        out of which categries. This will be checked using
        :py:func:`validate_group` .
    """
    validate_group(catergories_group, inv.categories)
    out_inv = inv.copy(no_gdfs=True)

    out_inv.gdf = gpd.GeoDataFrame(
        {
            # Sum all the categories containing that substance
            (group, substance): sum(
                (
                    inv.gdf[(cat, substance)]
                    for cat in categories
                    if (cat, substance) in inv.gdf
                )
            )
            for substance in inv.substances
            for group, categories in catergories_group.items()
        },
        geometry=inv.gdf.geometry,
        crs=inv.gdf.crs,
    )
    # Add the additional gdfs as well
    # Merging the categories directly
    out_inv.gdfs = {}
    for group, categories in catergories_group.items():
        group_gdfs = [inv.gdfs[cat] for cat in categories if cat in inv.gdfs]
        if group_gdfs:
            if len(group_gdfs) == 1:
                out_inv.gdfs[group] = group_gdfs[0]
            else:
                out_inv.gdfs[group] = pd.concat(group_gdfs, ignore_index=True)

    inv.history.append(f"groupped from {inv.categories} to {out_inv.categories}")
    inv._groupping = catergories_group

    return out_inv


# %%
groupped_inv = group_categories(out_inv, CH_2_GNFR)

# %%
import importlib
import emiproc.regrid

importlib.reload(emiproc.regrid)
import emiproc.regrid
from emiproc.regrid import geoserie_intersection, get_weights_mapping, weights_remap


def remap_inventory(inv: Inventory, grid: Grid, weigths_file: PathLike) -> Inventory:
    """Remap any inventory on the desired grid.

    This will also remap the additional gdfs of the inventory on that grid.


    :arg inv: The inventory from which to remap.
    :arg grid: The grid to remap to.
    :arg weigths_file: The file storing the weights.

    .. warning::

        Make sure the grid is defined on the same crs as the inventory.


    """
    weigths_file = Path(weigths_file)
    grid_cells = grid.gdf.to_crs(inv.gdf.crs)
    w_mapping = get_weights_mapping(
        weigths_file, inv.gdf.geometry, grid_cells, loop_over_inv_objects=False
    )
    out_gdf = gpd.GeoDataFrame(
        {
            key: weights_remap(w_mapping, inv.gdf[key], len(grid_cells))
            for key in inv.gdf.columns
            if not isinstance(inv.gdf[key].dtype, gpd.array.GeometryDtype)
        },
        geometry=grid_cells.geometry,
        crs=inv.gdf.crs,
    )
    # Add the other mappings
    for category, gdf in inv.gdfs.items():
        # Get the weights of that gdf
        w_file = weigths_file.with_stem(weigths_file.stem + f"_gdfs_{category}")
        w_mapping = get_weights_mapping(
            w_file,
            gdf.geometry,
            grid_cells,
            loop_over_inv_objects=True,
        )
        # Remap each substance
        for sub in gdf.columns:
            if isinstance(gdf[sub].dtype, gpd.array.GeometryDtype):
                continue  # Geometric column
            remapped = weights_remap(w_mapping, gdf[sub], len(grid_cells))
            if (category, sub) not in out_gdf:
                # Create new entry
                out_gdf[(category, sub)] = remapped
            else:
                # Add it to the category
                out_gdf[(category, sub)] += remapped

    # Return the output object
    out_inv = inv.copy(no_gdfs=True)
    out_inv.gdf = out_gdf
    out_inv.history.append(f"Remapped to grid {grid}")
    return out_gdf


grid_file = Path(
    r"C:\Users\coli\Documents\ZH-CH-emission\icon_Zurich_R19B9_wide_DOM01.nc"
)
icon_grid = ICONGrid(grid_file)
remaped_df = remap_inventory(groupped_inv, icon_grid, ".test_ch2icon")

# %%
def add_inventories(inv: Inventory, other_inv: Inventory):
    """Add inventories together. The must be on the same grid."""
    ...


def combine_inventories(
    inv_inside: Inventory, inv_outside: Inventory, separated_shape: Polygon
):
    """Combine two inventories and use a shape as the boundary between the two inventories."""
    ...


# %%
# View the data
col = ("GNFR_B", "CO2")
# mask = groupped_inv.gdf[col] > 0
# explore_multilevel(groupped_inv.gdf.loc[mask].iloc[10000:50000], col, logscale=True)
# %%
#%matplotlib qt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import matplotlib.style

matplotlib.style.use("default")

plt.imshow(
    groupped_inv.gdf[col]
    .to_numpy()
    .reshape((groupped_inv.grid.ny, groupped_inv.grid.nx)),
    norm=LogNorm(),
)
plt.show()

# %%
