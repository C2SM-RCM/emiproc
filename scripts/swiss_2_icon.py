"""Maps the swiss inventory to Icon."""
# %% Imports
import collections
import itertools
from pathlib import Path
import pandas as pd
from emiproc.inventories import Inventory, SwissRasters
from emiproc.inventories.utils import load_category
from emiproc.grids import LV95, WGS84
from shapely.geometry import Polygon, Point
from emiproc.regrid import geoserie_intersection
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
    df_eipwp=df_eipwp,
    df_emission=df_emissions,
)


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





        #return gpd.GeoDataFrame(
        #    {
        #        col: inv.gdf.loc[mask, col] * weights[mask]
        #        for col in inv.gdf.columns
        #        if col != "geometry"
        #    }
        #    | {"_weights": weights[mask]},
        #    geometry=intersection_shapes[mask],
        #    crs=inv.gdf.crs,
        #)


out_inv = inv.crop_with_shape(zh_poly, keep_outside=True)
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
    out_inv.categories = list(catergories_group.keys())
    out_inv.gdf = gpd.GeoDataFrame(
        {
            (group, substance): sum(
                (inv.get_emissions(cat, substance) for cat in categories)
            )
            for substance in inv.substances
            for group, categories in catergories_group.items()
        },
        geometry=inv.gdf.geometry,
        crs=inv.gdf.crs,
    )
    return out_inv


# %%
groupped_inv = group_categories(out_inv, CH_2_GNFR)

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
import numpy as np

gdf = groupped_inv.gdf.loc[::10000]
gdf["log"] = np.log10(gdf[("GNFR_B", "NOx")])
gdf.loc[gdf["log"] == -np.inf, "log"] = 0
gdf.explore("log")
# %%
