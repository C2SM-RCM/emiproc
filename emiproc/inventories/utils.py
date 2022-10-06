"""Utilities for the diffenrent inventory."""
from __future__ import annotations
import collections
import itertools
import json
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

import fiona
import geopandas as gpd
import pandas as pd
import numpy as np

from shapely.geometry import Point, MultiPolygon, Polygon

from emiproc.regrid import geoserie_intersection

if TYPE_CHECKING:
    from emiproc.inventories import Inventory


def list_categories(file: PathLike) -> list[str]:
    """Return the emission categories for the desired dataset."""
    return fiona.listlayers(file)


def load_category(file: PathLike, category: str) -> gpd.GeoDataFrame:
    """Load the geodataframe of the requested category."""
    return gpd.read_file(
        file,
        layer=category,
    )


def process_emission_category(
    file: PathLike,
    category: str,
    line_width: float = 10,
) -> gpd.GeoDataFrame:
    """Process an emission category.

    Is used for Zurich Mapluft but could be adapted to other inventories.

    Does the following:

    * Load the categorie from the file
    * Convert the line shapes to polygon using :py:arg:`line_width` .

    The absolute Emission for each shape are in unit: [kg / a].
    We convert here the line emissions in polygons.

    :return: A list of tuples containing the shapes and their emission value.
    """
    gdf = load_category(file, category)

    # Sometimes it is written in big and sometimes in small 🤷
    if "Shape_Length" in gdf or "SHAPE_Length" in gdf:
        # Convert lines into Polygon

        vector_geometry = gdf.geometry.buffer(line_width, cap_style=3)

        gdf.geometry = vector_geometry

    return gdf


def validate_group(categories_groups: dict[str, list[str]], all_categories: list[str]):
    """Check the validity of a group.

    The categories_groups is a mapping from group_name: list_of_categories_in_group.
    all_categories is a list contaning all categories to check.

    This will check that
    1. A category is not included in two groups.
    2. That all categories are inside a group.

    Raises error if the mapping is not valid.
    """

    all_categories_in_groupes = itertools.chain(*categories_groups.values())

    c_groups = collections.Counter(all_categories_in_groupes)
    c_all = collections.Counter(all_categories)
    if c_groups != c_all:
        raise ValueError(
            "Categories in 'categories_groups' are not matching 'all_categories'."
            " Problem cause by "
            f"duplicates: {c_groups - c_all} or "
            f"missing: {c_all - c_groups}"
        )


def crop_with_shape(
    inv: Inventory, shape: Polygon, keep_outside: bool = False
) -> Inventory:
    """Crop the inventory in place so that only what is inside the requested shape stays.

    For each shape/grid_cell of the inventory. Only the part that
    is included inside the shape will stay.
    The emission of the shape remaining will be determined using the
    ratio of the areas.

    .. warning::
        Make sure your shape is in the same crs as the inventory.
    """
    inv_out = inv.copy(no_gdfs=True)
    if inv.gdf is not None:

        # We keep the grid of the main gdf
        _, weights = geoserie_intersection(
            inv.geometry, shape, keep_outside=keep_outside, drop_unused=False
        )
        inv_out.gdf = gpd.GeoDataFrame(
            {
                col: inv.gdf[col] * weights
                for col in inv.gdf.columns
                if not isinstance(inv.gdf[col].dtype, gpd.array.GeometryDtype)
            },
            geometry=inv.geometry,
            crs=inv.gdf.crs,
        )

    inv_out.gdfs = {}
    for cat, gdf in inv.gdfs.items():
        if isinstance(gdf.geometry.iloc[0], Point):
            # Simply remove the point sources outside
            mask_within = gdf.geometry.within(shape)
            inv_out.gdfs[cat] = gdf.loc[~mask_within if keep_outside else mask_within]
        else:
            # We keep the grid of the main gdf
            new_geometry, weights = geoserie_intersection(
                gdf.geometry, shape, keep_outside=keep_outside, drop_unused=False
            )
            mask_non_zero = weights > 0
            inv_out.gdfs[cat] = gpd.GeoDataFrame(
                {
                    col: inv.gdf.loc[mask_non_zero, col] * weights[mask_non_zero]
                    for col in inv.gdf.columns
                    if col != "geometry"
                }
                | {"_weights": weights[mask_non_zero]},
                geometry=new_geometry[mask_non_zero],
                crs=inv.gdf.crs,
            )

    inv_out.history.append(f"Cropped using {shape=}, {keep_outside=}")
    return inv_out


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
            (group, substance): group_sum
            for substance in inv.substances
            for group, categories in catergories_group.items()
            # Only add the group if there are some non zero value
            if np.any(
                group_sum := sum(
                    (
                        inv.gdf[(cat, substance)]
                        for cat in categories
                        if (cat, substance) in inv.gdf
                    )
                )
            )
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
                # Case only one dataframe is registered
                out_inv.gdfs[group] = group_gdfs[0]
            else:
                # Otherwise merge the dataframes
                df_merged = pd.concat(group_gdfs, ignore_index=True)
                # Na values are no emission
                are_na = pd.isna(df_merged)
                # Replaces nan by 0                
                out_inv.gdfs[group] = df_merged.mask(are_na, 0.) 

    inv.history.append(f"groupped from {inv.categories} to {out_inv.categories}")
    inv._groupping = catergories_group

    return out_inv


if __name__ == "__main__":
    file = r"H:\ZurichEmissions\Data\mapLuft_2020_v2021\mapLuft_2020_v2021.gdb"
    categories = list_categories(file)
    info_mapping = {}
    for cat in categories:
        info_mapping[cat] = process_emission_category(file, cat).columns.to_list()
    with open(Path(file).with_suffix('.json'), "w+") as f:
        json.dump(info_mapping, f, indent=4)
