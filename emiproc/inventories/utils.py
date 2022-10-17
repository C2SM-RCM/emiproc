"""Utilities for the diffenrent inventory."""
from __future__ import annotations
import collections
import itertools
import json
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

import fiona
import geopandas as gpd
import pandas as pd
import numpy as np

from shapely.geometry import Point, MultiPolygon, Polygon
from emiproc.grids import Grid

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

    This will check that :

    * A category is not included in two groups.
    * That all categories are inside a group.

    :arg categories_groups: The mapping of how the categories are to be groupped.
    :arg all_categories: A list of the name of the categories in the inventar.
    :raise  ValueError: if the mapping is not valid.
    """

    all_categories_in_groupes = itertools.chain(*categories_groups.values())

    c_groups = collections.Counter(all_categories_in_groupes)
    c_all = collections.Counter(all_categories)
    if c_groups != c_all:
        raise ValueError(
            "Categories in 'categories_groups' are not matching 'all_categories'."
            " Problem cause by "
            f"duplicates or not in data: {c_groups - c_all} or "
            f"missing in group mapping: {c_all - c_groups}"
        )


def crop_with_shape(
    inv: Inventory,
    shape: Polygon,
    keep_outside: bool = False,
    weight_file: PathLike | None = None,
    modify_grid: bool = False,
) -> Inventory:
    """Crop the inventory in place so that only what is inside the requested shape stays.

    For each shape/grid_cell of the inventory. Only the part that
    is included inside the shape will stay.
    The emission of the shape remaining will be determined using the
    ratio of the areas.

    :arg inv: The inventory to crop.
    :arg shape: The shape around which to crop the inv.
    :arg keep_outside: Whether to keep only the outside shape.
    :arg weight_file: A file in which to store the weights.
    :arg modify_grid: Whether the main grid (the gdf) should be modified.
        Grid cells cropped will disappear.
        Grid cells intersected will be replace by the intersection with 
        the shape.

    .. warning::
        Make sure your shape is in the same crs as the inventory.
    """
    inv_out = inv.copy(no_gdfs=True)

    if weight_file is not None:
        weight_file = Path(weight_file).with_suffix(".npy")

        if modify_grid:
            warn(
                "Cannot cache the modified grid. Will compute it. "
                "Set 'modify_grid' to False or 'weight_file' to None "
                "to remove this warning."
            )


    if inv.gdf is not None:
        # Check if the weights are already computed
        if weight_file is not None and weight_file.is_file() and modify_grid == False:
            weights = np.load(weight_file)
        else:
            # Find the weight of the intersection, keep the same geometry
            intersection_shapes, weights = geoserie_intersection(
                inv.geometry, shape, keep_outside=keep_outside, drop_unused=modify_grid
            )
            if weight_file is not None:
                # Save the weight file
                np.save(weight_file, weights)

        inv_out.gdf = gpd.GeoDataFrame(
            {
                col: (inv.gdf.loc[intersection_shapes.index,col] if modify_grid else inv.gdf[col]) * weights
                for col in inv._gdf_columns
            },
            geometry=intersection_shapes if modify_grid else inv.geometry,
            crs=inv.gdf.crs,
        )
    else:
        inv_out.gdf = None

    inv_out.gdfs = {}
    for cat, gdf in inv.gdfs.items():
        if isinstance(gdf.geometry.iloc[0], Point):
            # Simply remove the point sources outside
            mask_within = gdf.geometry.within(shape)
            inv_out.gdfs[cat] = gdf.loc[~mask_within if keep_outside else mask_within]
        else:
            # We keep crop the geometry
            new_geometry, weights = geoserie_intersection(
                gdf.geometry, shape, keep_outside=keep_outside, drop_unused=False
            )
            mask_non_zero = weights > 0
            inv_out.gdfs[cat] = gpd.GeoDataFrame(
                {
                    col: gdf.loc[mask_non_zero, col] * weights[mask_non_zero]
                    for col in inv._gdf_columns
                },
                geometry=new_geometry[mask_non_zero],
                crs=gdf.crs,
            )

    inv_out.history.append(f"Cropped using {shape=}, {keep_outside=}")
    return inv_out


def group_categories(
    inv: Inventory,
    categories_group: dict[str, list[str]],
) -> Inventory:
    """Group the categories of an inventory in new categories.

    :arg inv: The Inventory to group.
    :arg categories_group: A mapping of which groups should be greated
        out of which categries. This will be checked using
        :py:func:`validate_group` .
    """
    validate_group(categories_group, inv.categories)
    out_inv = inv.copy(no_gdfs=True)

    if inv.gdf is not None:
        out_inv.gdf = gpd.GeoDataFrame(
            {
                # Sum all the categories containing that substance
                (group, substance): group_sum
                for substance in inv.substances
                for group, categories in categories_group.items()
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
            crs=inv.crs,
        )
    else:
        out_inv.gdf = None
    # Add the additional gdfs as well
    # Merging the categories directly
    out_inv.gdfs = {}
    for group, categories in categories_group.items():
        group_gdfs = [inv.gdfs[cat] for cat in categories if cat in inv.gdfs]
        if group_gdfs:
            if len(group_gdfs) == 1:
                # Case only one dataframe is registered
                out_inv.gdfs[group] = group_gdfs[0]
            else:
                # Otherwise merge the dataframes
                df_merged = pd.concat(group_gdfs, ignore_index=True)
                # Na values are no emission, replaces nan by 0
                out_inv.gdfs[group] = df_merged.mask(pd.isna(df_merged), 0.0)

    inv.history.append(f"groupped from {inv.categories} to {out_inv.categories}")
    inv._groupping = categories_group

    return out_inv


def add_inventories(inv: Inventory, other_inv: Inventory) -> Inventory:
    """Add inventories together.

    The followwing conditions must be required.
    * if the two invs have a gdf, they must be on the same grid
    """

    if inv.gdf is None and other_inv.gdf is not None:
        # as we want to put everything on inv.gdf later for simplicity
        return add_inventories(other_inv, inv)

    if inv.crs != other_inv.crs:
        raise ValueError("CRS of both inventories differ.")

    out_inv = inv.copy(no_gdfs=True)

    if inv.gdf is not None and other_inv.gdf is not None:
        # Check that they somehow have the same grid
        if not np.all(
            gpd.GeoSeries.geom_equals(inv.gdf.geometry, other_inv.gdf.geometry)
        ):
            raise ValueError(
                "Grids of the gdf of the two inventories are not the same."
            )
        # Get the columns for the new gdf
        cols_a = inv._gdf_columns
        cols_b = other_inv._gdf_columns
        all_cols = set(cols_a) | set(cols_b)

        # Sum the two gdfs
        gdf = gpd.GeoDataFrame(
            {
                col: (inv.gdf[col] if col in inv.gdf else 0)
                + (other_inv.gdf[col] if col in other_inv.gdf else 0)
                for col in all_cols
            },
            geometry=inv.gdf.geometry,
            crs=inv.crs,
        )
    elif inv.gdf is not None:
        # Just copy the gdf
        gdf = inv.gdf.copy()
    else:
        gdf = None

    out_inv.gdf = gdf

    # Process the gdfs
    gdfs = {}
    for cat, gdf in itertools.chain(inv.gdfs.items(), other_inv.gdfs.items()):
        if cat in gdfs:
            df_merged = pd.concat([gdfs[cat], gdf], ignore_index=True)
            # Na values are no emission, replaces nan by 0
            gdfs[cat] = df_merged.mask(pd.isna(df_merged), 0.0)
        else:
            gdfs[cat] = gdf.copy()
    out_inv.gdfs = gdfs

    out_inv.history.append(f"Added inventory {other_inv}")

    return out_inv


def combine_inventories(
    inv_inside: Inventory,
    inv_outside: Inventory,
    separated_shape: Polygon,
    output_grid: Grid | None = None,
):
    """Combine two inventories and use a shape as the boundary between the two inventories.

    .. note::
        This is not implemented yet.
        One should be careful about this implementation steps.
        Some comments are already in the code for what should be done.
    """
    # Crop the inventories around the shape

    # Drop substances/categories if only in one of them
    # Or assume they are 0

    # Check if an output grid was given
    # If yes, remap both on new grid
    # if false remap one of the two on the other: Decide how

    # Now that the two are on the same grid, we can simply

    raise NotImplementedError()


if __name__ == "__main__":
    file = r"H:\ZurichEmissions\Data\mapLuft_2020_v2021\mapLuft_2020_v2021.gdb"
    categories = list_categories(file)
    info_mapping = {}
    for cat in categories:
        info_mapping[cat] = process_emission_category(file, cat).columns.to_list()
    with open(Path(file).with_suffix(".json"), "w+") as f:
        json.dump(info_mapping, f, indent=4)
