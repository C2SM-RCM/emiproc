"""Utilities for the diffenrent inventory."""
from __future__ import annotations
import collections
import itertools
import json
import logging
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

import fiona
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr

from shapely.geometry import Point, MultiPolygon, Polygon
from emiproc.grids import Grid

from emiproc.regrid import geoserie_intersection
from emiproc.profiles.operators import (
    get_weights_of_gdf_profiles,
    group_profiles_indexes,
)

if TYPE_CHECKING:
    from emiproc.inventories import Inventory, Category


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
    convert_lines_to_polygons: bool = True,
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
    if convert_lines_to_polygons and ("Shape_Length" in gdf or "SHAPE_Length" in gdf):
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

    Point sources at the boundary will have their emissions value divided
    by 2.

    This might removes categories and substances from the inventory, if they
    are not present anymore !

    :arg inv: The inventory to crop.
    :arg shape: The shape around which to crop the inv.
    :arg keep_outside: Whether to keep only the emissions outside of the shape.
    :arg weight_file: A file in which to store the weights.
        If modify_grid is True, this will also save the shapes of the output.
        However saving/reading those shape can be slower than computing
        them.
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
            shapes_file = Path(weight_file).with_suffix(".gdb")

    if inv.gdf is not None:
        # Check if the weights are already computed
        if weight_file is not None and weight_file.is_file():
            weights = np.load(weight_file)
            if modify_grid:
                if not shapes_file.is_dir():
                    raise RuntimeError(
                        f"A {weight_file=} was found but no {shapes_file=}."
                        " Delete the weight file to recompute it."
                    )
                # Load cached shapes
                gdf_cached_shapes: gpd.GeoDataFrame = gpd.read_file(
                    shapes_file, engine="pyogrio"
                )
                # Index was set with cache
                intersection_shapes = gdf_cached_shapes.set_index(
                    "index", drop=True
                ).geometry
        else:
            # Find the weight of the intersection, keep the same geometry
            intersection_shapes, weights = geoserie_intersection(
                inv.geometry, shape, keep_outside=keep_outside, drop_unused=modify_grid
            )
            if weight_file is not None:
                # Save the weight file
                np.save(weight_file, weights)
                if modify_grid:
                    gpd.GeoDataFrame(
                        {"index": intersection_shapes.index},
                        geometry=intersection_shapes,
                        index=intersection_shapes.index,
                    ).to_file(shapes_file, engine="pyogrio")

        inv_out.gdf = gpd.GeoDataFrame(
            {
                col: (
                    # Select the correct values from the shapes
                    inv.gdf.loc[intersection_shapes.index, col].to_numpy()
                    if modify_grid
                    else inv.gdf[col]
                )
                * weights
                for col in inv._gdf_columns
            },
            geometry=intersection_shapes.reset_index(drop=True)
            if modify_grid
            else inv.geometry,
            crs=inv.gdf.crs,
        )
    else:
        inv_out.gdf = None

    inv_out.gdfs = {}
    for cat, gdf in inv.gdfs.items():
        cols = [col for col in inv.substances if col in gdf]
        if not cols:
            # No substance of the inventory is in this category
            # No need to crop anything (cropping this will create accessing error bug later in the loop)
            continue

        mask_points = gdf.geometry.apply(lambda x: isinstance(x, Point))
        if any(mask_points):
            point_geometries = gdf.geometry.loc[mask_points]
            points_gdf = gdf.loc[mask_points].copy()
            # Simply remove the point sources outside
            mask_intersects = point_geometries.intersects(shape)
            mask_boundary = point_geometries.intersects(shape.boundary)

            # Takes shapes of interest (inside/outside and the boundary)
            mask_shapes = (
                ~mask_intersects if keep_outside else mask_intersects
            ) | mask_boundary

            # Points at the boundary are divided by 2
            points_gdf.loc[mask_boundary, cols] /= 2
            inv_out.add_gdf(cat, points_gdf.loc[mask_shapes].reset_index(drop=True))
        if not all(mask_points):
            # We keep crop the geometry
            polys_gdf = gdf.loc[~mask_points]
            new_geometry, weights = geoserie_intersection(
                polys_gdf.geometry, shape, keep_outside=keep_outside, drop_unused=False
            )
            mask_non_zero = weights > 0
            inv_out.add_gdf(
                cat,
                gpd.GeoDataFrame(
                    {
                        col: polys_gdf.loc[mask_non_zero, col] * weights[mask_non_zero]
                        for col in cols
                    },
                    geometry=new_geometry[mask_non_zero],
                    crs=gdf.crs,
                ).reset_index(drop=True),
            )

    inv_out.history.append(f"Cropped using {shape=}, {keep_outside=}")
    return inv_out


def group_categories(
    inv: Inventory,
    categories_group: dict[str, list[str]],
    ignore_missing: bool = False,
) -> Inventory:
    """Group the categories of an inventory in new categories.

    Total emissions are summed among the categories.
    Vertical profiles are weightly averaged, such that the profiles
    of a category with higher emission is more taken into account.
    Point sources vertical profiles are not modified.

    :arg inv: The Inventory to group.
    :arg categories_group: A mapping of which groups should be greated
        out of which categries. This will be checked using
        :py:func:`validate_group` .
    :arg ignore_missing: If True, function will work even if some categories
        from the mapping are not in the inventory.
        Ex. ``{"group1": ["cat1", "cat2"], "group2": ["cat3", "cat4"]}``
        If ``cat3`` is not in the inventory, the function will work as if
        ``{"group1": ["cat1", "cat2"], "group2": ["cat4"]}`` was passed.
    """
    if ignore_missing:
        # Remove the missing categories
        categories_group = {
            group: [cat for cat in categories if cat in inv.categories]
            for group, categories in categories_group.items()
        }

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

        # Add missing profile -1 to the gdfs having no profiles column
        for profile_col in ["__v_profile__", "__t_profile__"]:
            if any((profile_col in gdf.columns for gdf in group_gdfs)):
                for gdf in group_gdfs:
                    if profile_col not in gdf.columns:
                        gdf[profile_col] = -1

        if group_gdfs:
            if len(group_gdfs) == 1:
                # Case only one dataframe is registered
                out_inv.gdfs[group] = group_gdfs[0]
            else:
                # Otherwise merge the dataframes
                df_merged = pd.concat(group_gdfs, ignore_index=True)
                # Na values are no emission, replaces nan by 0
                out_inv.gdfs[group] = df_merged.mask(pd.isna(df_merged), 0.0)

    # Group the vertical profiles
    # we group only on the gdf, as the gdfs will keep their own profiles
    for profiles_name, profiles_indexes_name in [
        ("v_profiles", "v_profiles_indexes"),
        ("t_profiles_groups", "t_profiles_indexes"),
    ]:
        profiles = getattr(inv, profiles_name)
        profiles_indexes: xr.DataArray = getattr(inv, profiles_indexes_name)

        out_profiles = getattr(out_inv, profiles_name)

        if (
            profiles is not None
            # if they don't depend on category, we don't need to create new profiles
            and "category" in profiles_indexes.dims
        ):
            new_profiles, new_indices = group_profiles_indexes(
                profiles,
                profiles_indexes,
                indexes_weights=get_weights_of_gdf_profiles(inv.gdf, profiles_indexes),
                categories_group=categories_group,
                groupping_dimension="category",
            )

            # Offset the indexes for merging with the profiles
            new_indices = xr.where(
                new_indices != -1,
                new_indices + len(profiles),
                -1,
            )
            out_profiles += new_profiles
            # Replace the old indexes by the new
            setattr(out_inv, profiles_name, out_profiles)
            setattr(out_inv, profiles_indexes_name, new_indices)
            out_inv.history.append(
                f"Generated new {profiles_indexes_name} from groupping."
            )

    out_inv.history.append(f"groupped from {inv.categories} to {out_inv.categories}")

    return out_inv


def add_inventories(inv: Inventory, other_inv: Inventory) -> Inventory:
    """Add inventories together.

    The following conditions must be required:

    * if the two invs have a gdf, they must be on the same grid

    :arg inv: The first inventory.
    :arg other_inv: The second inventory.
    """
    logger = logging.getLogger("emiproc.add_inventories")

    if inv.gdf is None and other_inv.gdf is not None:
        # as we want to put everything on inv.gdf later for simplicity
        return add_inventories(other_inv, inv)

    if inv.crs != other_inv.crs:
        raise ValueError("CRS of both inventories differ.")

    if other_inv.v_profiles is not None:
        if inv.v_profiles is not None:
            raise NotImplementedError(
                "We can currently not add inventories with vertical profiles."
            )
        else:
            logger.warn(
                f"Vertical profiles of {other_inv} are going to be lost."
                "Please place it in the first position of the arguments of function `add_inventories()`."
            )

    if other_inv.t_profiles_groups is not None:
        if inv.t_profiles_groups is not None:
            raise NotImplementedError(
                "We can currently not add inventories with time profiles."
            )
        else:
            logger.warn(
                f"Time profiles of {other_inv} are going to be lost."
                "Please place it in the first position of the arguments of function `add_inventories()`."
            )

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


def get_total_emissions(inv: Inventory) -> dict[str, dict[str, float]]:
    """Get the total emissions from the inventory.

    :arg inv: The inventory from which to get the total emissions.

    :return: A dictionary mapping substances to another dictionary
        which maps categories to values.
        A '__total__' key will be created in each substnace mapping
        with the total of all the categories.

        For exemple ::

            {
                "CO2": {
                    "cat1": 3.2,
                    "cat2": 4.3,
                    "__total__": 7.5,
                },
                "CH4": {
                    "cat2": 2.1,
                    "__total__": 2.1,
                },
                ...
            }
    """

    # Preapre the output dictionary
    out_dic = {sub: {} for sub in inv.substances}

    # First look for the emissions in the gdf
    for cat, sub in inv._gdf_columns:
        # Calculate the total sum
        out_dic[sub][cat] = inv.gdf[(cat, sub)].sum()

    # Second look for the emissions in the gdfs
    for cat, gdf in inv.gdfs.items():
        for sub in inv.substances:
            if sub not in gdf.columns:
                # this category does not have the substance
                continue
            if cat not in out_dic[sub]:
                out_dic[sub][cat] = 0
            # Add the total emissions
            out_dic[sub][cat] += gdf[sub].sum()
    # Add the total
    for dic in out_dic.values():
        dic["__total__"] = sum(dic.values())

    return out_dic


def scale_inventory(
    inv: Inventory, scaling_dict: dict[str, dict[str, float]]
) -> Inventory:
    """Get the total emissions from the inventory.

    :arg inv: The inventory from which to get the total emissions.

    :arg scaling_dict: A dictionary mapping substances to another dictionary
        which maps categories to values.
        The values must be scaling factor, which will multiply all the objects
        from the inventory with matching categories and substance.

        If a category/substance is not in the dict, it will not be scaled.
        For exemple ::

            {
                "CO2": {
                    "cat1": 1.3,
                    "cat2": 0.7,
                },
                "CH4": {
                    "cat2": 1.2,
                },
            }

    :return: A new inventory with its emission values rescaled.
    """
    # Deep copy of the inventory
    inv = inv.copy()

    # Iterate over the scaling dict to multiply the values
    for sub, sub_dict in scaling_dict.items():
        for cat, scaling_factor in sub_dict.items():
            if (cat, sub) in inv.gdf.columns:
                inv.gdf[(cat, sub)] *= scaling_factor
            if cat in inv.gdfs.keys() and sub in inv.gdfs[cat]:
                inv.gdfs[cat][sub] *= scaling_factor

    inv.history.append(f"Rescaled using {scaling_dict=}")
    return inv


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
