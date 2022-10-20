"""Differnt tools for doing the weights remapping."""
from __future__ import annotations
from pathlib import Path
from warnings import warn
import numpy as np
import geopandas as gpd
from typing import TYPE_CHECKING, Iterable
from shapely.geometry import Point, MultiPolygon, Polygon
from emiproc.utilities import ProgressIndicator
from scipy.sparse import coo_array, dok_matrix

if TYPE_CHECKING:
    from os import PathLike
    from emiproc.inventories import Inventory
    from emiproc.grids import Grid


def get_weights_mapping(
    weights_filepath: Path,
    shapes_inv: Iterable[Polygon | Point],
    shapes_out: Iterable[Polygon],
    loop_over_inv_objects: bool = False,
) -> dict[str, np.ndarray]:
    """Get the requested weights mapping.

    If it does not exists, calls :py:func:`calculate_weights_mapping` .
    See that function for the other arguments.
    and save the weights once computed.

    :arg weights_filepath: The name of the file in which to save the
        weights data. Emiproc will add some metadata to it.
        This file has to be a npz archive ending with suffix .npz .
        Emiproc will add the suffix if you don't.
    :arg shapes_inv: The shapes of the inventory.
        Shapes from which the remapping will be done.
    :arg shapes_out: The shapes to which the remapping will be done.
    :arg loop_over_inv_objects: Whether the loop should happend on the
        the inventory objects instead of the output shapes.
        This will be where the performance bottleneck resides.
        It is difficult to guess what is the good option but it
        can make big differences in some cases.
        If you have point sources in your shapes_inv, this MUST be set
        to True.


    """

    weights_filepath = Path(weights_filepath).with_suffix(f".npz")
    if loop_over_inv_objects:
        # Add a small marker
        weights_filepath = weights_filepath.with_stem(
            weights_filepath.stem + "_loopinv"
        )

    if not weights_filepath.exists():
        w_mapping = calculate_weights_mapping(
            shapes_inv,
            shapes_out,
            loop_over_inv_objects,
        )
        # Make sure dir is created
        weights_filepath.parent.mkdir(exist_ok=True, parents=True)
        np.savez(weights_filepath, **w_mapping)

    else:
        w_mapping = {**np.load(weights_filepath)}
    return w_mapping


def calculate_weights_mapping(
    shapes_inv: Iterable[Polygon | Point | MultiPolygon],
    shapes_out: Iterable[Polygon],
    loop_over_inv_objects: bool = False,
) -> dict[str, np.ndarray]:
    """Return a dictionary with the mapping.

    Every weight means: From which shape in the invetory
    to which shape in the output and the weight value is the proportion
    of the inventory weight present in the output.

    The output contains the weigths mapping.

    Point source ending in more than one cell will be splitt among the cells.

    :arg inv_indexes: The indexes from which shape in the inverntory
    :arg output_indexes: The indexes from which shape in the output
    :arg weights: The weight of this connexion (between 0 and 1).
        It means the percentage of the inv shape that should go in
        the output.
    """

    w_mapping = {
        "inv_indexes": [],
        "output_indexes": [],
        "weights": [],
    }

    if loop_over_inv_objects:
        shapes_vectorized = shapes_out
        shapes_looped = shapes_inv
    else:
        shapes_vectorized = shapes_inv
        shapes_looped = shapes_out

    if isinstance(shapes_vectorized, gpd.GeoDataFrame):
        gdf_vect = shapes_vectorized.geometry
    elif isinstance(shapes_vectorized, gpd.GeoSeries):
        gdf_vect = shapes_vectorized
    elif isinstance(shapes_vectorized, list):
        gdf_vect = gpd.GeoSeries(shapes_vectorized)
    else:
        raise TypeError(f"'shapes_vectorized' cannot be {type(shapes_vectorized)}")
    gdf_vect: gpd.GeoSeries

    # Check that only area sources are vectorized
    if not np.all(
        gdf_vect.map(lambda shape: isinstance(shape, Polygon | MultiPolygon))
    ):
        raise TypeError(
            "Non Polygon geometries were found on the grid but cannot be used for remapping"
        )

    if isinstance(shapes_looped, gpd.GeoDataFrame):
        shapes_looped = shapes_looped.geometry
    elif isinstance(shapes_looped, gpd.GeoSeries):
        shapes_looped = shapes_looped
    elif isinstance(shapes_looped, list):
        shapes_looped = gpd.GeoSeries(shapes_looped)
    else:
        raise TypeError(f"'shapes_looped' cannot be {type(shapes_looped)}")
    minx, miny, maxx, maxy = shapes_looped.total_bounds
    # Mask with only what is in the bounds
    gdf_vect = gdf_vect.cx[minx:maxx, miny:maxy]

    progress = ProgressIndicator(len(shapes_looped))

    # Loop over the output shapes
    for looped_index, shape in enumerate(shapes_looped):
        progress.step()

        intersect = gdf_vect.intersects(shape)
        if np.any(intersect):
            # Find where the intesection occurs
            intersecting_serie = gdf_vect.loc[intersect]
            from_indexes = intersecting_serie.index.to_list()
            if isinstance(shape, Point):
                # Check in which areas the point ended
                n_areas = len(from_indexes)
                weights = np.full(n_areas, 1.0 / n_areas)

            else:
                # Calculate the intersection areas
                areas = (
                    intersecting_serie.intersection(shape).area.to_numpy().reshape(-1)
                )
                if loop_over_inv_objects:
                    # Take the ratio of the shape from the overlap from the shape i
                    weights = areas / shape.area
                else:
                    # Take ratio of all the inventory shapes that were crossed
                    weights = areas / intersecting_serie.area

            # Find out the mapping indexes
            looped_indexes = np.full_like(from_indexes, looped_index)
            w_mapping["output_indexes"].append(
                from_indexes if loop_over_inv_objects else looped_indexes
            )

            w_mapping["inv_indexes"].append(
                looped_indexes if loop_over_inv_objects else from_indexes
            )

            w_mapping["weights"].append(weights)

    for key, l in w_mapping.items():
        if l:  # If any weights were added
            w_mapping[key] = np.concatenate(l, axis=0).reshape(-1)

    w_mapping["output_indexes"] = np.array(w_mapping["output_indexes"], dtype=int)
    w_mapping["inv_indexes"] = np.array(w_mapping["inv_indexes"], dtype=int)

    return w_mapping


def weights_remap(
    w_mapping: dict[str, np.ndarray],
    remapped_values: np.ndarray,
    output_size: int | tuple,
) -> np.ndarray:
    """Remap using the weights mapping and a sparse matrix.

    This allows for a very fast dot product calculation if the mapping.
    Is sparse.
    """
    if isinstance(output_size, int):
        out_len = output_size
    else:
        out_len = np.product(output_size)

    A = coo_array(
        (w_mapping["weights"], (w_mapping["output_indexes"], w_mapping["inv_indexes"])),
        shape=(out_len, len(remapped_values)),
        dtype=float,
    )

    return A.dot(remapped_values).reshape(output_size)


def calculate_weights_mapping_matrix(
    shapes_inv: Iterable[Polygon | Point | MultiPolygon],
    shapes_out: Iterable[Polygon],
    loop_over_inv_objects: bool = False,
) -> coo_array:
    """Return a dictionary with the mapping.

    .. warning::

        Not implemented yet.
        Some changes might have occur from the original function
        (weights mapping without matrix).

    Every weight means: From which shape in the invetory
    to which shape in the output and the weight value is the proportion
    of the inventory weight present in the output.

    The output contains the weigths mapping.

    :arg inv_indexes: The indexes from which shape in the inverntory
    :arg output_indexes: The indexes from which shape in the output
    :arg weights: The weight of this connexion (between 0 and 1).
        It means the percentage of the inv shape that should go in
        the output.



    """

    if loop_over_inv_objects:
        shapes_vectorized = shapes_out
        shapes_looped = shapes_inv
    else:
        shapes_vectorized = shapes_inv
        shapes_looped = shapes_out

    w_mapping = dok_matrix((len(shapes_out), len(shapes_inv)), dtype=float)

    progress = ProgressIndicator(len(shapes_looped))
    progress.step()

    if isinstance(shapes_vectorized, gpd.GeoDataFrame):
        gdf_vect = shapes_vectorized.geometry
    elif isinstance(shapes_vectorized, gpd.GeoSeries):
        gdf_vect = shapes_vectorized
    elif isinstance(shapes_vectorized, list):
        gdf_vect = gpd.GeoSeries(shapes_vectorized)
    else:
        raise TypeError("'Given illegal type for shapes to process'")
    gdf_vect: gpd.GeoSeries

    # Loop over the output shapes
    for looped_index, shape in enumerate(shapes_looped):
        progress.step()

        intersect = gdf_vect.intersects(shape)
        if np.any(intersect):
            if isinstance(shape, Point):
                # Should be only one intersection
                # Note. it happened to me that one point was right at the border !
                # This will not be handleld
                from_indexe = np.nonzero(intersect.to_numpy())[0][0]

                if loop_over_inv_objects:
                    w_mapping[from_indexe, looped_index] = 1
                else:
                    w_mapping[looped_index, from_indexe] = 1

            else:
                # Find where the intesection occurs
                intersecting_serie = gdf_vect.loc[intersect]
                # Calculate the intersection areas
                areas = (
                    intersecting_serie.intersection(shape).area.to_numpy().reshape(-1)
                )

                # Find out the mapping indexes
                from_indexes = np.nonzero(intersect.to_numpy())[0].reshape(-1)
                looped_indexes = np.full_like(from_indexes, looped_index)

                if loop_over_inv_objects:
                    w_mapping[from_indexes, looped_indexes] = areas / shape.area
                else:
                    w_mapping[looped_indexes, from_indexes] = areas / shape.area

    return w_mapping


def geoserie_intersection(
    geometry: gpd.GeoSeries,
    shape: Polygon,
    keep_outside: bool = False,
    drop_unused: bool = True,
) -> tuple[gpd.GeoSeries, np.ndarray]:
    """Calculate the intersection of a geoserie and a shape.

    This can be an expensive operation you might want to cache.

    :arg geometry: The serie of shapes from you inventory or grid.
    :arg shape: A polygon which will be used for cropping.
    :arg keep_outside: Whether to keep only the outer region of the geometry
        instead the of the inner.
        If this is true, the weigth
    :arg drop_unused: Whether all the shapes from the geometry serie should
        be kept. If True, the returned serie will remove these grid shapes.
        The index of the returned serie will correspond to the original shapes.
        If False, the returned geoserie will contain geometries from the
        original grid but the weights will be 0.

    :return cropped_shapes, weights: Return a tuple containing
        the shapes of the data but
        cropped with respect to the shape and
        the weights that correspond to the proportion of the original
        geometry shapes that is present in the cropping shape.
        The weights are 0 if the geometry is not included in the cropping shape
        and 1 if the geometry is fully included. This is the opposite
        when keep_outside is set to True.


    """
    # Check the geometry that intersect the shape
    mask_intersect = geometry.intersects(shape)
    mask_within = geometry.within(shape)
    mask_boundary_intersect = mask_intersect & (~mask_within)

    # Find the intersection sahpes of the boundary shapes
    shapes_boundary_intersect = geometry.loc[mask_boundary_intersect].intersection(
        shape
    )
    weigths_boundary_intersect = (
        shapes_boundary_intersect.area / geometry.loc[mask_boundary_intersect].area
    )

    weights = np.zeros(len(geometry))
    weights[mask_within] = 1.0
    weights[mask_boundary_intersect] = weigths_boundary_intersect
    intersection_shapes = geometry.copy()

    if keep_outside:
        # Outside inverses the weights
        weights = 1.0 - weights
        mask = (~mask_within) | mask_boundary_intersect
        intersection_shapes.loc[mask_boundary_intersect] = intersection_shapes.loc[
            mask_boundary_intersect
        ].difference(shape)
    else:
        mask = mask_within | mask_boundary_intersect
        intersection_shapes.loc[mask_boundary_intersect] = shapes_boundary_intersect

    # Return only what was used
    if drop_unused:
        mask = mask & (weights > 0)
        # Reset the index to make sure we just created a new grid
        return intersection_shapes.loc[mask], weights[mask]
    else:
        return intersection_shapes, weights


def remap_inventory(inv: Inventory, grid: Grid, weigths_file: PathLike) -> Inventory:
    """Remap any inventory on the desired grid.

    This will also remap the additional gdfs of the inventory on that grid.


    :arg inv: The inventory from which to remap.
    :arg grid: The grid to remap to.
    :arg weigths_file: The file storing the weights.

    .. warning::

        To make sure the grid is defined on the same crs as the inventory,
        this funciton will call geopandas.to_crs to the grid geometries.



    """
    weigths_file = Path(weigths_file)
    if inv.crs is not None:
        grid_cells = grid.gdf.to_crs(inv.crs)
    else:
        grid_cells = grid.gdf

    if inv.gdf is not None:
        # Remap the main data
        w_mapping = get_weights_mapping(
            weigths_file, inv.gdf.geometry, grid_cells, loop_over_inv_objects=False
        )
        mapping_dict = {
            key: weights_remap(w_mapping, inv.gdf[key], len(grid_cells))
            for key in inv.gdf.columns
            if not isinstance(inv.gdf[key].dtype, gpd.array.GeometryDtype)
        }
    else:
        mapping_dict = {}

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
            if (category, sub) not in mapping_dict:
                # Create new entry
                mapping_dict[(category, sub)] = remapped
            else:
                # Add it to the category
                mapping_dict[(category, sub)] += remapped

    # Create the output inv
    out_inv = inv.copy(no_gdfs=True)
    out_inv.gdf = gpd.GeoDataFrame(
        mapping_dict,
        geometry=grid_cells.geometry,
        crs=inv.crs,
    )
    out_inv.gdfs = {}
    out_inv.history.append(f"Remapped to grid {grid}")

    return out_inv
