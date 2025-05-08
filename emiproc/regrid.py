"""Different functions for doing the weights remapping."""

from __future__ import annotations
import logging
from pathlib import Path
from warnings import warn
import numpy as np
import geopandas as gpd
from typing import TYPE_CHECKING, Iterable
from shapely.geometry import Point, MultiPolygon, Polygon
from emiproc.utilities import ProgressIndicator
from scipy.sparse import coo_array, dok_matrix
from emiproc.grids import Grid
from emiproc.profiles.operators import get_weights_of_gdf_profiles, remap_profiles

logger = logging.getLogger("emiproc.regrid")

if TYPE_CHECKING:
    from os import PathLike
    from emiproc.inventories import Inventory


def get_weights_mapping(
    weights_filepath: Path | None,
    shapes_inv: Iterable[Polygon | Point],
    shapes_out: Iterable[Polygon],
    loop_over_inv_objects: bool = False,
    method: str = "new",
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
    logger.debug(
        "get_weights_mapping("
        f"{weights_filepath=},"
        f"{shapes_inv=},"
        f"{shapes_out=},"
        f"{loop_over_inv_objects=},"
        f"{method=},"
        ")"
    )
    if weights_filepath is not None:
        weights_filepath = Path(weights_filepath).with_suffix(f".npz")
        if loop_over_inv_objects:
            # Add a small marker
            weights_filepath = weights_filepath.with_stem(
                weights_filepath.stem + "_loopinv"
            )

    if (weights_filepath is None) or (not weights_filepath.exists()):
        w_mapping = calculate_weights_mapping(
            shapes_inv, shapes_out, loop_over_inv_objects, method
        )
        if weights_filepath is not None:
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
    method: str = "new",
) -> dict[str, np.ndarray]:
    """Return a dictionary with the mapping.

    Every weight means: From which shape in the invetory
    to which shape in the output and the weight value is the proportion
    of the inventory weight present in the output.

    The output contains the weights mapping.

    Point source ending in more than one cell will be splitt among the cells.

    :arg inv_indexes: The indexes from which shape in the inverntory
    :arg output_indexes: The indexes from which shape in the output
    :arg weights: The weight of this connexion (between 0 and 1).
        It means the percentage of the inv shape that should go in
        the output.
    """

    # shapes_inv = inv.gdf.geometry
    # shapes_out = grid.gdf.to_crs(inv.crs)
    # loop_over_inv_objects=False
    logger.info(
        "calculating weights mapping "
        f"from {len(shapes_inv)} inventory shapes "
        f"to {len(shapes_out)} grid cells."
    )

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
        shapes_vect = shapes_vectorized.geometry
    elif isinstance(shapes_vectorized, gpd.GeoSeries):
        shapes_vect = shapes_vectorized
    elif isinstance(shapes_vectorized, list):
        shapes_vect = gpd.GeoSeries(shapes_vectorized)
    else:
        raise TypeError(f"'shapes_vectorized' cannot be {type(shapes_vectorized)}")
    shapes_vect: gpd.GeoSeries

    # Check that only area sources are vectorized
    if not np.all(
        shapes_vect.map(lambda shape: isinstance(shape, (Polygon, MultiPolygon)))
    ):
        raise TypeError(
            "Non Polygon geometries were found on the grid but cannot be used for"
            " remapping. Use 'loop_over_inv_objects' if you want to remap points to a"
            " grid."
        )

    if isinstance(shapes_looped, gpd.GeoDataFrame):
        shapes_looped = shapes_looped.geometry
    elif isinstance(shapes_looped, gpd.GeoSeries):
        shapes_looped = shapes_looped
    elif isinstance(shapes_looped, list):
        shapes_looped = gpd.GeoSeries(shapes_looped)
    else:
        raise TypeError(f"'shapes_looped' cannot be {type(shapes_looped)}")
    shapes_looped: gpd.GeoSeries
    minx, miny, maxx, maxy = shapes_looped.total_bounds
    if minx != maxx and miny != maxy:
        # Seems to remove all the data if boundaries are equal
        # Mask with only what is in the bounds
        shapes_vect = shapes_vect.cx[minx:maxx, miny:maxy]

    progress = ProgressIndicator(len(shapes_looped))

    if method == "old":
        # Loop over the output shapes
        for looped_index, shape in enumerate(shapes_looped):
            progress.step()

            intersect = shapes_vect.intersects(shape)
            if np.any(intersect):
                # Find where the intesection occurs
                intersecting_serie = shapes_vect.loc[intersect]
                from_indexes = intersecting_serie.index.to_list()
                if isinstance(shape, Point):
                    # Check in which areas the point ended
                    n_areas = len(from_indexes)
                    weights = np.full(n_areas, 1.0 / n_areas)

                else:
                    # Calculate the intersection areas
                    areas = (
                        intersecting_serie.intersection(shape)
                        .area.to_numpy()
                        .reshape(-1)
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

    elif method == "new":
        # Merge the two geometries using intersections
        gdf_in = gpd.GeoDataFrame(geometry=shapes_vect)
        gdf_out = gpd.GeoDataFrame(geometry=shapes_looped)
        gdf_weights = gdf_in.sjoin(gdf_out, rsuffix="out")
        gdf_weights = gdf_weights.merge(
            gdf_out, left_on="index_out", right_index=True, suffixes=("", "_out")
        )
        gdf_weights.index.name = "index_inv"
        gdf_weights = gdf_weights.assign(
            geometry_inter=lambda d: (
                d["geometry"].intersection(gpd.GeoSeries(d["geometry_out"]))
            )
        )

        if loop_over_inv_objects:
            # Calculate weights for polygons
            gdf_weights["weights"] = (
                gdf_weights.geometry_inter.area / gdf_weights.geometry_out.area
            )

            # Process the points
            gdf_points = gdf_weights.loc[gdf_weights.geometry_out.type == "Point"]
            if gdf_points.shape[0]:
                nareas_points = gdf_points.groupby("index_out").transform(
                    np.count_nonzero
                )["geometry"]
                gdf_weights.loc[gdf_weights.geometry_out.type == "Point", "weights"] = (
                    1 / nareas_points
                )

            # Extract indices
            gdf_weights = gdf_weights.sort_values(by=["index_inv", "index_out"])
            w_mapping["inv_indexes"] = gdf_weights.index_out.to_numpy()
            w_mapping["output_indexes"] = gdf_weights.index.to_numpy()

        else:
            # Calculate weights and extract indices
            gdf_weights["weights"] = (
                gdf_weights.geometry_inter.area / gdf_weights.geometry.area
            )
            gdf_weights = gdf_weights.sort_values(by=["index_out", "index_inv"])
            w_mapping["inv_indexes"] = gdf_weights.index.to_numpy()
            w_mapping["output_indexes"] = gdf_weights.index_out.to_numpy()

        w_mapping["weights"] = gdf_weights.weights.to_numpy()

    else:
        raise ValueError(f"'method' must be one of ['new', 'old'] not {method}.")
    # Ensure types
    w_mapping["output_indexes"] = np.array(w_mapping["output_indexes"], dtype=int)
    w_mapping["inv_indexes"] = np.array(w_mapping["inv_indexes"], dtype=int)
    w_mapping["weights"] = np.array(w_mapping["weights"], dtype=float)

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
        out_len = np.prod(output_size)

    A = coo_array(
        (w_mapping["weights"], (w_mapping["output_indexes"], w_mapping["inv_indexes"])),
        shape=(out_len, len(remapped_values)),
        dtype=float,
    )

    return A.dot(remapped_values).reshape(output_size)


def weights_remap_matrix(
    w_matrix: coo_array,
    remapped_values: np.ndarray,
) -> np.ndarray:
    """Remap using the weights mapping and a sparse matrix.

    This is the same as :py:func:`weights_remap` but uses a matrix as input.
    This allow for not having to build the matrix multiple times
    """
    return w_matrix.dot(remapped_values)


def geoserie_intersection(
    geometry: gpd.GeoSeries,
    shape: Polygon,
    keep_outside: bool = False,
    drop_unused: bool = True,
) -> tuple[gpd.GeoSeries, np.ndarray]:
    """Calculate the intersection of a geoserie and a shape.

    This can be an expensive operation you might want to cache.

    .. note:: This is used for cropping polygons only.

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
    weights_boundary_intersect = (
        shapes_boundary_intersect.area / geometry.loc[mask_boundary_intersect].area
    )

    weights = np.zeros(len(geometry))
    weights[mask_within] = 1.0
    weights[mask_boundary_intersect] = weights_boundary_intersect
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


def remap_inventory(
    inv: Inventory,
    grid: Grid | gpd.GeoSeries,
    weights_file: PathLike | None = None,
    method: str = "new",
    keep_gdfs: bool = False,
    weigths_file: PathLike | None = None,
) -> Inventory:
    """Remap any inventory on the desired grid.

    This will also remap the additional gdfs of the inventory on that grid.


    :arg inv: The inventory from which to remap.
    :arg grid: The grid to remap to.
    :arg weights_file: The file storing the weights.
    :arg method: The method to use for remapping. See :py:func:`calculate_weights_mapping`.
    :arg keep_gdfs: Whether to keep the additional gdfs (shapped emissions) of the inventory.

    .. warning::

        To make sure the grid is defined on the same crs as the inventory,
        this funciton will call geopandas.to_crs to the grid geometries.



    """
    if weigths_file is not None:
        logger.warning(
            "The argument 'weigths_file' is deprecated because of a typo. "
            "Use 'weights_file' instead."
        )
        if weights_file is not None:
            raise ValueError(
                "You cannot use both 'weights_file' and 'weigths_file' at the same time. "
                "Please use only 'weights_file'."
            )
        weights_file = weigths_file

    if weights_file is not None:
        weights_file = Path(weights_file)

    if isinstance(grid, Grid) or issubclass(type(grid), Grid):
        grid_cells = gpd.GeoSeries(grid.cells_as_polylist, crs=grid.crs)
    elif isinstance(grid, gpd.GeoSeries):
        grid_cells = grid.reset_index(drop=True)
    else:
        raise TypeError(f"grid must be of type Grid or gpd.Geoseries, not {type(grid)}")

    # Treat possible issues with crs not matching
    if inv.crs is not None:
        if grid_cells.crs != inv.crs:
            # convert the grid cells to the correct crs
            grid_cells = grid_cells.to_crs(inv.crs)
    else:
        if grid_cells.crs is not None:
            raise ValueError(
                "The inventory given has no crs, but the grid has. "
                "Assign a crs to the inventory before remapping."
            )

    if inv.gdf is not None:
        # Remap the main data
        w_mapping_grid = get_weights_mapping(
            weights_file,
            inv.gdf.geometry,
            grid_cells,
            loop_over_inv_objects=False,
            method=method,
        )
        # Create the weights matrix
        if max(w_mapping_grid["output_indexes"]) > len(grid_cells):
            raise ValueError(
                f"Error in weights mapping: {max(w_mapping_grid['output_indexes'])=} >"
                f" {len(grid_cells)=}"
            )
        if max(w_mapping_grid["inv_indexes"]) > len(inv.gdf):
            raise ValueError(
                f"Error in weights mapping: {max(w_mapping_grid['inv_indexes'])=} >"
                f" {len(inv.gdf)=}"
            )
        w_matrix = coo_array(
            (
                w_mapping_grid["weights"],
                (w_mapping_grid["output_indexes"], w_mapping_grid["inv_indexes"]),
            ),
            shape=(len(grid_cells), len(inv.gdf)),
            dtype=float,
        )
        # Perform the remapping on each column
        mapping_dict = {
            key: weights_remap_matrix(w_matrix, inv.gdf[key])
            for key in inv._gdf_columns
        }

    else:
        mapping_dict = {}

    # Add the other mappings
    if not keep_gdfs:
        for category, gdf in inv.gdfs.items():
            # Get the weights of that gdf
            if weights_file is None:
                w_file = None
            else:
                w_file = weights_file.with_stem(weights_file.stem + f"_gdfs_{category}")
            w_mapping = get_weights_mapping(
                w_file,
                gdf.geometry.reset_index(drop=True),
                grid_cells,
                loop_over_inv_objects=True,
                method=method,
            )
            # Remap each substance
            for sub in gdf.columns:
                if isinstance(gdf[sub].dtype, gpd.array.GeometryDtype):
                    continue  # Geometric column
                remapped = weights_remap(
                    w_mapping,
                    # Reset the index, same as the grid was applied in the weights mapping function
                    gdf[sub].reset_index(drop=True),
                    len(grid_cells),
                )
                if (category, sub) not in mapping_dict:
                    # Create new entry
                    mapping_dict[(category, sub)] = remapped
                else:
                    # Add it to the category
                    mapping_dict[(category, sub)] += remapped

    # Create the output inv
    out_inv = inv.copy(
        no_gdfs=True,
        # Copy the profiles to the new inventory
        profiles=True,
    )
    out_inv.grid = grid
    out_inv.gdf = gpd.GeoDataFrame(
        mapping_dict,
        geometry=grid_cells,
        crs=inv.crs,
    )
    if keep_gdfs:
        out_inv.gdfs = {key: gdf.copy(deep=True) for key, gdf in inv.gdfs.items()}
    else:
        out_inv.gdfs = {}

    # Remap the profiles as well
    for index_name, profile_name in [
        ("t_profiles_indexes", "t_profiles_groups"),
        ("v_profiles_indexes", "v_profiles"),
    ]:
        if not hasattr(inv, index_name):
            continue
        indexes = getattr(inv, index_name)
        if indexes is None or "cell" not in indexes.dims:
            continue
        profiles = getattr(inv, profile_name)

        new_profiles, new_indexes = remap_profiles(
            profiles=profiles,
            profiles_indexes=indexes,
            emissions_weights=get_weights_of_gdf_profiles(
                inv.gdf, profiles_indexes=indexes
            ),
            weights_mapping=w_mapping_grid,
        )

        out_inv.set_profiles(new_profiles, new_indexes)

    out_inv.history.append(f"Remapped to grid {grid}, {keep_gdfs=}")

    return out_inv
