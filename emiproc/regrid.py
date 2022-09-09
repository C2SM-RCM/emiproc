"""Differnt tools for doing the weights remapping."""

from pathlib import Path
import numpy as np
import geopandas as gpd
from typing import Iterable
from shapely.geometry import Point, MultiPolygon, Polygon
from emiproc.utilities import ProgressIndicator
from scipy.sparse import coo_array, dok_matrix


def get_weights_mapping(
    weights_filepath: Path,
    shapes_inv: Iterable[Polygon | Point],
    shapes_out: Iterable[Polygon],
    loop_over_inv_objects: bool = False,
) -> dict[str, np.ndarray]:
    """Get the requested weights mapping.

    If it does not exists, calls :py:func:`calculate_weights_mapping`
    See that function for the other arguments.
    and save the weights once computed.
    :arg weights_filepath: The name of the file in which to save the
        weights data. Emiproc will add some metadata to it.

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
        gdf_vect = gpd.GeoSeries(geometry=shapes_vectorized).geometry
    else:
        raise TypeError("'Given illegal type for shapes to process'")
    gdf_vect: gpd.GeoSeries

    if isinstance(shapes_looped, gpd.GeoDataFrame):
        shapes_looped = shapes_looped.geometry.to_list()
    elif isinstance(shapes_looped, gpd.GeoSeries):
        shapes_looped = shapes_looped.to_list()



    progress = ProgressIndicator(len(shapes_looped))

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
                w_mapping["output_indexes"].append(
                    [from_indexe] if loop_over_inv_objects else [looped_index]
                )

                w_mapping["inv_indexes"].append(
                    [looped_index] if loop_over_inv_objects else [from_indexe]
                )

                w_mapping["weights"].append([1])

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
                w_mapping["output_indexes"].append(
                    from_indexes if loop_over_inv_objects else looped_indexes
                )

                w_mapping["inv_indexes"].append(
                    looped_indexes if loop_over_inv_objects else from_indexes
                )

                w_mapping["weights"].append(areas / shape.area)

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

    TODO: FIX this wont work
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

    w_mapping = dok_matrix(( len(shapes_out), len(shapes_inv)), dtype=float)

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
