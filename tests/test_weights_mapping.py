"""Test the weights mapping function."""
from __future__ import annotations

import pytest
import geopandas as gpd
from typing import Any, Iterable
from shapely.geometry import Point, Polygon
from emiproc.regrid import calculate_weights_mapping


# Create the geometetries of an inventory
squares = gpd.GeoSeries(
    [
        Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),
        Polygon(((0, 1), (0, 2), (1, 2), (1, 1))),
        Polygon(((1, 0), (1, 1), (2, 1), (2, 0))),
        Polygon(((1, 1), (1, 2), (2, 2), (2, 1))),
        Polygon(((2, 1), (2, 2), (3, 2), (3, 1))),
    ]
)
triangles = gpd.GeoSeries(
    [
        Polygon(((0.5, 0.5), (0.5, 1.5), (1.5, 1.5))),
        Polygon(((0.5, 0.5), (1.5, 0.5), (1.5, 1.5))),
        Polygon(((2.5, 0.5), (1.5, 1.5), (1.5, 0.5))),
        Polygon(((2.5, 0.5), (2.5, 1.5), (1.5, 1.5))),
    ]
)

points = gpd.GeoSeries(
    [
        Point(0.75, 0.75),  # on the boundary of the triangles
        Point(0.25, 0.25),
        Point(1.2, 1),
        Point(1, 1),  # Between all the squares
        Point(-1, -1),  # Outside
    ]
)

expected_weights = [
    # This are the weights that should be expected
    (0, 0, 1 / 8),
    (1, 0, 1 / 4),
    (2, 0, 0),
    (3, 0, 1 / 8),
    (4, 0, 0),
    (0, 1, 1 / 8),
    (1, 1, 0),
    (2, 1, 1 / 4),
    (3, 1, 1 / 8),
    (4, 1, 0),
    (0, 2, 0),
    (1, 2, 0),
    (2, 2, 1 / 4),
    (3, 2, 1 / 8),
    (4, 2, 0),
    (0, 3, 0),
    (1, 3, 0),
    (2, 3, 0),
    (3, 3, 1 / 8),
    (4, 3, 1 / 4),
]

weights_triangle_to_square = [
    (0, 0, 0.25),
    (0, 1, 0.5),
    (0, 2, 0.0),
    (0, 3, 0.25),
    (0, 4, 0.0),
    (1, 0, 0.25),
    (1, 1, 0.0),
    (1, 2, 0.5),
    (1, 3, 0.25),
    (1, 4, 0.0),
    (2, 2, 0.5),
    (2, 3, 0.25),
    (2, 4, 0.0),
    (3, 2, 0.0),
    (3, 3, 0.25),
    (3, 4, 0.5),
]

weights_points_to_square = [
    (0, 0, 1),
    (1, 0, 1),
    (2, 2, 0.5),
    (2, 3, 0.5),
    (3, 0, 0.25),
    (3, 1, 0.25),
    (3, 2, 0.25),
    (3, 3, 0.25),
    # Point 4 is not included in the grid
]

weights_points_to_triangles = [
    (0, 0, 0.5),
    (0, 1, 0.5),
    (2, 1, 1),
    (3, 0, 0.5),
    (3, 1, 0.5),
]


def check_equal_to_weights(
    weights_tested: dict[str, Iterable[float | int]],
    weights_ref: list[tuple[int, int, float]],
):
    # We will remove the weights at each encounter
    missing_weights = weights_ref.copy()
    for f, t, w in zip(
        weights_tested["inv_indexes"],
        weights_tested["output_indexes"],
        weights_tested["weights"],
    ):
        w_tuple = (f, t, w)
        if w_tuple not in missing_weights:
            raise ValueError(f"Unexpected weight detected: {w_tuple}")
        missing_weights.remove(w_tuple)

    for missing_w_tuple in missing_weights:
        # Check weights that were added but should not
        if missing_w_tuple[2] != 0:
            raise ValueError(f"Extra weights detected: {missing_w_tuple}")


# test functions
def test_simple_case():
    check_equal_to_weights(
        calculate_weights_mapping(squares, triangles), expected_weights
    )


def test_loop_inv():
    check_equal_to_weights(
        calculate_weights_mapping(squares, triangles, loop_over_inv_objects=True),
        expected_weights,
    )


def test_simple_case_tri_to_square():
    check_equal_to_weights(
        calculate_weights_mapping(
            triangles,
            squares,
        ),
        weights_triangle_to_square,
    )


def test_loop_tri_to_square():
    check_equal_to_weights(
        calculate_weights_mapping(triangles, squares, loop_over_inv_objects=True),
        weights_triangle_to_square,
    )


def test_points():
    check_equal_to_weights(
        calculate_weights_mapping(points, squares, loop_over_inv_objects=True),
        weights_points_to_square,
    )


def test_points_on_triangles():
    check_equal_to_weights(
        calculate_weights_mapping(points, triangles, loop_over_inv_objects=True),
        weights_points_to_triangles,
    )


def test_points_not_vect_raise_error():
    with pytest.raises(TypeError):
        check_equal_to_weights(
            calculate_weights_mapping(points, squares, loop_over_inv_objects=False),
            weights_points_to_square,
        )
