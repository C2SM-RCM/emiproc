"""Utilities for the diffenrent inventory."""

import collections
import itertools
from os import PathLike

import fiona
import geopandas as gpd
import numpy as np

from shapely.geometry import Point, MultiPolygon, Polygon


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
    emission_names: list[str] = ["Emission_CO2"],
    line_width: float = 10,
) -> tuple[gpd.GeoSeries, dict[str : list[float]]]:
    """Process an emission category.

    The absolute Emission for each shape are in unit: [kg / a].
    We convert here the line emissions in polygons.

    :return: A list of tuples containing the shape and its emission value.
    """
    df = load_category(file, category)

    emission_values = {
        emission_name: ... for emission_name in emission_names if emission_name in df
    }

    # Extract the emissions
    for emission_name in emission_values.keys():
        emission_values[emission_name] = df[emission_name].to_numpy()
        print(
            category,
            "total",
            emission_name,
            "{:e}".format(emission_values[emission_name].sum()),
        )

    vector_geometry = df.geometry

    if "Shape_Length" in df or "SHAPE_Length" in df:
        # Convert lines into Polygon

        vector_geometry = vector_geometry.buffer(line_width, cap_style=3)

    return vector_geometry, emission_values
