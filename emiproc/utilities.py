"""Utility functions and constants for emission processing."""
from __future__ import annotations
import sys
import time
from warnings import warn
from enum import Enum
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import fiona
import numpy as np
import geopandas as gpd

from shapely.geometry import Polygon, MultiPolygon

from emiproc.grids import Grid, WGS84, WGS84_PROJECTED
from emiproc.country_code import country_codes
from emiproc import FILES_DIR


# constants to convert from yr -> sec
DAY_PER_YR = 365.25
SEC_PER_DAY = 86400
SEC_PER_YR = DAY_PER_YR * SEC_PER_DAY
HOUR_PER_YR = DAY_PER_YR * 24


class Units(Enum):
    """Units for emissions."""

    KG_PER_YEAR = "kg/y"
    KG_PER_M2_PER_S = "kg/m2/s"


def grid_polygon_intersects(
    grid: gpd.GeoSeries, poly: Polygon | MultiPolygon
) -> np.ndarray:
    """Return a mask of the intersection from grid with the polygon.

    This will be fast if the grid is way lareger than the polygon.
    Or if the polygon is mostly ouside of the picture
    """
    if isinstance(poly, MultiPolygon):
        return np.logical_or.reduce(
            [grid_polygon_intersects(grid, p) for p in poly.geoms]
        )

    grid_boundary = Polygon.from_bounds(*grid.total_bounds)

    # poly boundaries
    minx, miny, maxx, maxy = poly.bounds
    sub_grid = grid.cx[minx:maxx, miny:maxy]

    # Crop the part outside of the country
    # TODO: check if this is really useful
    poly_cropped = poly.intersection(grid_boundary)

    out = np.zeros(len(grid), dtype=bool)

    out[sub_grid.index] = sub_grid.intersects(poly_cropped)

    return out


def get_natural_earth(
    resolution: str = "10m", category: str = "physical", name: str = "coastline"
) -> gpd.GeoDataFrame:
    """Download the natureal earth file and returns it if needed."""

    path_to_save = FILES_DIR / "natural_earth" / f"ne_{resolution}_{category}_{name}"
    if not path_to_save.exists():
        URL_TEMPLATE = (
            "https://naturalearth.s3.amazonaws.com/{resolution}_"
            "{category}/ne_{resolution}_{name}.zip"
        )
        path_to_save.mkdir(parents=True, exist_ok=True)
        url = URL_TEMPLATE.format(resolution=resolution, category=category, name=name)
        resp = urlopen(url)
        zipfile = ZipFile(BytesIO(resp.read()))
        zipfile.extractall(path_to_save)

    # Load the country file
    shpfile = str(path_to_save / f"ne_{resolution}_{name}.shp")
    return gpd.read_file(shpfile)


def compute_country_mask(output_grid: Grid, resolution: str, nprocs: int):
    """Determine the country-code for each gridcell and return the grid.

    Each gridcell gets assigned to code of the country with the most
    area in the cell.

    If for a given grid cell, no country is found (Ocean for example),
    the country-code 0 is assigned.

    If a country-code for a country is not found, country-code -1 is
    assigned.

    Parameters
    ----------
    output_grid :
        Contains all necessary information about the output grid
    resolution :
        The resolution for the used shapefile, used as argument for
        cartopy.io.shapereader.natural_earth()
    nprocs :
        Number of processes used to compute the codes in parallel

    Returns
    -------
    np.array(shape(output_grid.nx, output_grid.ny), dtype=int)
    """

    if resolution in ["10m", "50m"]:
        print(
            f"Computing the country mask with {resolution} resolution.\n"
            "Consider using a coarser resolution to speed up the process "
            "if necessary."
        )
    start = time.time()

    countries_gdf = get_natural_earth(
        resolution=resolution, category="cultural", name="admin_0_countries"
    )

    if hasattr(output_grid, "gdf") and output_grid.gdf is not None:
        grid_gdf = output_grid.gdf
    else:
        grid_gdf = gpd.GeoDataFrame(
            geometry=output_grid.cells_as_polylist, crs=output_grid.crs
        )

    if output_grid.crs != WGS84:
        # make sure the grid is in WGS84 as is the country data
        grid_gdf = grid_gdf.to_crs(WGS84)

    progress = ProgressIndicator(len(countries_gdf) + 10)

    grid_boundary = Polygon.from_bounds(*grid_gdf.geometry.total_bounds)

    country_shapes: dict[str, Polygon] = {}
    country_corresponding_codes: list[int] = []
    country_mask = np.zeros(len(grid_gdf), dtype=int)

    # Reduce to the bounds of the grid
    mask_countries_in_grid = countries_gdf.intersects(grid_boundary)
    countries_gdf = countries_gdf.loc[mask_countries_in_grid]

    for geometry, iso3 in zip(countries_gdf.geometry, countries_gdf["ADM0_A3"]):
        progress.step()

        mask_intersect = grid_polygon_intersects(grid_gdf.geometry, geometry)
        if np.any(mask_intersect):
            grid_gdf.loc[grid_gdf.index, iso3] = mask_intersect
            country_shapes[iso3] = geometry
            if iso3 not in country_codes.keys():
                warn(
                    f"{iso3} not in emiproc country list (emiproc.country_codes.py). "
                    "Mask value of -1 will be assigned."
                )
            country_corresponding_codes.append(
                # Set to -1 if not known
                country_codes.get(iso3, -1)
            )
    country_corresponding_codes = np.array(country_corresponding_codes)

    # Find how many countries each cell intersected
    progress.step()
    number_of_intersections = grid_gdf[country_shapes.keys()].sum(axis=1)

    # Cells having only one country
    progress.step()
    one_cell_df = grid_gdf.loc[number_of_intersections == 1, country_shapes.keys()]
    # Will have in col-0 the index of countries from on_cell_df and col-1 the codes
    cell_country_maps = np.argwhere(one_cell_df.to_numpy())
    country_mask[
        one_cell_df.index[cell_country_maps[:, 0]]
    ] = country_corresponding_codes[cell_country_maps[:, 1]]

    # Find indexes of cell in more than one country
    progress.step()
    many_cells_df = grid_gdf.loc[number_of_intersections > 1, country_shapes.keys()]
    if len(many_cells_df) > 0:
        cell_country_duplicates = np.argwhere(many_cells_df.to_numpy())

        # Create two arrays for preparing intersection area between grid cells and countries
        grid_shapes = gpd.GeoSeries(
            grid_gdf.loc[number_of_intersections > 1].geometry.iloc[
                cell_country_duplicates[:, 0]
            ],
            crs=grid_gdf.crs,
        )
        countries = gpd.GeoSeries(
            np.array([s for s in country_shapes.values()], dtype=object)[
                cell_country_duplicates[:, 1]
            ],
            crs=grid_gdf.crs,
        )
        # Calculate the intersection area
        intersection_shapes = grid_shapes.intersection(countries, align=False)
        # Use projected crs to get correct area
        intersection_areas = intersection_shapes.to_crs(WGS84_PROJECTED).area

        # Prepare a matrix for comparing the area of intersection of cells with each country
        u, i = np.unique(intersection_areas.index, return_inverse=True)

        # rows match each cell that contain duplicate, columns match
        areas_matrix = np.zeros(
            (np.max(i) + 1, np.max(cell_country_duplicates[:, 1]) + 1)
        )
        areas_matrix[
            cell_country_duplicates[:, 0], cell_country_duplicates[:, 1]
        ] = intersection_areas
        # Find the countries in which the area is the largest
        country_mask[many_cells_df.index] = country_corresponding_codes[
            np.argmax(areas_matrix, axis=1)
        ]

    end = time.time()
    print(f"Computation is over, it took\n{int(end - start)} seconds")

    return country_mask.reshape((output_grid.nx, output_grid.ny))


def confirm_prompt(question: str) -> bool:
    """Small function to as a prompt with y/n to the user.

    :return: True if the user said y, False otherwise
    """
    reply = None
    while reply not in ("y", "n"):
        reply = input(f"{question} (y/n): ").casefold()
    return reply == "y"


class ProgressIndicator:
    """Used to show progress for long operations.

    To not break the progress indicator, make sure there is no
    output to stdout between calls to step()
    """

    steps: int
    curr_step: int

    def __init__(self, steps: int):
        """steps : int
        How many steps the operation takes
        """
        self.curr_step = 0
        self.steps = steps
        self.stream = sys.stdout

        self._curr_progress_step = 0

    def step(self):
        """Advance one step, update the progress indicator"""
        self.curr_step += 1
        if (self.curr_step - self._curr_progress_step) / self.steps > 0.001:
            # Show progress when you can update a x.x % value only
            self._show_progress()

    def _show_progress(self):
        self._curr_progress_step = self.curr_step
        progress = self.curr_step / self.steps * 100
        self.stream.write(f"\r{progress:.1f}%")
        if self.curr_step == self.steps:
            self.stream.write("\r")
