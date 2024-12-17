"""Utility functions and constants for emission processing."""

from __future__ import annotations

import json
import logging
import sys
import time
import urllib
from enum import Enum
from functools import cache
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Literal, overload
from urllib.request import urlopen
from warnings import warn
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import MultiPolygon, Polygon

from emiproc import FILES_DIR, PROCESS
from emiproc.grids import WGS84, WGS84_PROJECTED, Grid

# constants to convert from yr -> sec
DAY_PER_YR = 365.25
SEC_PER_HOUR = 3600
HOUR_PER_DAY = 24
SEC_PER_DAY = SEC_PER_HOUR * HOUR_PER_DAY
SEC_PER_YR = DAY_PER_YR * SEC_PER_DAY
HOUR_PER_YR = DAY_PER_YR * HOUR_PER_DAY


def get_day_per_year(year: int | None) -> int | float:
    """Get the number of days in a year accounting for leap years."""
    if year is None:
        logger = logging.getLogger("emiproc.get_day_per_year")
        logger.warning("Year is None, using 365.25")
        return DAY_PER_YR
    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        return 366
    return 365


class Units(Enum):
    """Units for emissions."""

    KG_PER_YEAR = "kg y-1"
    KG_PER_HOUR = "kg h-1"
    KG_PER_M2_PER_S = "kg m-2 s-1"
    MUG_PER_M2_PER_S = "µg m-2 s-1"


PER_M2_UNITS = [Units.KG_PER_M2_PER_S, Units.MUG_PER_M2_PER_S]
PER_CELL_UNITS = [Units.KG_PER_YEAR, Units.KG_PER_HOUR]


def grid_polygon_intersects(
    grid: gpd.GeoSeries,
    poly: Polygon | MultiPolygon,
    within: bool = False,
) -> np.ndarray:
    """Return a mask of the intersection from grid with the polygon.

    This will be fast if the grid is way larger than the polygon.
    Or if the polygon is mostly ouside of the picture.

    :arg grid: The grid to use, as a GeoSeries
    :arg poly: The polygon to check which grid cells intersect with.
    :arg within: If True, return the cells that are fully within the polygon.
        If False, return the cells that intersect with the polygon.
    """
    if isinstance(poly, MultiPolygon):
        return np.logical_or.reduce(
            [grid_polygon_intersects(grid, p, within=within) for p in poly.geoms]
        )

    grid_boundary = Polygon.from_bounds(*grid.total_bounds)

    # poly boundaries
    minx, miny, maxx, maxy = poly.bounds
    sub_grid = grid.cx[minx:maxx, miny:maxy]

    # Crop the part outside of the country
    # TODO: check if this is really useful
    poly_cropped = poly.intersection(grid_boundary)

    out = np.zeros(len(grid), dtype=bool)

    if within:
        mask = sub_grid.within(poly_cropped)
    else:
        mask = sub_grid.intersects(poly_cropped)
    out[sub_grid.index] = mask

    return out


def get_timezones(
    update: bool = False,
    version: str | None = None,
    simplify_tolerance: float = 100,
) -> gpd.GeoDataFrame:
    """Load the timezones shapefile.

    If the file is not found, it will be downloaded from the github repository
    `timezone-boundary-builder <https://github.com/evansiroky/timezone-boundary-builder>`_
    and saved in the files/timezones directory of emiproc.

    :arg update: Download the latest version of the file.
    :arg simplify_tolerance: The tolerance for simplifying the timezones shapes.
        This is used to speed up the calculation and io processes.
        The unit is meters, as we use a projected crs. See
        `geopandas.GeoSeries.simplify <https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.simplify.html#geopandas.GeoSeries.simplify>`_
        for more information.
        Set this to 0 or False to disable simplification.
    :arg version: The version to download. If not given, the latest version will be downloaded.
        The version correspond to the name of a release (eg: 2023b)

    :returns: The timezones with the shape of each zone.

    """
    logger = logging.getLogger("emiproc.get_timezones")

    latest_dir = FILES_DIR / "timezones" / "latest"
    if version:
        path_to_save = FILES_DIR / "timezones" / version
    else:
        path_to_save = latest_dir
    path_to_save.mkdir(parents=True, exist_ok=True)

    raw_file_path = path_to_save / "combined-shapefile-with-oceans.shp"

    # First get the correct version we need
    if not raw_file_path.is_file() or update:
        repo_url_api = (
            "https://api.github.com/repos/evansiroky/timezone-boundary-builder"
        )
        repo_url = "https://github.com/evansiroky/timezone-boundary-builder"

        # Get the latest version

        if version is None:
            # Get the version to download
            with urllib.request.urlopen(repo_url_api + "/releases/latest") as response:
                if response.status == 200:
                    release_info = json.loads(response.read().decode("utf-8"))
                else:
                    raise ValueError(
                        "Failed to retrieve latest release info. Status code:"
                        f" {response.status}"
                    )

            latest_version = release_info["tag_name"]
            version_to_download = latest_version
        else:
            # Download the specific version
            version_to_download = version

        # Download the file
        url = f"{repo_url}/releases/download/{version_to_download}/timezones-with-oceans.shapefile.zip"
        logger.log(PROCESS, f"Downloading timezones from {url}")
        with urllib.request.urlopen(url) as response:
            if response.status == 200:
                resp = response.read()
            else:
                raise ValueError(
                    "Failed to retrieve latest release info. Status code:"
                    f" {response.status}"
                )
        zipfile = ZipFile(BytesIO(resp))
        zipfile.extractall(path_to_save)

    if not simplify_tolerance:
        return gpd.read_file(str(raw_file_path))

    # Simiplified file
    simplifed_file = raw_file_path.with_stem(
        raw_file_path.stem + f"_simplified_{simplify_tolerance}"
    )
    if not simplifed_file.exists() or update:
        logger.log(PROCESS, f"Simplifying timezones to {simplifed_file}")

        gdf_raw = gpd.read_file(str(raw_file_path))
        # Put on a projected map such that the tolerance is in meters
        gdf_simp = gdf_raw.to_crs(WGS84_PROJECTED).simplify(simplify_tolerance)

        gpd.GeoDataFrame(
            data=gdf_raw[["tzid"]], geometry=gdf_simp.geometry, crs=gdf_simp.crs
        ).to_file(simplifed_file)

    # Load the file
    return gpd.read_file(str(simplifed_file))


def get_timezone_mask(output_grid: Grid, **kwargs) -> np.ndarray:
    """Determine the timezone for each gridcell and return the grid.

    This function calls :py:func:`get_timezones` to get the timezones shapefile.
    Any keyword arguments will be passed to that function.

    Each gridcell gets assigned to the timezone with the most
    area in the cell.

    If for a given grid cell, no timezone is found,
    the timezone UTC is assigned, but the dataset should in theory cover the whole world.

    :arg output_grid: Grid object on which to calculate the timezones.


    :returns:
        Gridded data with the timezone of each gridcell as an array of string
        with the shape of the grid.
    """
    logger = logging.getLogger("emiproc.get_timezone_mask")
    # Put timezones on a projected crs
    gdf = get_timezones(**kwargs)

    grid_gdf = output_grid.gdf.to_crs(gdf.crs)

    # Subselect only the timezones we are interested in
    grid_bounds = grid_gdf.total_bounds
    gdf = gpd.GeoDataFrame(
        gdf.cx[grid_bounds[0] : grid_bounds[2], grid_bounds[1] : grid_bounds[3]]
    )

    # Get the longest timzone names
    longest_name_len = gdf["tzid"].str.len().max()
    timezones = np.full(len(grid_gdf), fill_value="UTC", dtype=f"U{longest_name_len}")

    from emiproc.regrid import calculate_weights_mapping

    weights_mapping = calculate_weights_mapping(grid_gdf, gdf)

    logger.log(PROCESS, "Finished calculated weights mapping")

    mask_completely_in = weights_mapping["weights"] == 1.0

    timezones[weights_mapping["inv_indexes"][mask_completely_in]] = gdf["tzid"].loc[
        weights_mapping["output_indexes"][mask_completely_in]
    ]

    for inv_index in np.unique(weights_mapping["inv_indexes"][~mask_completely_in]):
        # Get the country index wehre it is the largest
        mask_this_index = weights_mapping["inv_indexes"] == inv_index
        this_output_indexes = weights_mapping["output_indexes"][mask_this_index]
        this_weights = weights_mapping["weights"][mask_this_index]
        idx = np.argmax(this_weights)
        timezones[inv_index] = gdf["tzid"].loc[this_output_indexes[idx]]

    return timezones.reshape((output_grid.nx, output_grid.ny))


@cache
def get_natural_earth(
    resolution: str = "10m",
    category: str = "physical",
    name: str = "coastline",
) -> gpd.GeoDataFrame:
    """Download the natural earth file and returns it.

    For more information about the natural earth data, see
    `the natural earth website <https://www.naturalearthdata.com/>`_.

    As this function reads a large dataset, it caches it in case of many uses
    (ex: testing).

    :arg resolution: The resolution for the used shapefile.
        Available resolutions are: '10m', '50m', '110m'
    :arg category: The category of the shapefile.
        Available categories are: 'physical', 'cultural', 'raster'
    :arg name: The name of the shapefile.
        Category of the data to download. Many are availables, look at the
        natural earth website for more info.

    :returns: The shapefile as a GeoDataFrame

    """
    logger = logging.getLogger("emiproc.get_natural_earth")
    path_to_save = FILES_DIR / "natural_earth" / f"ne_{resolution}_{category}_{name}"
    if not path_to_save.exists():
        logger.log(
            PROCESS,
            f"Downloading the natural earth file {resolution}_{category}_{name}",
        )
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


def get_country_mask(
    output_grid: Grid | gpd.GeoSeries,
    resolution: str = "110m",
    weight_filepath: PathLike | None = None,
    return_fractions: bool = False,
) -> np.ndarray | xr.DataArray:
    """Determine the country-code for each gridcell and return the grid.

    Each gridcell gets assigned to code of the country with the most
    area in the cell.

    If for a given grid cell, no country is found (Ocean for example),
    the country-code '-99' is assigned.


    :arg output_grid:
        Contains all necessary information about the output grid.
        Can be a Grid object or a GeoSeries with a custom geometry.
    :arg resolution:
        The resolution for the used shapefile, used as argument for
        :py:func:`get_natural_earth`
    :arg return_fractions:
        In case you want to know the fraction of each country in each grid cell,
        instead of just the main country, set this to True.
        If True, this will return a `xarray.DataArray` with the fraction of each country.
        If False (default), this will return a numpy array with the main country code.


    :returns: Gridded data with the country identifier of each country (eg. BUR).
        Array of 3 char strings of the shape of the grid.
    """
    logger = logging.getLogger("emiproc.get_country_mask")

    if weight_filepath is not None:
        suffix = ".npy" if not return_fractions else ".nc"
        weight_filepath = Path(weight_filepath)
        if weight_filepath.suffix != suffix:
            raise ValueError(
                f"Weight file {weight_filepath} should have suffix {suffix}"
                f" with arg: {return_fractions=}"
            )
        if weight_filepath.is_file():
            try:
                if suffix == ".npy":
                    weigts = np.load(weight_filepath)
                    if weigts.shape == (output_grid.nx, output_grid.ny):
                        return weigts
                    else:
                        logger.warning(
                            f"Weight file {weight_filepath} does not match the grid"
                            " shape, ignoring it."
                        )
                        weight_filepath = None
                elif suffix == ".nc":
                    weigts = xr.load_dataarray(weight_filepath)
                    return weigts
            except Exception as e:
                warn(f"Could not load weight file {weight_filepath}, {e}")
                weight_filepath = None

    if resolution in ["10m", "50m"]:
        logger.log(
            PROCESS,
            f"Computing the country mask with {resolution} resolution.\n"
            "Consider using a coarser resolution to speed up the process "
            "if necessary.",
        )
    start = time.time()

    ne_name = "admin_0_countries"
    countries_gdf = get_natural_earth(
        resolution=resolution, category="cultural", name=ne_name
    )
    if isinstance(output_grid, Grid):
        grid_gdf = output_grid.gdf.copy(deep=True)
    elif isinstance(output_grid, gpd.GeoSeries):
        grid_gdf = gpd.GeoDataFrame(geometry=output_grid)
    else:
        raise TypeError(
            f"output_grid should be a Grid or a GeoSeries, not {type(output_grid)}"
        )

    if grid_gdf.crs != WGS84:
        # make sure the grid is in WGS84 as is the country data
        grid_gdf = grid_gdf.to_crs(WGS84)

    grid_boundary = grid_gdf.geometry.total_bounds

    country_shapes: dict[str, Polygon] = {}
    country_corresponding_codes: list[int] = []
    # 3 char str from ISO, missing value is also -99 so we keep for consistency
    country_mask = np.full(len(grid_gdf), fill_value="-99", dtype="U3")

    # Reduce to the bounds of the grid
    countries_gdf = countries_gdf.cx[
        grid_boundary[0] : grid_boundary[2], grid_boundary[1] : grid_boundary[3]
    ]

    progress = ProgressIndicator(len(countries_gdf) + 10)

    for geometry, iso3, sovereignt in zip(
        countries_gdf.geometry, countries_gdf["ISO_A3_EH"], countries_gdf["SOVEREIGNT"]
    ):
        progress.step()

        mask_intersect = grid_polygon_intersects(grid_gdf.geometry, geometry)
        if np.any(mask_intersect):
            if iso3 == "-99":
                # Countries with missing iso3 code
                logger.info(
                    f"Country {sovereignt=} has no iso3 code (ISO_A3_EH) in natural"
                    f" earth data '{ne_name}'"
                )
                continue
            grid_gdf.loc[grid_gdf.index, iso3] = mask_intersect
            country_shapes[iso3] = geometry
            country_corresponding_codes.append(iso3)
    country_corresponding_codes = np.array(
        country_corresponding_codes,
        # 3 char str
        dtype="U3",
    )

    if return_fractions:
        # Create the output empty array
        da = xr.DataArray(
            np.zeros((len(country_corresponding_codes), len(grid_gdf))),
            coords={
                "country": country_corresponding_codes,
                "cell": grid_gdf.index,
            },
            dims=["country", "cell"],
        )

    # Find how many countries each cell intersected
    progress.step()
    number_of_intersections = grid_gdf[country_shapes.keys()].sum(axis=1)

    # Cells having only one country
    progress.step()
    if not return_fractions:
        # Some small optimization for countries that are in only one cell
        one_cell_df = grid_gdf.loc[number_of_intersections == 1, country_shapes.keys()]
        # Will have in col-0 the index of countries from on_cell_df and col-1 the codes
        cell_country_maps = np.argwhere(one_cell_df.to_numpy())
        country_mask[one_cell_df.index[cell_country_maps[:, 0]]] = (
            country_corresponding_codes[cell_country_maps[:, 1]]
        )
        # Find indexes of cell in more than one country
        progress.step()
        mask_many_cells = number_of_intersections > 1
    else:
        # Take only cells with at least one country
        mask_many_cells = number_of_intersections >= 1
    many_cells_df = grid_gdf.loc[mask_many_cells, country_shapes.keys()]
    if len(many_cells_df) > 0:
        # First column is the index of the cell, second column is the index of the country
        cell_country_duplicates = np.argwhere(many_cells_df.to_numpy())

        # Create two arrays for preparing intersection area between grid cells and countries
        grid_shapes = gpd.GeoSeries(
            grid_gdf.loc[mask_many_cells].geometry.iloc[cell_country_duplicates[:, 0]],
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

        # rows match each cell that contain duplicate, columns match countries
        areas_matrix = np.zeros(
            (np.max(i) + 1, np.max(cell_country_duplicates[:, 1]) + 1)
        )
        areas_matrix[cell_country_duplicates[:, 0], cell_country_duplicates[:, 1]] = (
            intersection_areas
        )
        if return_fractions:
            # Get the fractions of each country in each cell
            cell_areas = grid_gdf.to_crs(WGS84_PROJECTED).area
            fractions = areas_matrix / cell_areas[u].values[:, None]
            da.loc[
                {
                    "country": country_corresponding_codes,
                    "cell": many_cells_df.index,
                }
            ] = fractions.T
        else:
            # Find the countries in which the area is the largest
            country_mask[many_cells_df.index] = country_corresponding_codes[
                np.argmax(areas_matrix, axis=1)
            ]

    end = time.time()
    logger.log(PROCESS, f"Computation is over, it took {int(end - start)} seconds")
    if return_fractions:
        if weight_filepath is not None:
            da.to_netcdf(weight_filepath)
        return da
    if isinstance(output_grid, Grid):
        country_mask = country_mask.reshape((output_grid.nx, output_grid.ny))
    if weight_filepath is not None:
        np.save(weight_filepath, country_mask)
    return country_mask


def confirm_prompt(question: str) -> bool:
    """Small function to as a prompt with y/n to the user.

    :return: True if the user said y, False otherwise
    """
    reply = None
    while reply not in ("y", "n"):
        reply = input(f"{question} (y/n): ").casefold()
    return reply == "y"


def total_emissions_almost_equal(
    total_dict_1: dict[str, dict[str, float]],
    total_dict_2: dict[str, dict[str, float]],
    relative_tolerance: float = 1e-5,
) -> bool:
    """Test that the total emissions of two inventories are almost equal.

    :arg total_dict_1: The first total emissions dictionary
    :arg total_dict_2: The second total emissions dictionary
    :arg relative_tolerance: The relative tolerance to use for the comparison.
        The comparison is done as follows:
        abs(total_dict_1 - total_dict_2) < relative_tolerance * (total_dict_1 + total_dict_2) / 2

    :returns: True if the total emissions are almost equal, False otherwise
    :raises ValueError: If the two dictionaries have different keys.
    """
    for sub in total_dict_1.keys() | total_dict_2.keys():
        if sub not in total_dict_1 or sub not in total_dict_2:
            raise ValueError(f"Subcategory {sub} is missing in one of the dictionaries")
        for cat in total_dict_1[sub].keys() | total_dict_2[sub].keys():
            if cat not in total_dict_1[sub] or cat not in total_dict_2[sub]:
                raise ValueError(
                    f"Category {cat} is missing in one of the dictionaries for substance {sub}"
                )
            # Get a very small proportion of the total emissions
            err_allowed = (
                relative_tolerance
                * (total_dict_1[sub][cat] + total_dict_2[sub][cat])
                / 2
            )

            if abs(total_dict_1[sub][cat] - total_dict_2[sub][cat]) > err_allowed:
                return False
    return True


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
        # Take the stream from a logger if possible
        logger = logging.getLogger("emiproc.ProgressIndicator")
        logger.setLevel(PROCESS)
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
        self.stream = handler.stream

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


if __name__ == "__main__":
    progg = ProgressIndicator(100)
    for i in range(100):
        progg.step()
        time.sleep(0.1)
