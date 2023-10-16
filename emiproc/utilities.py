"""Utility functions and constants for emission processing."""
from __future__ import annotations
import sys
import time
import json
from warnings import warn
from enum import Enum
from io import BytesIO
import urllib
from urllib.request import urlopen
from zipfile import ZipFile
import fiona
import numpy as np
import geopandas as gpd

from shapely.geometry import Polygon, MultiPolygon

from emiproc.grids import Grid, WGS84, WGS84_PROJECTED
from emiproc import FILES_DIR



# constants to convert from yr -> sec
DAY_PER_YR = 365.25
SEC_PER_HOUR = 3600
HOUR_PER_DAY = 24
SEC_PER_DAY = SEC_PER_HOUR * HOUR_PER_DAY
SEC_PER_YR = DAY_PER_YR * SEC_PER_DAY
HOUR_PER_YR = DAY_PER_YR * HOUR_PER_DAY


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

def get_timezones(update: bool = False, version: str | None = None) -> gpd.GeoDataFrame:
    """Load the timezones shapefile.
    
    If the file is not found, it will be downloaded from the github repository
    `https://github.com/evansiroky/timezone-boundary-builder`_
    and saved in the files/timezones directory of emiproc.
    """


    latest_dir = FILES_DIR / "timezones" / "latest"
    if version: 
        path_to_save = FILES_DIR / "timezones" / version
    else:
        path_to_save = latest_dir
    path_to_save.mkdir(parents=True, exist_ok=True)

    file_path = path_to_save / "combined-shapefile-with-oceans.shp"

    # First get the correct version we need 
    if not file_path.is_file() or update:

        repo_url_api = "https://api.github.com/repos/evansiroky/timezone-boundary-builder"
        repo_url = "https://github.com/evansiroky/timezone-boundary-builder"

        # Get the latest version

        if version is None: 
            # Get the version to download
            with urllib.request.urlopen(repo_url_api + "/releases/latest") as response:
                if response.status == 200:
                    release_info = json.loads(response.read().decode('utf-8'))
                else:
                    raise ValueError(f"Failed to retrieve latest release info. Status code: {response.status}")

            latest_version = release_info["tag_name"]
            version_to_download = latest_version
        else:
            # Download the specific version
            version_to_download = version
        
        # Download the file
        url = f"{repo_url}/releases/download/{version_to_download}/timezones-with-oceans.shapefile.zip"
        print(f"Downloading timezones from {url}")
        with urllib.request.urlopen(url) as response:
            if response.status == 200:
                resp = response.read()
            else:
                raise ValueError(f"Failed to retrieve latest release info. Status code: {response.status}")
        zipfile = ZipFile(BytesIO(resp))
        zipfile.extractall(path_to_save)

    # Load the country file
    shpfile = str(file_path)
    return gpd.read_file(shpfile)
        


def get_timezone_mask(output_grid: Grid, simplify_tolerance: float = 100, **kwargs) -> np.ndarray:
    """Determine the timezone for each gridcell and return the grid."""

    # Put timezones on a projected crs
    gdf = get_timezones(**kwargs).to_crs(WGS84_PROJECTED)


    grid_gdf = output_grid.gdf.to_crs(gdf.crs)

    # Subselect only the timezones we are interested in
    grid_bounds = grid_gdf.total_bounds
    gdf = gdf.cx[grid_bounds[0]:grid_bounds[2], grid_bounds[1]:grid_bounds[3]]
    gdf= gpd.GeoDataFrame(
        gdf['tzid'], 
        geometry=gdf.simplify(simplify_tolerance)
    )

    # Get the longest timzone names 
    longest_name_len = gdf['tzid'].str.len().max()
    timezones = np.full(len(grid_gdf), fill_value='UTC', dtype=f"U{longest_name_len}")

    from emiproc.regrid import calculate_weights_mapping

    weights_mapping = calculate_weights_mapping(grid_gdf, gdf)

    print('finished calculated weights mapping')

    mask_completely_in = weights_mapping['weights'] == 1.0

    timezones[weights_mapping['inv_indexes'][mask_completely_in]] = gdf['tzid'].loc[weights_mapping['output_indexes'][mask_completely_in]]

    for inv_index in np.unique(weights_mapping['inv_indexes'][~mask_completely_in]):
        # Get the country index wehre it is the largest
        mask_this_index = weights_mapping['inv_indexes'] == inv_index
        this_output_indexes = weights_mapping['output_indexes'][mask_this_index]
        this_weights = weights_mapping['weights'][mask_this_index]
        idx = np.argmax(this_weights)
        timezones[inv_index] = gdf['tzid'].loc[this_output_indexes[idx]]  

    return timezones.reshape((output_grid.nx, output_grid.ny))



def get_natural_earth(
    resolution: str = "10m", category: str = "physical", name: str = "coastline"
) -> gpd.GeoDataFrame:
    """Download the natural earth file and returns it if needed."""

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



def compute_country_mask(output_grid: Grid, resolution: str = "110m") -> np.ndarray:
    """Determine the country-code for each gridcell and return the grid.

    Each gridcell gets assigned to code of the country with the most
    area in the cell.

    If for a given grid cell, no country is found (Ocean for example),
    the country-code -99 is assigned.


    :arg output_grid:
        Contains all necessary information about the output grid
    :arg resolution:
        The resolution for the used shapefile, used as argument for
        :py:func:`get_natural_earth`

    :returns country_ids: np.array(shape(output_grid.nx, output_grid.ny), dtype=int)
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

    grid_gdf = output_grid.gdf


    if output_grid.crs != WGS84:
        # make sure the grid is in WGS84 as is the country data
        grid_gdf = grid_gdf.to_crs(WGS84)

    grid_boundary = Polygon.from_bounds(*grid_gdf.geometry.total_bounds)

    country_shapes: dict[str, Polygon] = {}
    country_corresponding_codes: list[int] = []
    # 3 char str from ISO 
    country_mask = np.full(len(grid_gdf), fill_value="-99", dtype="U3")

    # Reduce to the bounds of the grid
    mask_countries_in_grid = countries_gdf.intersects(grid_boundary)
    countries_gdf = countries_gdf.loc[mask_countries_in_grid]

    progress = ProgressIndicator(len(countries_gdf) + 10)

    for geometry, iso3 in zip(countries_gdf.geometry, countries_gdf["ISO_A3_EH"]):
        progress.step()

        mask_intersect = grid_polygon_intersects(grid_gdf.geometry, geometry)
        if np.any(mask_intersect):
            grid_gdf.loc[grid_gdf.index, iso3] = mask_intersect
            country_shapes[iso3] = geometry
            country_corresponding_codes.append(
                iso3
            )
    country_corresponding_codes = np.array(
        country_corresponding_codes,
        # 3 char str
        dtype="U3"
    )

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
