#!/usr/bin/env python
# coding: utf-8
"""
This file contains a collection of functions and constants used for generating
gridded emissions for COSMO
"""
import os
from pathlib import Path
import sys
import time
from warnings import warn
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import netCDF4 as nc
import numba
import numpy as np
import geopandas as gpd

from importlib import import_module
from multiprocessing import Pool
from shapely.geometry import Polygon, MultiPolygon

from .grids import Grid, WGS84, WGS84_PROJECTED
from .country_code import country_codes


# constants to convert from yr -> sec
DAY_PER_YR = 365.25
SEC_PER_DAY = 86400
SEC_PER_YR = DAY_PER_YR * SEC_PER_DAY

def get_out_varname(var, cat, cfg, **kw_format):
    """
    Create a variable name.
    It uses the format given in the configuration file
    and combines it with the species and categories. 
    """
    varname_format = getattr(cfg, 'varname_format', '{species}_{category}')

    var = cfg.in2out_species.get(var, var)
    cat = cfg.in2out_category.get(cat, cat)

    return varname_format.format(species=var, category=cat, **kw_format)


def grid_polygon_intersects(grid: gpd.GeoSeries, poly: Polygon | MultiPolygon) -> np.ndarray:
    """Return a mask of the intersection from grid with the polygon.

    This will be fast if the grid is way lareger than the polygon.
    Or if the polygon is mostly ouside of the picture
    """
    if isinstance(poly, MultiPolygon):
        return np.logical_or.reduce([grid_polygon_intersects(grid, p) for p in poly.geoms])

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

def write_variable(ncfile, variable, var_name, latname, lonname, unit,
                   overwrite=False):
    """
    Create a new variable or add to existing variable.

    ncfile              : netCDF file
    variable (np.array) : array with values to be written
    var_name (str)      : name of variable

    latname (str)
    lonname (str)
    unit (str)

    overwrite           : overwrite existing variables (instead of adding
                          values), default = False
    """
    if var_name not in ncfile.variables.keys():
        ncfile.createVariable(var_name, float, (latname, lonname))

        if lonname == "rlon" and latname == "rlat":
            ncfile[var_name].grid_mapping = "rotated_pole"

        ncfile[var_name].units = unit
        ncfile[var_name][:] = variable

    elif overwrite:
        ncfile[var_name][:] = variable
    else:
        ncfile[var_name][:] += variable


def write_variable_ICON(ncfile, variable, var_name, unit, overwrite=False):
    """
    As function 'write_variable' but for the case of an ICON output grid
    """
    if var_name not in ncfile.variables.keys():
        ncfile.createVariable(var_name, float, ("cell"))

        ncfile[var_name].units = unit
        ncfile[var_name][:] = variable[0,:]

    elif overwrite:
        ncfile[var_name][:] = variable[0,:]
    else:
        ncfile[var_name][:] += variable[0,:]


def read_emi_from_file(path):
    """Read the emissions from a textfile at path.

    Parameters
    ----------
    path : str

    Returns
    -------
    np.array(shape=(self.nx, self.ny), dtype=float)
        Emissions as read from file
    """
    no_data = -9999
    emi_grid = np.loadtxt(path, skiprows=6)

    emi_grid[emi_grid == no_data] = 0

    return np.fliplr(emi_grid.T)



def load_cfg(cfg_path):
    """Load config file"""
    # Remove a (possible) trailing file-extension from the config path
    # (importlib doesn't want it)
    cfg_path = os.path.splitext(cfg_path)[0]
    try:
        sys.path.append(os.path.dirname(os.path.realpath(cfg_path)))
        cfg = import_module(os.path.basename(cfg_path))
    except IndexError:
        raise FileNotFoundError(
            f"Provided config file {cfg_path} doesn't exist"
        )

    return cfg


def prepare_output_file(output_grid, metadata, dataset):
    """Add lat & lon dimensions and variables to the dataset, handle rotated pole

    Creates & writes dimensions and variables for longitude and latitude.
    Handles the rotated pole.
    Adds the supplied metadata attributes.

    Parameters
    ----------
    output_grid : COSMOGrid
        Contains information about the output grid (only called for COSMOGrid)
    metadata : dict(str : str)
        Containing global file attributes. Used as argument to
        netCDF4.Dataset.setncatts.
    dataset : netCDF4.Dataset
        Writable (empty) netCDF Dataset
    """
    # Create the dimensions and the rotated pole
    if (
        output_grid.pollon == 180 or output_grid.pollon == 0
    ) and output_grid.pollat == 90:
        lonname = "lon"
        latname = "lat"
    else:
        lonname = "rlon"
        latname = "rlat"
        var_rotpol = dataset.createVariable("rotated_pole", str)
        var_rotpol.grid_mapping_name = "rotated_latitude_longitude"
        var_rotpol.grid_north_pole_latitude = output_grid.pollat
        var_rotpol.grid_north_pole_longitude = output_grid.pollon
        var_rotpol.north_pole_grid_longitude = 0.0

    dataset.createDimension(lonname, output_grid.nx)
    dataset.createDimension(latname, output_grid.ny)

    # Create the variables associated to the dimensions
    var_lon = dataset.createVariable(lonname, "float32", lonname)
    var_lon.axis = "X"
    var_lon.units = "degrees_east"
    var_lon.standard_name = "longitude"
    var_lon[:] = output_grid.lon_range()

    var_lat = dataset.createVariable(latname, "float32", latname)
    var_lat.axis = "Y"
    var_lat.units = "degrees_north"
    var_lat.standard_name = "latitude"
    var_lat[:] = output_grid.lat_range()

    dataset.setncatts(metadata)


def prepare_ICON_output_file(output_grid, metadata, dataset):
    """Prepares the output file in the format of the unstructered ICON-grid.

    Copies all dimensions, variables and attributes from the input file
    Adds the supplied metadata attributes.

    Parameters
    ----------
    output_grid : ICONGrid
        Contains information about the output grid (only called for ICONGrid)
    metadata : dict(str : str)
        Containing global file attributes. Used as argument to
        netCDF4.Dataset.setncatts.
    dataset : netCDF4.Dataset
        Writable (empty) netCDF Dataset
    """

    with nc.Dataset(output_grid.dataset_path) as src:
        # copy attributes
        for name in src.ncattrs():
            dataset.setncattr(name, src.getncattr(name))
        # copy dimensions
        for name, dimension in src.dimensions.items():
            dataset.createDimension(
                name, (len(dimension)))
        # copy all file data
        for name, variable in src.variables.items():
            x = dataset.createVariable(name, variable.datatype, variable.dimensions)
            dataset.variables[name][:] = src.variables[name][:]

    dataset.setncatts(metadata)


def add_country_mask(country_mask, dataset, model):
    """Create and write the country mask to the dataset.

    Parameters
    ----------
    country_mask : np.array(dtype=int)
        Contains the country mask, so has the shape (lon, lat)
    dataset : netCDF4.Dataset
        Writable netCDF Dataset
    """
    if model == "cosmo":
        if "rotated_pole" in dataset.variables:
            var = dataset.createVariable("country_ids", "short", ("rlat", "rlon"))
            var.grid_mapping = "rotated_pole"
        else:
            var = dataset.createVariable("country_ids", "short", ("lat", "lon"))
    elif model == "icon":
        var = dataset.createVariable("country_ids", "short", ("cell"))

    var.long_name = "EMEP_country_code"
    # Transpose the country mask to conform with the storage of netcdf
    # python: (lon, lat), FORTRAN: (lat, lon)

    if model == "cosmo":
        var[:] = country_mask.T
    elif model == "icon":
        var[:] = country_mask


def get_country_mask(output_path, suffix, output_grid, resolution, nprocs):
    """Returns the country-mask, either loaded from disk or computed.

    If there already exists a file at
    output_path/country_mask_{resolution}_{suffix}.nc
    ask the user if he wants to recompute.

    Parameters
    ----------
    output_path : str
        Path to the directory where the country-mask is stored
    suffix : str
    output_grid : grids.COSMOGrid or grids.ICONGrid
        Contains all necessary information about the output grid
    resolution : str
        The resolution for the used shapefile, used as argument for
        cartopy.io.shapereader.natural_earth()
    nprocs : int
        Number of processes used to compute the codes in parallel

    Returns
    -------
    np.array(shape(output_grid.nx, output_grid.ny), dtype=int)
    """
    cmask_path = os.path.join(
        output_path, f"country_mask_{resolution}_{suffix}.nc"
    )

    if os.path.isfile(cmask_path):
        print(
            "Would you like to overwite the "
            f"country mask found in {cmask_path}?"
        )
        answer = input("y/[n]\n")
        compute_mask = answer == "y"
    else:
        compute_mask = True

    if compute_mask:
        nc_metadata = {
            "DESCRIPTION": "Country codes",
            "DATAORIGIN": "natural earth",
            "CREATOR": "OAE preprocessing script",
            "AFFILIATION": "Empa Duebendorf, Switzerland",
            "DATE CREATED": time.ctime(time.time()),
        }
        country_mask = compute_country_mask(output_grid, resolution, nprocs)
        print(np.nanmax(country_mask))
        with nc.Dataset(cmask_path, "w") as dataset:
            if output_grid.__class__.__name__ == "COSMOGrid":
                prepare_output_file(output_grid, nc_metadata, dataset)
                add_country_mask(country_mask, dataset, "cosmo")
            elif output_grid.__class__.__name__ == "ICONGrid":
                prepare_ICON_output_file(output_grid, nc_metadata, dataset)
                add_country_mask(country_mask, dataset, "icon")
    else:
        # Transpose country mask when reading in
        # netCDF/Fortran: (lat, lon), python: (lon, lat)
        country_mask = nc.Dataset(cmask_path)["country_ids"][:].T

    return country_mask


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
    if resolution in ['10m', '50m']:
        print(
            f"Computing the country mask with {resolution} resolution.\n"
            "Consider using a coarser resolution to speed up the process "
            "if necessary."
        )
    start = time.time()

    shpfilename = shpreader.natural_earth(
        resolution=resolution, category="cultural", name="admin_0_countries"
    )
    countries = list(shpreader.Reader(shpfilename).records())

    if hasattr(output_grid, 'gdf') and output_grid.gdf is not None:
        grid_gdf = output_grid.gdf
    else:
        grid_gdf = gpd.GeoDataFrame(
            geometry=output_grid.cells_as_polylist(),
            crs=output_grid.crs
        )

    if output_grid.crs != WGS84:
        # make sure the grid is in WGS84 as is the country data
        grid_gdf = grid_gdf.to_crs(WGS84)

    progress = ProgressIndicator(len(countries) + 10)

    grid_boundary = Polygon.from_bounds(*grid_gdf.geometry.total_bounds)

    country_shapes: dict[str, Polygon] = {}
    country_corresponding_codes: list[int] = []
    country_mask = np.zeros(len(grid_gdf), dtype=int)
    for i, country in enumerate(countries):
        
        progress.step()
        iso3 = country.attributes["ADM0_A3"]
        # Reduce to the bounds

        if not country.geometry.intersects(grid_boundary):
            # They dont match
            continue

        mask_intersect = grid_polygon_intersects(grid_gdf.geometry, country.geometry)
        if np.any(mask_intersect):
            grid_gdf.loc[grid_gdf.index, iso3] = mask_intersect
            country_shapes[iso3] = country.geometry
            if iso3 not in country_codes.keys():
                warn(
                    f"{iso3} not in emiproc country list. "
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
    country_mask[one_cell_df.index[cell_country_maps[:, 0]]] = country_corresponding_codes[
        cell_country_maps[:, 1]
    ]

    # Find indexes of cell in more than one country
    progress.step()
    many_cells_df = grid_gdf.loc[number_of_intersections > 1, country_shapes.keys()]

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
    areas_matrix = np.zeros((np.max(i) + 1, np.max(cell_country_duplicates[:, 1]) + 1))
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

# This is an old function not used any more !
def assign_country_code(indices, output_grid, projections, countries):
    """Assign the country codes on the gridcells indicated by indices.

    Each gridcell gets assigned to code of the country with the most
    area in the cell.

    If for a given grid cell, no country is found (Ocean for example),
    the country-code 0 is assigned.

    If a country-code for a country is not found, country-code -1 is
    assigned and a message is printed.

    Parameters
    ----------
    indices : list(tuple(int, int))
        A list of indices indicating for which gridcells to compute the
        country code.
    output_grid : grids.COSMOGrid or grids.ICONGrid
        Contains all necessary information about the output grid
    projections : dict(str, cartopy.crs.Projection)
        Dict containing two elements:
        'shapefile': Projection used in the coordinates of the countries
        'cosmo': Projection used by the cosmo_grid
    countries : list(cartopy.io.shapereader.Record)
        List of the countries that is searched.

    Returns
    -------
    np.array(shape=(len(indices)), dtype=int)
        The country codes for the cells, ordered the same way as indices
    """
    # Have to recreate the COSMO projection since cartopy projections
    # don't work with deepcopy (which is used by multiprocessing to send
    # arguments of functions to other processes)
    if output_grid.__class__.__name__ == "COSMOGrid":
        projections["cosmo"] = ccrs.RotatedPole(
            pole_longitude=output_grid.pollon, pole_latitude=output_grid.pollat
        )

    # Name of the country attribute holding the ISO3 country abbreviation
    iso3 = "ADM0_A3"

    # Keep track of countries with no code
    no_country_code = set()

    # Get country bounds outside loop
    country_bounds = [country.bounds for country in countries]

    country_mask = np.empty(len(indices), dtype=int)
    for k, (i, j) in enumerate(indices):

        output_cell_x, output_cell_y = output_grid.cell_corners(i, j)
        output_cell_x = np.array(output_cell_x)
        output_cell_y = np.array(output_cell_y)

        if output_grid.__class__.__name__ == "COSMOGrid":
            projected_corners = projections["shapefile"].transform_points(
                projections["cosmo"], output_cell_x, output_cell_y
            )
        elif output_grid.__class__.__name__ == "ICONGrid":
            projected_corners = np.array([
                output_cell_x, output_cell_y, np.zeros(output_cell_x.shape)
            ]).T

        projected_cell = Polygon(projected_corners)

        intersected_countries = [
            country
            for country, bounds in zip(countries, country_bounds)
            if intersection_possible(projected_corners, *bounds)
            and projected_cell.intersects(country.geometry)
        ]

        try:
            # Multiple countries might be intersected, assign the one with the
            # largest area in the cell
            cell_country_name = max(
                intersected_countries,
                key=lambda c: projected_cell.intersection(c.geometry).area,
            ).attributes[iso3]
            cell_country_code = country_codes[cell_country_name]
        except ValueError:
            # max on empty sequence raises ValueError
            # If no intersection was found, country code 0 is assigned. It
            # corresponds to ocean.
            # Even if a cell is predominantly ocean, it will still get assigned
            # the country code of the largest country.
            cell_country_code = 0
        except KeyError:
            # Didn't find country name in country_codes
            no_country_code.add(cell_country_name)
            cell_country_code = -1

        country_mask[k] = cell_country_code

    if len(no_country_code) > 0:
        print(
            "The following countries were found,",
            "but didn't have a corresponding code",
        )
        print(no_country_code)

    return country_mask



@numba.jit(nopython=True)
def minmax(array):
    """Find both min and max value of arr in one pass"""
    minval = maxval = array[0]

    for e in array[1:]:
        if e < minval:
            minval = e
        if e > maxval:
            maxval = e

    return minval, maxval


@numba.jit(nopython=True)
def intersection_possible(points, c_minx, c_miny, c_maxx, c_maxy):
    """Determine if country could contain any of the points.

    The maximum and minimum x and y values of the country and the points
    are compared, to determine if any of the points lie in the bounding box
    of country.

    For rectangles oriented along the coordinate axes, Polygon.intersects()
    is faster than this function. However, we're working with arbitrarily
    oriented (and possibly distorted) gridcells, for which
    Polygon.intersects() is considerably slower.

    Parameters
    ----------
    points : np.array(shape=(4,3), dtype=float)
        Containing the 4 corners of a cell, ordered clockwise

    *country.bounds

    Returns
    -------
    bool
        True if an intersection is possible, False otherwise
    """
    # check x-coordinates
    p_minx, p_maxx = minmax(points[:, 0])

    if c_minx >= p_maxx or c_maxx <= p_minx:
        return False

    # check y-coordinates (if this still needs to be done)
    p_miny, p_maxy = minmax(points[:, 1])

    return c_miny < p_maxy and c_maxy > p_miny


def compute_map_from_inventory_to_cosmo(
    output_grid: Grid,
    inv_grid: Grid,
    nprocs: int,
):
    """Compute the mapping from inventory to cosmo grid.

    Loop over all inventory cells and determine which cosmo cells they overlap
    with. This is done in parallel.

    The result is a 2d array, where for each inventory cell a list is stored.
    That list contains triplets (i, j, r), where i, j are the indices of
    cosmo cells. r is the ratio of the overlap between the inventory and the
    cosmo cell and the total area of the inventory cell.

    Parameters
    ----------
    output_grid : grids.COSMOGrid or grids.ICONGrid
        Contains all necessary information about the output grid
    inv_grid : grids.InventoryGrid
        Contains all necessary information about the inventory grid
    nprocs : int
        Number of processes used to compute the mapping in parallel

    Returns
    -------
    np.array(shape=(inv_grid.shape), dtype=list(tuple(int, int, float)))
    """
    print(
        "Computing the mapping from the", 
        getattr(output_grid, 'name', 'output_grid'),
        "and the inventory",
        getattr(inv_grid, "name", ''),
        "grid..."
    )
    start = time.time()

    lon_size = inv_grid.nx
    lat_size = inv_grid.ny

    # This is the interpolation that will be returned
    mapping = np.empty((lon_size, lat_size), dtype=object)

    # Projection used to convert from the inventory coordinate system
    inv_projection = inv_grid.get_projection()

    # Projection used to convert to the cosmo grid
    output_projection = output_grid.get_projection()

    
    # default method
    progress = ProgressIndicator(lon_size)
    if inv_grid.crs != output_grid.crs:
        raise ValueError("CRS of inventory and output grid don't match.")
    if nprocs == 1:
        for i in range(lon_size):
            progress.step()
            cells = []
            for j in range(lat_size):
                inv_cell_corners_x, inv_cell_corners_y = inv_grid.cell_corners(
                    i, j
                )
                cell_in_output_projection = output_projection.transform_points(
                    inv_projection, inv_cell_corners_x, inv_cell_corners_y
                )
                cells.append(cell_in_output_projection)

            mapping[i, : ] = [output_grid.intersected_cells(c) for c in cells]
    else:
        with Pool(nprocs) as pool:
            for i in range(lon_size):
                progress.step()
                cells = []
                for j in range(lat_size):
                    inv_cell_corners_x, inv_cell_corners_y = inv_grid.cell_corners(
                        i, j
                    )
                    cell_in_output_projection = output_projection.transform_points(
                        inv_projection, inv_cell_corners_x, inv_cell_corners_y
                    )
                    cells.append(cell_in_output_projection)

                mapping[i, :] = pool.map(output_grid.intersected_cells, cells)

    end = time.time()

    print(f"Computation is over, it took\n{int(end - start)} seconds.")

    return mapping


def get_gridmapping(
    output_path: Path | str,
    suffix: str,
    output_grid: Grid,
    inv_grid: Grid,
    nprocs: int = 1,
):
    """Returns the interpolation between the inventory and COSMO grid.

    If there already exists a file at output_path/mapping_{suffix}.npy ask
    the user if he wants to recompute.

    Parameters
    ----------
    output_path : str
        Path to the directory where the country-mask is stored
    suffix : str
    output_grid : grids.COSMOGrid or grids.ICONGrid
        Contains all necessary information about the output grid
    inv_grid : grids.Grid
        Contains all necessary information about the inventory grid
    nprocs : int
        How many processes are used for the parallel computation

    Returns
    -------
    np.array(shape=(inv_grid.shape), dtype=list(tuple(int, int, float)))
        See the docstring of compute_map_from_inventory_to_cosmo()


    Warning
    -------
    To load the grid file, pickle is used.
    This is not safe. Only files that you trust should be loaded this way.
    """
    mapping_path = os.path.join(output_path, f"mapping_{suffix}.npy")

    if os.path.isfile(mapping_path):
        print(
            "Would you like to overwite the "
            f"gridmapping found in {mapping_path}?"
        )
        answer = input("y/[n] \n")
        compute_map = answer == "y"
    else:
        # Make sure we create the ouptut dir
        Path(output_path).mkdir(exist_ok=True, parents=True)
        
        compute_map = True

    if compute_map:
        mapping = compute_map_from_inventory_to_cosmo(
            output_grid, inv_grid, nprocs
        )

        np.save(mapping_path, mapping)
    else:
        mapping = np.load(mapping_path, allow_pickle=True)

    return mapping


def confirm_prompt(question: str) -> bool:
    """Small funciton to as a prompt with y/n to the user.
    
    :return: True if the ueser said y, False if use said no
    """
    reply = None
    while reply not in ("y", "n"):
        reply = input(f"{question} (y/n): ").casefold()
    return (reply == "y")


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



