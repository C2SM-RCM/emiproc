#!/usr/bin/env python
# coding: utf-8
"""
This file contains a collection of functions and constants used for generating
gridded emissions for COSMO
"""
import os
import sys
import time

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import netCDF4 as nc
import numpy as np

from importlib import import_module
from multiprocessing import Pool
from shapely.geometry import Polygon

from .country_code import country_codes


# constants to convert from yr -> sec
DAY_PER_YR = 365.25
SEC_PER_DAY = 86400
SEC_PER_YR = DAY_PER_YR * SEC_PER_DAY




def write_variable(ncfile, variable, var_name, latname, lonname, unit):
    """
    Create a new variable or add to existing variable.

    ncfile              : netCDF file
    variable (np.array) : array with values to be written
    var_name (str)      : name of variable

    latname (str)
    lonname (str)
    unit (str)
    """

    if var_name not in ncfile.variables.keys():
        ncfile.createVariable(var_name, float, (latname, lonname))

        if lonname == "rlon" and latname == "rlat":
            ncfile[var_name].grid_mapping = "rotated_pole"

        ncfile[var_name].units = unit
        ncfile[var_name][:] = variable
    else:
        ncfile[var_name][:] += variable




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


def prepare_output_file(cosmo_grid, metadata, dataset):
    """Add lat & lon dimensions and variables to the dataset, handle rotated pole

    Creates & writes dimensions and variables for longitude and latitude.
    Handles the rotated pole.
    Adds the supplied metadata attributes.

    Parameters
    ----------
    cosmo_grid : COSMOGrid
        Contains information about the cosmo grid
    metadata : dict(str : str)
        Containing global file attributes. Used as argument to
        netCDF4.Dataset.setncatts.
    dataset : netCDF4.Dataset
        Writable (empty) netCDF Dataset
    """
    # Create the dimensions and the rotated pole
    if (
        cosmo_grid.pollon == 180 or cosmo_grid.pollon == 0
    ) and cosmo_grid.pollat == 90:
        lonname = "lon"
        latname = "lat"
    else:
        lonname = "rlon"
        latname = "rlat"
        var_rotpol = dataset.createVariable("rotated_pole", str)
        var_rotpol.grid_mapping_name = "rotated_latitude_longitude"
        var_rotpol.grid_north_pole_latitude = cosmo_grid.pollat
        var_rotpol.grid_north_pole_longitude = cosmo_grid.pollon
        var_rotpol.north_pole_grid_longitude = 0.0

    dataset.createDimension(lonname, cosmo_grid.nx)
    dataset.createDimension(latname, cosmo_grid.ny)

    # Create the variables associated to the dimensions
    var_lon = dataset.createVariable(lonname, "float32", lonname)
    var_lon.axis = "X"
    var_lon.units = "degrees_east"
    var_lon.standard_name = "longitude"
    var_lon[:] = cosmo_grid.lon_range()

    var_lat = dataset.createVariable(latname, "float32", latname)
    var_lat.axis = "Y"
    var_lat.units = "degrees_north"
    var_lat.standard_name = "latitude"
    var_lat[:] = cosmo_grid.lat_range()

    dataset.setncatts(metadata)


def add_country_mask(country_mask, dataset):
    """Create and write the country mask to the dataset.

    Parameters
    ----------
    country_mask : np.array(dtype=int)
        Contains the country mask, so has the shape (lon, lat)
    dataset : netCDF4.Dataset
        Writable netCDF Dataset
    """
    if "rotated_pole" in dataset.variables:
        var = dataset.createVariable("country_ids", "short", ("rlat", "rlon"))
        var.grid_mapping = "rotated_pole"
    else:
        var = dataset.createVariable("country_ids", "short", ("lat", "lon"))

    var.long_name = "EMEP_country_code"
    # Transpose the country mask to conform with the storage of netcdf
    # python: (lon, lat), FORTRAN: (lat, lon)
    var[:] = country_mask.T


def get_country_mask(output_path, suffix, cosmo_grid, resolution, nprocs):
    """Returns the country-mask, either loaded from disk or computed.

    If there already exists a file at
    output_path/country_mask_{resolution}_{suffix}.nc
    ask the user if he wants to recompute.

    Parameters
    ----------
    output_path : str
        Path to the directory where the country-mask is stored
    suffix : str
    cosmo_grid : grids.COSMOGrid
        Contains all necessary information about the cosmo grid
    resolution : str
        The resolution for the used shapefile, used as argument for
        cartopy.io.shapereader.natural_earth()
    nprocs : int
        Number of processes used to compute the codes in parallel

    Returns
    -------
    np.array(shape(cosmo_grid.nx, cosmo_grid.ny), dtype=int)
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
        country_mask = compute_country_mask(cosmo_grid, resolution, nprocs)
        with nc.Dataset(cmask_path, "w") as dataset:
            prepare_output_file(cosmo_grid, nc_metadata, dataset)
            add_country_mask(country_mask, dataset)
    else:
        # Transpose country mask when reading in
        # netCDF/Fortran: (lat, lon), python: (lon, lat)
        country_mask = nc.Dataset(cmask_path)["country_ids"][:].T

    return country_mask


def compute_country_mask(cosmo_grid, resolution, nprocs):
    """Determine the country-code for each gridcell and return the grid.

    Each gridcell gets assigned to code of the country with the most
    area in the cell.

    If for a given grid cell, no country is found (Ocean for example),
    the country-code 0 is assigned.

    If a country-code for a country is not found, country-code -1 is
    assigned.

    Parameters
    ----------
    cosmo_grid : grids.COSMOGrid
        Contains all necessary information about the cosmo grid
    resolution : str
        The resolution for the used shapefile, used as argument for
        cartopy.io.shapereader.natural_earth()
    nprocs : int
        Number of processes used to compute the codes in parallel

    Returns
    -------
    np.array(shape(cosmo_grid.nx, cosmo_grid.ny), dtype=int)
    """
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

    projections = {
        # Projection of the shapefile: WGS84, which the PlateCarree defaults to
        "shapefile": ccrs.PlateCarree(),
        "cosmo": cosmo_grid.get_projection(),
    }

    # arguments to assign_country_code()
    arguments = [
        (
            [(i, j) for j in range(cosmo_grid.ny)],  # indices of cells
            cosmo_grid,
            projections,
            countries,
        )
        for i in range(cosmo_grid.nx)
    ]

    with Pool(nprocs) as pool:
        res = pool.starmap(assign_country_code, arguments)
        country_mask = np.array(res, dtype=int)

    end = time.time()
    print(f"Computation is over, it took\n{int(end - start)} seconds")

    return country_mask


def assign_country_code(indices, cosmo_grid, projections, countries):
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
    cosmo_grid : grids.COSMOGrid
        Contains all necessary information about the cosmo grid
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
    projections["cosmo"] = ccrs.RotatedPole(
        pole_longitude=cosmo_grid.pollon, pole_latitude=cosmo_grid.pollat
    )

    # Name of the country attribute holding the ISO3 country abbreviation
    iso3 = "ADM0_A3"

    # Keep track of countries with no code
    no_country_code = set()

    country_mask = np.empty(len(indices), dtype=int)
    for k, (i, j) in enumerate(indices):
        cosmo_cell_x, cosmo_cell_y = cosmo_grid.cell_corners(i, j)

        projected_corners = projections["shapefile"].transform_points(
            projections["cosmo"], cosmo_cell_x, cosmo_cell_y
        )
        projected_cell = Polygon(projected_corners)

        intersected_countries = [
            country
            for country in countries
            if intersection_possible(country, projected_corners)
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


def intersection_possible(country, points):
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
    country : cartopy.io.shapereader.Record
    points : np.array(shape=(4,3), dtype=float)
        Containing the 4 corners of a cell, ordered clockwise
    Returns
    -------
    bool
        True if an intersection is possible, False otherwise
    """

    def minmax(array):
        """Find both min and max value of arr in one pass"""
        minval = maxval = array[0]

        for e in array[1:]:
            if e < minval:
                minval = e
            if e > maxval:
                maxval = e

        return minval, maxval

    c_minx, c_miny, c_maxx, c_maxy = country.bounds
    p_minx, p_maxx = minmax(points[:, 0])
    p_miny, p_maxy = minmax(points[:, 1])

    if (
        c_minx < p_maxx
        and c_maxx > p_minx
        and c_miny < p_maxy
        and c_maxy > p_miny
    ):
        return True
    return False


def compute_map_from_inventory_to_cosmo(cosmo_grid, inv_grid, nprocs):
    """Compute the mapping from inventory to cosmo grid.

    Loop over all inventory cells and determine which cosmo cells they overlap
    with. This is done in parallel.

    The result is a 2d array, where for each inventory cell a list is stored.
    That list contains triplets (i, j, r), where i, j are the indices of
    cosmo cells. r is the ratio of the overlap between the inventory and the
    cosmo cell and the total area of the inventory cell.

    Parameters
    ----------
    cosmo_grid : grids.COSMOGrid
        Contains all necessary information about the cosmo grid
    inv_grid : grids.InventoryGrid
        Contains all necessary information about the inventory grid
    nprocs : int
        Number of processes used to compute the mapping in parallel

    Returns
    -------
    np.array(shape=(inv_grid.shape), dtype=list(tuple(int, int, float)))
    """
    print("Computing the mapping between the cosmo and the inventory grid...")
    start = time.time()

    lon_size = len(inv_grid.lon_range())
    lat_size = len(inv_grid.lat_range())

    # This is the interpolation that will be returned
    mapping = np.empty((lon_size, lat_size), dtype=object)

    # Projection used to convert from the inventory coordinate system
    inv_projection = inv_grid.get_projection()

    # Projection used to convert to the cosmo grid
    cosmo_projection = cosmo_grid.get_projection()

    progress = ProgressIndicator(lon_size)
    with Pool(nprocs) as pool:
        for i in range(lon_size):
            progress.step()
            cells = []
            for j in range(lat_size):
                inv_cell_corners_x, inv_cell_corners_y = inv_grid.cell_corners(
                    i, j
                )
                cell_in_cosmo_projection = cosmo_projection.transform_points(
                    inv_projection, inv_cell_corners_x, inv_cell_corners_y
                )
                cells.append(cell_in_cosmo_projection)

            mapping[i, :] = pool.map(cosmo_grid.intersected_cells, cells)

    end = time.time()

    print(f"Computation is over, it took\n{int(end - start)} seconds.")

    return mapping


def get_gridmapping(output_path, suffix, cosmo_grid, inv_grid, nprocs):
    """Returns the interpolation between the inventory and COSMO grid.

    If there already exists a file at output_path/mapping_{suffix}.npy ask
    the user if he wants to recompute.

    Parameters
    ----------
    output_path : str
        Path to the directory where the country-mask is stored
    suffix : str
    cosmo_grid : grids.COSMOGrid
        Contains all necessary information about the cosmo grid
    inv_grid : grids.Grid
        Contains all necessary information about the inventory grid
    nprocs : int
        How many processes are used for the parallel computation

    Returns
    -------
    np.array(shape=(inv_grid.shape), dtype=list(tuple(int, int, float)))
        See the docstring of compute_map_from_inventory_to_cosmo()
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
        compute_map = True

    if compute_map:
        mapping = compute_map_from_inventory_to_cosmo(
            cosmo_grid, inv_grid, nprocs
        )

        np.save(mapping_path, mapping)
    else:
        mapping = np.load(mapping_path)

    return mapping


class ProgressIndicator:
    """Used to show progress for long operations.

    To not break the progress indicator, make sure there is no
    output to stdout between calls to step()
    """

    def __init__(self, steps):
        """steps : int
            How many steps the operation takes
        """
        self.curr_step = 0
        self.tot_steps = steps
        self.stream = sys.stdout

    def step(self):
        """Advance one step, update the progress indicator"""
        self.curr_step += 1
        self._show_progress()

    def _show_progress(self):
        progress = self.curr_step / self.tot_steps * 100
        self.stream.write(f"\r{progress:.1f}%")
        if self.curr_step == self.tot_steps:
            self.stream.write("\r")
