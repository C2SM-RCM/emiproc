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

from country_code import country_codes


# constants to convert from yr -> sec
DAY_PER_YR = 365.25
SEC_PER_DAY = 86400
SEC_PER_YR = DAY_PER_YR * SEC_PER_DAY


def load_cfg(cfg_path):
    """Load config file"""
    try:
        sys.path.append(os.path.dirname(os.path.realpath(cfg_path)))
        cfg = import_module(os.path.basename(cfg_path))
    except IndexError:
        raise FileNotFoundError(
            f"Provided config file {cfg_path} doesn't exist"
        )

    return cfg


def prepare_output_file(cosmo_grid, dataset):
    """Add lat & lon dimensions and variables to the dataset, handle rotated pole

    Creates & writes dimensions and variables for longitude and latitude.
    Handles the rotated pole.

    Parameters
    ----------
    cosmo_grid : COSMOGrid
        Contains information about the cosmo grid
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


def compute_country_mask(cosmo_grid, resolution):
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

    Returns
    -------
    np.array(shape(cosmo_grid.nx, cosmo_grid.ny), dtype=int)
    """
    start = time.time()
    print("Computing the country mask...")
    shpfilename = shpreader.natural_earth(
        resolution=resolution, category="cultural", name="admin_0_countries"
    )
    # Name of the country attribute holding the ISO3 country abbreviation
    iso3 = "ADM0_A3"
    countries = list(shpreader.Reader(shpfilename).records())

    country_mask = np.empty((cosmo_grid.nx, cosmo_grid.ny))

    cosmo_projection = cosmo_grid.get_projection()
    # Projection of the shapefile: WGS84, which the PlateCarree defaults to
    shapefile_projection = ccrs.PlateCarree()

    # Store countries with no defined code for user feedback
    no_country_code = set()

    progress = ProgressIndicator(cosmo_grid.nx)
    for i in range(cosmo_grid.nx):
        progress.step()
        for j in range(cosmo_grid.ny):
            # Get the corners of the cell in lat/lon coord
            cosmo_cell_x, cosmo_cell_y = cosmo_grid.cell_corners(i, j)

            projected_corners = shapefile_projection.transform_points(
                cosmo_projection, cosmo_cell_x, cosmo_cell_y
            )
            projected_cell = Polygon(projected_corners)

            intersected_countries = [
                country
                for country in countries
                if intersection_possible(country, projected_corners)
                and projected_cell.intersects(country.geometry)
            ]

            # If no intersection was found, country code 0 is assigned. It
            # corresponds to ocean.
            # If a cell is predominantly ocean, it will still get assigned
            # the country code of the largest country.
            cell_country_code = 0

            if intersected_countries:
                if len(intersected_countries) == 1:
                    cell_country_name = intersected_countries[0].attributes[
                        iso3
                    ]
                else:
                    # Multiple countries intersected, assign the one with the
                    # largest area in the cell
                    area = 0
                    for country in intersected_countries:
                        new_area = projected_cell.intersection(
                            country.geometry
                        ).area
                        if area < new_area:
                            area = new_area
                            cell_country_name = country.attributes[iso3]

                # Find code from name
                try:
                    cell_country_code = country_codes[cell_country_name]
                except KeyError:
                    no_country_code.add(cell_country_name)
                    cell_country_code = -1

            country_mask[i, j] = cell_country_code

    end = time.time()
    print(f"Computation is over, it took\n{int(end - start)} seconds")

    if len(no_country_code) > 0:
        print(
            "The following countries were found,",
            "but didn't have a corresponding code",
        )
        print(no_country_code)

    return country_mask


def get_country_mask(output_path, cosmo_grid, resolution):
    """Returns the country-mask, either loaded from disk or computed.

    If there already exists a file at output_path/country_mask.nv ask the
    user if he wants to recompute.

    Parameters
    ----------
    output_path : str
        Path to the directory where the country-mask is stored
    cosmo_grid : grids.COSMOGrid
        Contains all necessary information about the cosmo grid
    resolution : str
        The resolution for the used shapefile, used as argument for
        cartopy.io.shapereader.natural_earth()
    """
    cmask_path = os.path.join(output_path, f"country_mask_{resolution}.nc")

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
        country_mask = compute_country_mask(cosmo_grid, resolution)
        with nc.Dataset(cmask_path, "w") as dataset:
            prepare_output_file(cosmo_grid, dataset)
            add_country_mask(country_mask, dataset)
    else:
        # Transpose country mask when reading in
        # netCDF/Fortran: (lat, lon), python: (lon, lat)
        country_mask = nc.Dataset(cmask_path)["country_ids"][:].T

    return country_mask


def cell_corners(lon_var, lat_var, inv_name, i, j, cfg):
    if inv_name == "tno":
        x_tno = lon_var[i]
        y_tno = lat_var[j]
        cell_x = np.array(
            [
                x_tno + cfg.tno_dx / 2,
                x_tno + cfg.tno_dx / 2,
                x_tno - cfg.tno_dx / 2,
                x_tno - cfg.tno_dx / 2,
            ]
        )
        cell_y = np.array(
            [
                y_tno + cfg.tno_dy / 2,
                y_tno - cfg.tno_dy / 2,
                y_tno - cfg.tno_dy / 2,
                y_tno + cfg.tno_dy / 2,
            ]
        )
        proj = ccrs.PlateCarree()
    elif inv_name == "vprm":
        globe = ccrs.Globe(
            ellipse=None, semimajor_axis=6370000, semiminor_axis=6370000
        )
        lambert = ccrs.LambertConformal(
            central_longitude=12.5,
            central_latitude=51.604,
            standard_parallels=[51.604],
            globe=globe,
        )

        center_lambert = lambert.transform_point(
            lon_var[j, i], lat_var[j, i], ccrs.PlateCarree()
        )
        cell_x = np.array(
            [
                center_lambert[0] + cfg.tno_dx / 2,
                center_lambert[0] + cfg.tno_dx / 2,
                center_lambert[0] - cfg.tno_dx / 2,
                center_lambert[0] - cfg.tno_dx / 2,
            ]
        )
        cell_y = np.array(
            [
                center_lambert[1] + cfg.tno_dy / 2,
                center_lambert[1] - cfg.tno_dy / 2,
                center_lambert[1] - cfg.tno_dy / 2,
                center_lambert[1] + cfg.tno_dy / 2,
            ]
        )
        proj = lambert
    elif inv_name == "edgar":
        x_tno = lon_var[i]
        y_tno = lat_var[j]
        cell_x = np.array(
            [x_tno + cfg.edgar_dx, x_tno + cfg.edgar_dx, x_tno, x_tno]
        )
        cell_y = np.array(
            [y_tno + cfg.edgar_dy, y_tno - cfg.edgar_dy, y_tno, y_tno]
        )
        proj = ccrs.PlateCarree()
    elif (
        inv_name == "meteotest"
        or inv_name == "maiolica"
        or inv_name == "carbocount"
    ):
        x1_ch, y1_ch = swiss2wgs84(lat_var[j], lon_var[i])  # i-lon, j-lat
        x2_ch, y2_ch = swiss2wgs84(lat_var[j] + 200, lon_var[i] + 200)
        cell_x = np.array([x2_ch, x2_ch, x1_ch, x1_ch])
        cell_y = np.array([y2_ch, y1_ch, y1_ch, y2_ch])
        proj = ccrs.PlateCarree()
    else:
        print(
            "Inventory %s is not supported yet. Consider defining your own or using tno or vprm."
            % inv_name
        )

    return cell_x, cell_y, proj


def get_dim_var(inv, inv_name, cfg):
    if inv_name == "tno":
        lon_dim = inv.dimensions["longitude"].size
        lat_dim = inv.dimensions["latitude"].size
        lon_var = inv["longitude"][:]
        lat_var = inv["latitude"][:]
    elif inv_name == "vprm":
        lon_dim = inv.dimensions["west_east"].size
        lat_dim = inv.dimensions["south_north"].size
        lon_var = inv["lon"][:]
        lat_var = inv["lat"][:]
    elif inv_name == "edgar":
        lon_var = np.arange(cfg.edgar_xmin, cfg.edgar_xmax, cfg.edgar_dx)
        lat_var = np.arange(cfg.edgar_ymin, cfg.edgar_ymax, cfg.edgar_dy)
        lon_dim = len(lon_var)
        lat_dim = len(lat_var)
    elif (
        inv_name == "meteotest"
        or inv_name == "maiolica"
        or inv_name == "carbocount"
    ):
        lon_var = np.array(
            [cfg.ch_xll + i * cfg.ch_cell for i in range(0, cfg.ch_xn)]
        )
        lat_var = np.array(
            [cfg.ch_yll + i * cfg.ch_cell for i in range(0, cfg.ch_yn)]
        )
        lon_dim = np.shape(lon_var)[0]
        lat_dim = np.shape(lat_var)[0]
    else:
        print(
            "Inventory %s is not supported yet. Consider defining your own or using tno or vprm."
            % inv_name
        )

    return lon_dim, lat_dim, lon_var, lat_var


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


def get_gridmapping(output_path, cosmo_grid, inv_grid, nprocs):
    """Returns the interpolation between the TNO and COSMO grid.

    If there already exists a file at output_path/mapping.npy ask the
    user if he wants to recompute.

    Parameters
    ----------
    output_path : str
        Path to the directory where the country-mask is stored
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
    mapping_path = os.path.join(output_path, "mapping.npy")
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


def swiss2wgs84(x, y):
    """
    Convert Swiss LV03 coordinates (x easting and y northing) to WGS 84 based
    on swisstopo approximated soluation (0.1" accuracy).

    remove the first digit of x,y
    """
    x = (x - 200000.0) / 1000000.0
    y = (y - 600000.0) / 1000000.0

    lon = (
        2.6779094
        + 4.728982 * y
        + 0.791484 * y * x
        + 0.1306 * y * x ** 2
        - 0.0436 * y ** 3
    ) / 0.36

    lat = (
        16.9023892
        + 3.238272 * x
        - 0.270978 * y ** 2
        - 0.002528 * x ** 2
        - 0.0447 * y ** 2 * x
        - 0.0140 * x ** 3
    ) / 0.36

    return lon, lat


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
