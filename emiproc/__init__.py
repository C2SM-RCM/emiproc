
from glob import glob
import os
import sys
import time

import netCDF4
import numpy as np

from netCDF4 import Dataset

from . import utilities as util
from . import edgar
from emiproc import append_inventories




def process_swiss(cfg, interpolation, country_mask, out, latname, lonname):
    """
    Process "Swiss National Emission Inventory" created by Meteotest Inc.
    """
    if cfg.add_total_emissions:
        total_flux = {}
        for var in cfg.species:
            var_name = cfg.in2out_species.get(var, var)
            total_flux[var_name] = np.zeros((cfg.output_grid.ny, cfg.output_grid.nx))

    for cat in cfg.categories:
        for var in cfg.species:
            print('Species', var, 'Category', cat)

            if cfg.inventory == 'swiss-cc':
                constfile = os.path.join(
                    cfg.input_path, "".join([cat, "10_", "*_kg.txt"])
                )
            elif cfg.inventory == 'swiss-art':
                constfile = os.path.join(
                    cfg.input_path, "".join(['e', cat, '15_', var, '*'])
                )

            emi = np.zeros((cfg.input_grid.nx, cfg.input_grid.ny))
            for filename in sorted(glob(constfile)):
                print(filename)
                emi += util.read_emi_from_file(filename)  # (lon,lat) 

            out_var = np.zeros((cfg.output_grid.ny, cfg.output_grid.nx))
            for lon in range(np.shape(emi)[0]):
                for lat in range(np.shape(emi)[1]):
                    for (x, y, r) in interpolation[lon, lat]:
                        out_var[y, x] += emi[lon, lat] * r

            output_area = 1.0 / cfg.output_grid.gridcell_areas()

            # convert units
            if cfg.model == 'cosmo-ghg':
                # COSMO-GHG: kg.year-1.cell-1 to kg.m-2.s-1
                out_var *= output_area.T / util.SEC_PER_YR
                unit = 'kg m-2 s-1'
            elif cfg.model == 'cosmo-art':
                # COSMO-ART:  g.year-1.cell-1 to kg.h-1.cell-1
                out_var *= 1.0 / (24.0 * util.DAY_PER_YR) / 1000.0
                unit = 'kg h-1 cell-1'
            else:
                raise RuntimeError

            # only allow positive fluxes
            out_var[out_var < 0] = 0

            # write new or add to exisiting variable
            out_var_name = util.get_out_varname(var, cat, cfg)
            print('Write as variable:', out_var_name)
            util.write_variable(out, out_var, out_var_name, latname, lonname,
                                unit)

            if cfg.add_total_emissions:
                var_name = cfg.in2out_species.get(var, var)
                total_flux[var_name] += out_var


    # Calcluate total emission/flux per species
    if cfg.add_total_emissions:
        for s in cfg.species:
            s = cfg.in2out_species.get(s, s)
            print('Write total emissions for variable:', s)
            util.write_variable(out, total_flux[s], var_name, latname, lonname,
                                unit)


def process_rotgrid(cfg, interpolation, country_mask, out, latname, lonname):
    """
    Process Emission defined on a COSMO rotated grid.
    Could have been created by an inversion system and needs to be remapped.
    """
    with Dataset(cfg.input_path) as cosmo_in:
        for cat in cfg.categories:
            for var in cfg.species:
                print('Species', var, 'Category', cat)
                emi = cosmo_in[var]*(cfg.input_grid.gridcell_areas().T)
                out_var = np.zeros((cfg.output_grid.ny, cfg.output_grid.nx))
                for lon in range(np.shape(emi)[1]):
                    for lat in range(np.shape(emi)[0]):
                        for (x, y, r) in interpolation[lon, lat]:
                            out_var[y, x] += emi[lat, lon] * r 

                output_area = 1.0 / cfg.output_grid.gridcell_areas()

                # convert units

                # COSMO-GHG: kg.s-1.cell-1 to kg.m-2.s-1
                out_var *= output_area.T 
                unit = 'kg m-2 s-1'

                # only allow positive fluxes
                out_var[out_var < 0] = 0

                # write new or add to exisiting variable
                out_var_name = util.get_out_varname(var, cat, cfg)
                print('Write as variable:', out_var_name)
                util.write_variable(out, out_var, out_var_name, latname, lonname,
                                    unit)



def process_tno(cfg, interpolation, country_mask, out, latname, lonname):
    """
    Process TNO inventories.
    """
    # Load or compute the interpolation maps
    with Dataset(cfg.input_path) as tno:

        # Get emission category codes from TNO file
        if 'emis_cat_code' in tno.variables:

            # GNFR-based
            tno_cat_codes = tno.variables['emis_cat_code'][:]
            tno_cat_codes = netCDF4.stringtochar(tno_cat_codes)
            tno_cat_codes = [s.tostring().decode('ascii').strip('\00')
                             for s in tno_cat_codes]

        elif 'emis_cat_shortsnap' in tno.variables:

            # SNAP-based (e.g. TNO/MACC-III)
            tno_cat_codes = tno.variables['emis_cat_shortsnap'][:]
            tno_cat_codes = [str(c) for c in tno_cat_codes]


        # Mask corresponding to the area/point sources
        selection_area = tno["source_type_index"][:] == 1
        selection_point = tno["source_type_index"][:] == 2

        # Area of the COSMO grid cells
        output_area = 1.0 / cfg.output_grid.gridcell_areas()

        for cat in cfg.categories:

            # In emission_category_index, we have the
            # index of the category, starting with 1.
            # mask corresponding to the given category
            selection_cat = np.array(
                [
                    tno["emission_category_index"][:]
                    == tno_cat_codes.index(cat) + 1
                    #for c in cat_list
                ]
            )

            # mask corresponding to the given category for area/point
            selection_cat_area = np.array(
                [selection_cat.any(0), selection_area]
            ).all(0)
            selection_cat_point = np.array(
                [selection_cat.any(0), selection_point]
            ).all(0)

            for s in cfg.species:
                print("Species", s, "Category", cat)
                out_var_area = np.zeros(
                    (cfg.output_grid.ny, cfg.output_grid.nx)
                )
                out_var_point = np.zeros(
                    (cfg.output_grid.ny, cfg.output_grid.nx)
                )

                var = tno[s][:]
                lon_indices = tno['longitude_index'][:].data - 1
                lat_indices = tno['latitude_index'][:].data - 1

                start = time.time()
                for (i, source) in enumerate(var.data):
                    if selection_cat_area[i]:
                        lon_ind = lon_indices[i]
                        lat_ind = lat_indices[i]
                        for (x, y, r) in interpolation[lon_ind, lat_ind]:
                            out_var_area[y, x] += var[i] * r
                    if selection_cat_point[i]:
                        try:
                            indx, indy = cfg.output_grid.indices_of_point(
                                tno["longitude_source"][i],
                                tno["latitude_source"][i],
                            )
                        except IndexError:
                            # Point lies outside the cosmo grid
                            continue

                        out_var_point[indy, indx] += var[i]

                end = time.time()
                print("Gridding took %.1f seconds" % (end - start))

                # convert units
                if cfg.model == 'cosmo-ghg':
                    # COSMO-GHG: kg.year-1.cell-1 to kg.m-2.s-1
                    out_var_point *= output_area.T / util.SEC_PER_YR
                    out_var_area *= output_area.T / util.SEC_PER_YR
                    unit = 'kg m-2 s-1'

                if cfg.model == 'icon':
                    # ICON-OEM: kg.year-1.cell-1 to kg.cell-1.s-1
                    out_var_point *= 1.0 / util.SEC_PER_YR
                    out_var_area *= 1.0 / util.SEC_PER_YR
                    unit = 'kg cell-1 s-1'

                elif cfg.model == 'cosmo-art':
                    # COSMO-ART:  kg.year-1.cell-1 to kg.h-1.cell-1
                    out_var_point *= 1.0 / (24.0 * util.DAY_PER_YR)
                    out_var_area *= 1.0 / (24.0 * util.DAY_PER_YR)
                    unit = 'kg h-1 cell-1'
                else:
                    raise RuntimeError


                for (t, sel, out_var) in zip(
                    ["AREA", "POINT"],
                    [selection_cat_area, selection_cat_point],
                    [out_var_area, out_var_point],
                ):
                    if sel.any():
                        out_var_name = util.get_out_varname(s, cat, cfg,
                            				source_type=t)
                        print('Write as variable:', out_var_name)
                        if cfg.model.startswith("cosmo"):
                            util.write_variable(out, out_var, out_var_name,
                                                latname, lonname, unit)
                        elif cfg.model.startswith("icon"):
                            print(out_var.shape)
                            util.write_variable_ICON(out, out_var,
                                                out_var_name, unit)



def main(cfg):
    """
    Main function for gridding emissions.
    """
    os.makedirs(cfg.output_path, exist_ok=True)

    # Load or compute the country mask
    country_mask = util.get_country_mask(
        cfg.output_path,
        cfg.output_grid.name,
        cfg.output_grid,
        cfg.shpfile_resolution,
        cfg.nprocs,
    )
    # Load or compute the mapping between the inventory and output grid
    interpolation = util.get_gridmapping(
        cfg.output_path,
        cfg.input_grid.name,
        cfg.output_grid,
        cfg.input_grid,
        cfg.nprocs,
    )

    # Set names for longitude and latitude
    if cfg.model.startswith("cosmo"):
        if (
            cfg.output_grid.pollon == 180 or cfg.output_grid.pollon == 0
        ) and cfg.output_grid.pollat == 90:
            lonname = "lon"
            latname = "lat"
            print(
                "Non-rotated grid: pollon = %f, pollat = %f"
                % (cfg.output_grid.pollon, cfg.output_grid.pollat)
             )
        else:
            lonname = "rlon"
            latname = "rlat"
            print(
                "Rotated grid: pollon = %f, pollat = %f"
                % (cfg.output_grid.pollon, cfg.output_grid.pollat)
            )
    elif cfg.model.startswith("icon"):
        lonname = None
        latname = None
        print("ICON grid")

    # Starts writing out the output file
    output_file = os.path.join(cfg.output_path, cfg.output_name)

    with Dataset(output_file, "w") as out:
        if cfg.model.startswith("cosmo"):
            util.prepare_output_file(cfg.output_grid, cfg.nc_metadata, out)
            util.add_country_mask(country_mask, out, "cosmo")
        elif cfg.model.startswith("icon"):
            util.prepare_ICON_output_file(cfg.output_grid, cfg.nc_metadata, out)
            util.add_country_mask(country_mask, out, "icon")

        if cfg.inventory == 'TNO':
            process_tno(cfg, interpolation, country_mask, out, latname,
                        lonname)

        elif cfg.inventory in ['swiss-cc', 'swiss-art']:
            process_swiss(cfg, interpolation, country_mask, out, latname,
                        lonname)
        elif cfg.inventory == 'COSMO':
            process_rotgrid(cfg, interpolation, country_mask, out, latname,
                        lonname)

        elif cfg.inventory == 'EDGAR':
            edgar.process_edgar(cfg, interpolation, country_mask, out, latname,
                        lonname)












