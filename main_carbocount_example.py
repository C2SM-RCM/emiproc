import os
import sys

import numpy as np
import utilities as util

from glob import glob
from netCDF4 import Dataset


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


def main(cfg_path):
    """ The main script for processing Swiss inventories.
    Takes a configuration file as input"""

    # Load the configuration file
    cfg = util.load_cfg(cfg_path)

    os.makedirs(cfg.output_path, exist_ok=True)

    # Load or compute the country mask
    country_mask = util.get_country_mask(
        cfg.output_path,
        cfg.cosmo_grid.name,
        cfg.cosmo_grid,
        cfg.shpfile_resolution,
        cfg.nprocs,
    )

    # Load or compute the interpolation map
    interpolation = util.get_gridmapping(
        cfg.output_path,
        cfg.swiss_grid.name,
        cfg.cosmo_grid,
        cfg.swiss_grid,
        cfg.nprocs,
    )

    # Set names for longitude and latitude
    lonname = "rlon"
    latname = "rlat"
    if (
        cfg.cosmo_grid.pollon == 180 or cfg.cosmo_grid.pollon == 0
    ) and cfg.cosmo_grid.pollat == 90:
        lonname = "lon"
        latname = "lat"

    # Starts writing out the output file
    output_path = os.path.join(
        cfg.output_path, f"emis_{cfg.year}_{cfg.gridname}.nc"
    )
    with Dataset(output_path, "w") as out:
        util.prepare_output_file(cfg.cosmo_grid, cfg.nc_metadata, out)
        util.add_country_mask(country_mask, out)

        # Swiss inventory specific
        total_flux = {}
        for var in cfg.species:
            total_flux[var] = np.zeros((cfg.cosmo_grid.ny, cfg.cosmo_grid.nx))

        for cat in cfg.ch_cat:
            for var in cfg.species:
                constfile = os.path.join(
                    cfg.input_path, "".join([cat.lower(), "10_", "*_kg.txt"])
                )
                out_var_name = var + "_" + cfg.mapping[cat]

                emi = np.zeros((cfg.swiss_grid.nx, cfg.swiss_grid.ny))
                for filename in sorted(glob(constfile)):
                    print(filename)
                    emi += read_emi_from_file(filename)  # (lon,lat)

                out_var = np.zeros((cfg.cosmo_grid.ny, cfg.cosmo_grid.nx))
                for lon in range(np.shape(emi)[0]):
                    for lat in range(np.shape(emi)[1]):
                        for (x, y, r) in interpolation[lon, lat]:
                            out_var[y, x] += emi[lon, lat] * r

                cosmo_area = 1.0 / cfg.cosmo_grid.gridcell_areas()

                # convert unit from kg.year-1.cell-1 to kg.m-2.s-1
                out_var *= cosmo_area.T / util.SEC_PER_YR

                # only allow positive fluxes
                out_var[out_var < 0] = 0

                if out_var_name not in out.variables.keys():
                    out.createVariable(out_var_name, float, (latname, lonname))
                    if lonname == "rlon" and latname == "rlat":
                        out[out_var_name].grid_mapping = "rotated_pole"
                    out[out_var_name].units = "kg m-2 s-1"
                    out[out_var_name][:] = out_var
                else:
                    out[out_var_name][:] += out_var

                total_flux[var] += out_var

        # Calcluate total emission/flux per species
        for s in cfg.species:
            out.createVariable(s, float, (latname, lonname))
            out[s].units = "kg m-2 s-1"
            if lonname == "rlon" and latname == "rlat":
                out[s].grid_mapping = "rotated_pole"
            out[s][:] = total_flux[s]


if __name__ == "__main__":
    try:
        config_path = sys.argv[1]
    except IndexError:
        raise RuntimeError("Please supply a config file.")
    main(config_path)
