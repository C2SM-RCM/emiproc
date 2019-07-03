import os
import sys
import time

import numpy as np

from netCDF4 import Dataset

import utilities as util


def var_name(s, cat, cat_kind):
    """Returns the name of a variable for a given species and cat
    input :
       - s : species name ("CO2", "CH4" ...)
       - cat : Category name or number
       - cat_kind : Kind of category. must be "SNAP" or "NFR"
    output :
       - returns a string which concatenates the species with
         the category number
    """
    out_var_name = s + "_"
    if cat_kind == "SNAP":
        if cat > 9:
            out_var_name += str(cat) + "_"
        else:
            out_var_name += "0" + str(cat) + "_"
    elif cat_kind == "NFR":
        out_var_name += cat + "_"
    else:
        print("Wrong cat_kind in the config file. Must be SNAP or NFR")
        raise ValueError

    return out_var_name


def main(cfg_path):
    """ The main script for processing TNO inventory.
    Takes a configuration file path as input"""

    cfg = util.load_cfg(cfg_path)

    # Load or compute the country mask
    country_mask = util.get_country_mask(
        cfg.output_path, cfg.cosmo_grid, cfg.shpfile_resolution, cfg.nprocs
    )
    # Load or compute the mapping between the inventory and COSMO grid
    interpolation = util.get_gridmapping(
        cfg.output_path, cfg.cosmo_grid, cfg.tno_grid, cfg.nprocs
    )

    # Set names for longitude and latitude
    if (
        cfg.cosmo_grid.pollon == 180 or cfg.cosmo_grid.pollon == 0
    ) and cfg.cosmo_grid.pollat == 90:
        lonname = "lon"
        latname = "lat"
        print(
            "Non-rotated grid: pollon = %f, pollat = %f"
            % (cfg.cosmo_grid.pollon, cfg.cosmo_grid.pollat)
        )
    else:
        lonname = "rlon"
        latname = "rlat"
        print(
            "Rotated grid: pollon = %f, pollat = %f"
            % (cfg.cosmo_grid.pollon, cfg.cosmo_grid.pollat)
        )

    # Starts writing out the output file
    output_file = os.path.join(cfg.output_path, cfg.output_name)
    with Dataset(output_file, "w") as out:
        util.prepare_output_file(cfg.cosmo_grid, out)
        util.add_country_mask(country_mask, out)

        # Load or compute the interpolation maps
        with Dataset(cfg.tnofile) as tno:
            # From here onward, quite specific for TNO

            # Mask corresponding to the area/point sources
            selection_area = tno["source_type_index"][:] == 1
            selection_point = tno["source_type_index"][:] == 2

            # Area of the COSMO grid cells
            cosmo_area = 1.0 / cfg.cosmo_grid.gridcell_areas()

            for cat in cfg.output_cat:
                # In emission_category_index, we have the
                # index of the category, starting with 1.

                # mask corresponding to the given category
                selection_cat = np.array(
                    [
                        tno["emission_category_index"][:]
                        == cfg.tno_cat.index(cat) + 1
                    ]
                )

                # mask corresponding to the given category for area/point
                selection_cat_area = np.array(
                    [selection_cat.any(0), selection_area]
                ).all(0)
                selection_cat_point = np.array(
                    [selection_cat.any(0), selection_point]
                ).all(0)

                species_list = cfg.species
                for s in species_list:
                    print("Species", s, "Category", cat)
                    out_var_area = np.zeros(
                        (cfg.cosmo_grid.ny, cfg.cosmo_grid.nx)
                    )
                    out_var_point = np.zeros(
                        (cfg.cosmo_grid.ny, cfg.cosmo_grid.nx)
                    )

                    var = tno[s.lower()][:]

                    start = time.time()
                    for (i, source) in enumerate(var):
                        if selection_cat_area[i]:
                            lon_ind = tno["longitude_index"][i] - 1
                            lat_ind = tno["latitude_index"][i] - 1
                            for (x, y, r) in interpolation[lon_ind, lat_ind]:
                                out_var_area[y, x] += var[i] * r
                        if selection_cat_point[i]:
                            try:
                                indx, indy = cfg.cosmo_grid.indices_of_point(
                                    tno["longitude_source"][i],
                                    tno["latitude_source"][i],
                                )
                            except IndexError:
                                # Point lies outside the cosmo grid
                                continue

                            out_var_point[indy, indx] += var[i]

                    end = time.time()
                    print("it takes ", end - start, "sec")

                    # convert unit from kg.year-1.cell-1 to kg.m-2.s-1
                    out_var_point *= cosmo_area.T / util.SEC_PER_YR
                    out_var_area *= cosmo_area.T / util.SEC_PER_YR

                    out_var_name = var_name(s, cat, cfg.cat_kind)
                    for (t, sel, out_var) in zip(
                        ["AREA", "POINT"],
                        [selection_cat_area, selection_cat_point],
                        [out_var_area, out_var_point],
                    ):
                        if sel.any():
                            out.createVariable(
                                out_var_name + t, float, (latname, lonname)
                            )
                            out[out_var_name + t].units = "kg m-2 s-1"
                            if lonname == "rlon" and latname == "rlat":
                                out[
                                    out_var_name + t
                                ].grid_mapping = "rotated_pole"
                            out[out_var_name + t][:] = out_var


if __name__ == "__main__":
    try:
        config_path = sys.argv[1]
    except IndexError:
        raise RuntimeError("Please supply a config file.")
    main(config_path)
