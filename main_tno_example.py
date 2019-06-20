from make_online_emissions import *


def var_name(s, cat, cat_kind):
    """Returns the name of a variable for a given species and cat
    input : 
       - s : species name ("CO2", "CH4" ...)
       - cat : Category name or number
       - cat_kind : Kind of category. must be "SNAP" or "NFR" 
    output :
       - returns a string which concatenates the species with the category number
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
    Takes a configuration file as input"""

    """Load the configuration file"""
    cfg = load_cfg(cfg_path)

    """Load or compute the country mask"""
    country_mask = get_country_mask(cfg)

    """Set names for longitude and latitude"""
    if (cfg.pollon == 180 or cfg.pollon == 0) and cfg.pollat == 90:
        lonname = "lon"
        latname = "lat"
        print(
            "Non-rotated grid: pollon = %f, pollat = %f"
            % (cfg.pollon, cfg.pollat)
        )
    else:
        lonname = "rlon"
        latname = "rlat"
        print(
            "Rotated grid: pollon = %f, pollat = %f" % (cfg.pollon, cfg.pollat)
        )

    """Starts writing out the output file"""
    output_path = cfg.output_path
    output_file = os.path.join(cfg.output_path, cfg.output_name)
    with nc.Dataset(output_file, "w") as out:
        prepare_output_file(cfg, out, country_mask)

        """Load or compute the interpolation maps"""
        with nc.Dataset(cfg.tnofile) as tno:
            interpolation = get_interpolation(cfg, tno)

            """From here onward, quite specific for TNO"""

            """Mask corresponding to the area/point sources"""
            selection_area = tno["source_type_index"][:] == 1
            selection_point = tno["source_type_index"][:] == 2

            """Area of the COSMO grid cells"""
            cosmo_area = 1.0 / gridbox_area(cfg)

            for cat in cfg.output_cat:
                """In emission_category_index, we have the index of the category, starting with 1."""

                """mask corresponding to the given category"""
                selection_cat = np.array(
                    [
                        tno["emission_category_index"][:]
                        == cfg.tno_cat.index(cat) + 1
                    ]
                )

                """mask corresponding to the given category for area/point"""
                selection_cat_area = np.array(
                    [selection_cat.any(0), selection_area]
                ).all(0)
                selection_cat_point = np.array(
                    [selection_cat.any(0), selection_point]
                ).all(0)

                species_list = cfg.species
                for s in species_list:
                    print("Species", s, "Category", cat)
                    out_var_area = np.zeros((cfg.ny, cfg.nx))
                    out_var_point = np.zeros((cfg.ny, cfg.nx))

                    var = tno[s.lower()][:]

                    start = time.time()
                    for (i, source) in enumerate(var):
                        if selection_cat_area[i]:
                            lon_ind = tno["longitude_index"][i] - 1
                            lat_ind = tno["latitude_index"][i] - 1
                            for (x, y, r) in interpolation[lon_ind, lat_ind]:
                                out_var_area[y, x] += var[i] * r
                        if selection_cat_point[i]:
                            (indx, indy) = interpolate_to_cosmo_point(
                                tno["latitude_source"][i],
                                tno["longitude_source"][i],
                                cfg,
                            )
                            if (
                                indx >= 0
                                and indx < cfg.nx
                                and indy >= 0
                                and indy < cfg.ny
                            ):
                                out_var_point[indy, indx] += var[i]

                    end = time.time()
                    print("it takes ", end - start, "sec")

                    # convert unit from kg.year-1.cell-1 to kg.m-2.s-1
                    out_var_point *= cosmo_area.T / SEC_PER_YR
                    out_var_area *= cosmo_area.T / SEC_PER_YR

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
        config = sys.argv[1]
    except IndexError:
        config = "./config_tno"
    main(config)
