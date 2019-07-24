from netCDF4 import Dataset
import utilities as util
import sys


def main(cfg_path):
    """
    This script takes two processed inventories 
    and copies all of their variables into a new one
    """
    cfg = util.load_cfg(cfg_path)

    with Dataset(cfg.inv_out,'w') as outf:
        util.prepare_output_file(cfg.cosmo_grid, outf)

        for inv,name in zip([cfg.inv_1,cfg.inv_2],
                            [cfg.inv_name_1,cfg.inv_name_2]):
            with Dataset(inv) as inf:
                if inv == cfg.inv_1:
                    util.add_country_mask(inf['country_ids'][:].T, outf)

                for v in inf.variables:
                    if v in ['rotated_pole','rlon','rlat','country_ids','lon','lat']:
                        continue

                    var = inf[v]
                    nc_var = outf.createVariable(v+'_'+name, var.dtype, var.dimensions)
                    nc_var.units = var.units
                    nc_var.grid_mapping = var.grid_mapping
                    nc_var[:] = var[:]


if __name__ == "__main__":
    try:
        config_path = sys.argv[1]
    except IndexError:
        raise RuntimeError("Please supply a config file.")
    main(config_path)
