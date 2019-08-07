
from netCDF4 import Dataset
import sys

from . import utilities as util


def main(cfg):
    """
    This script takes two processed inventories 
    and copies all of their variables into a new one
    """

    with Dataset(cfg.inv_out,'w') as outf:
        util.prepare_output_file(cfg.cosmo_grid, cfg.nc_metadata, outf)

        for inv,name in zip([cfg.inv_1,cfg.inv_2],
                            [cfg.inv_name_1,cfg.inv_name_2]):
            with Dataset(inv) as inf:
                if inv == cfg.inv_1:
                    util.add_country_mask(inf['country_ids'][:].T, outf)

                for v in inf.variables:
                    if v in ['rotated_pole','rlon','rlat','country_ids','lon','lat']:
                        continue

                    var = inf[v]
                    nc_var = outf.createVariable(v+name, var.dtype, var.dimensions)
                    nc_var.units = var.units
                    nc_var.grid_mapping = var.grid_mapping
                    nc_var[:] = var[:]

