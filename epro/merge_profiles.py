
import shutil

from netCDF4 import Dataset
import numpy as np

from .country_code import country_codes as cc

SHAPE = 24

def main(inv1, inv2, countries, profile_path_in, profile_path_out):
    """
    Merge two sets of time profiles in single files where hourofday scaling
    factors are set to 0 in inv1 for given countries and set to 0 in inv2 for
    all other countries. The merged files can be used to merge the inventories
    online when running COSMO.

    Parameter
    ---------
    inv1 (str) :: name of first inventory

    inv2 (str) :: name of second inventory

    countries (list) :: country codes (e.g. 'CH') for all countries set to 0 in
                        inv1

    profile_path_in :: list of input paths for hourofday, dayofweek and
                       monthofyear netCDF files

    profile_path_out :: list of output paths for merge hourofday, dayofweek and
                        monthofyear netCDF files
    """

    for path_1, path_2 in zip(profile_path_in, profile_path_out):
        shutil.copy(path_1,path_2)

    for done,path in enumerate(profile_path_out):
        with Dataset(path,'a') as prof:
            for v in prof.variables.copy():
                if v == 'country':
                    continue
                var = prof[v]

                nc_vars = []
                for inv in [inv_1,inv_2]:
                    nc_var = prof.createVariable(v+'_'+inv, var.dtype, var.dimensions)
                    nc_var.long_name = var.long_name + " for inventory %s" % (inv)
                    nc_var.units = "1"
                    nc_var.comment = var.comment
                    nc_var[:] = var[:]
                    nc_vars.append(nc_var)

                if not done:
                    for i,c in enumerate(prof['country'][:]):
                        country_name = [name for name, code in cc.items() if (code == c)]
                        deleted = False
                        for todel in countries:
                            if todel in country_name:
                                print(nc_vars[0])
                                nc_vars[0][:,i] = np.zeros(shape)
                                deleted = True
                        if not deleted:
                            print(nc_vars[1])
                            nc_vars[1][:,i] = np.zeros(shape)

