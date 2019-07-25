"""
Script to create two sets of profiles 
in order to merge inventories when runnning COSMO
"""


from country_code import country_codes as cc
from netCDF4 import Dataset
import shutil
import numpy as np

inv_1 = 'TNO'
inv_2 = 'Carbocount'

# List of countries to set to 0 in inv_1
# All others countries will be set to 0 in inv_2
countries = ['CH']

profile_path_in = ['./example_output/hourofday.nc','./example_output/dayofweek.nc','./example_output/monthofyear.nc']
profile_path_out = ['./example_output/hourofday_merged.nc','./example_output/dayofweek_merged.nc','./example_output/monthofyear_merged.nc']
dimension = 24


for path_1,path_2 in zip(profile_path_in, profile_path_out):
    shutil.copy(path_1,path_2)

def main():
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
                                nc_vars[0][:,i] = np.zeros(dimension)
                                deleted = True
                        if not deleted:
                            print(nc_vars[1])
                            nc_vars[1][:,i] = np.zeros(dimension)

if __name__ == "__main__":
    main()
