#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create artificial emission files.

Created on Wed Nov 14 09:25:48 2018

@author: jae
"""

import netCDF4 as nc
import numpy as np
import os
from datetime import datetime
from make_online_emissions import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import countries
import grid

inpath = "/newhome/jae/gitlab/online-emission-processing/testdata/"
base_inventory = os.path.join(inpath, 'emis_2018_tno_CO2_CO_CH4_1km.nc')
outfile = os.path.join(inpath, 'merged_inventories.nc')

list_inventories = ['emis_2018_maiolica_CH4_1km.nc',
                    'emis_2018_carbocount_CO2_1km.nc',
                    'emis_2018_meteotest_CO_1km.nc'
                   ]

def main(cfg_path):
    """ The main script for processing TNO inventory. 
    Takes a configuration file as input"""

    """Load the configuration file"""
    cfg = load_cfg(cfg_path)

    """Load or compute the country mask"""
    country_mask = get_country_mask(cfg)
    country_mask = np.transpose(country_mask)
    mask = country_mask == country_codes['CH']

    with nc.Dataset(base_inventory, "r") as src, nc.Dataset(outfile, "w") as dst:
        print(base_inventory)
        # Copy global attributes all at once via dictionary
        dst.setncatts(src.__dict__)

        # Copy all dimensions 
        for name, dimension in src.dimensions.items():
            dst.createDimension(name, len(dimension))

        # Copy all variables
        for name, variable in src.variables.items():
            print(name)
            x = dst.createVariable(name, variable.datatype,
                                   variable.dimensions,
                                   zlib=True, complevel=9)
            # Copy variable attributes all at once via dictionary
            dst[name].setncatts(src[name].__dict__)
            if 'rotated_pole' not in name:
                dst[name][:] = src[name][:]

        # Merge inventories
        for fname in list_inventories:
            infile = os.path.join(inpath, fname)
            print(infile)
            with nc.Dataset(infile, "r") as inv:
                for name, variable in src.variables.items():
                    if ('CO2' in name and 'carbocount' in infile) or \
                       ('CH4' in name and 'maiolica' in infile) or \
                       (('CO_' in name or name == 'CO') and 'meteotest' in infile):
                        print('Overwriting variable %s' % name)
                        np.copyto(dst[name][:], inv[name][:], where=mask)


if __name__ == '__main__':                                                     
    cfg_name = sys.argv[1]
    main("./config_" + cfg_name)

