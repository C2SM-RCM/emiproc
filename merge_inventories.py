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
base_inventory = os.path.join(inpath, "emis_2018_tno_CO2_CO_CH4_1km.nc")
outfile = os.path.join(inpath, "merged_inventories.nc")

list_inventories = [
    "emis_2018_maiolica_CH4_1km.nc",
    "emis_2018_carbocount_CO2_1km.nc",
    "emis_2018_meteotest_CO_1km.nc",
]


def main(cfg_path):
    """ The main script for processing TNO inventory. 
    Takes a configuration file as input"""

    """Load the configuration file"""
    cfg = load_cfg(cfg_path)

    """Load or compute the country mask"""
    country_mask = get_country_mask(cfg)
    country_mask = np.transpose(country_mask)
    mask = country_mask == country_codes["CH"]

    sum_co2_tno = 0
    sum_co2_merged = 0
    sum_co_tno = 0
    sum_co_merged = 0
    sum_ch4_tno = 0
    sum_ch4_merged = 0
    with nc.Dataset(base_inventory, "r") as src, nc.Dataset(
        outfile, "w"
    ) as dst:
        print(base_inventory)
        # Copy global attributes all at once via dictionary
        dst.setncatts(src.__dict__)

        # Copy all dimensions
        for name, dimension in src.dimensions.items():
            dst.createDimension(name, len(dimension))

        # Copy all variables
        for name, variable in src.variables.items():
            print(name)
            x = dst.createVariable(
                name,
                variable.datatype,
                variable.dimensions,
                zlib=True,
                complevel=9,
            )
            # Copy variable attributes all at once via dictionary
            dst[name].setncatts(src[name].__dict__)
            if "rotated_pole" not in name:
                dst[name][:] = src[name][:]

        sum_co2_tno = np.sum(src["CO2"][:])
        sum_co_tno = np.sum(src["CO"][:])
        sum_ch4_tno = np.sum(src["CH4"][:])

        # Merge inventories
        for fname in list_inventories:
            infile = os.path.join(inpath, fname)
            print(infile)
            with nc.Dataset(infile, "r") as inv:
                for name, variable in src.variables.items():
                    if (
                        ("CO2" in name and "carbocount" in infile)
                        or ("CH4" in name and "maiolica" in infile)
                        or (
                            ("CO_" in name or name == "CO")
                            and "meteotest" in infile
                        )
                    ):
                        print("Overwriting variable %s" % name)
                        var_dst = dst.variables[name][:]
                        np.copyto(
                            var_dst, inv.variables[name][:], where=mask[:]
                        )
                        dst.variables[name][:] = var_dst

        sum_co2_merged = np.sum(dst["CO2"][:])
        sum_co_merged = np.sum(dst["CO"][:])
        sum_ch4_merged = np.sum(dst["CH4"][:])

    print("sum_co2_tno", sum_co2_tno)
    print("sum_co2_merged", sum_co2_merged)
    print("sum_co_tno", sum_co_tno)
    print("sum_co_merged", sum_co_merged)
    print("sum_ch4_tno", sum_ch4_tno)
    print("sum_ch4_merged", sum_ch4_merged)


if __name__ == "__main__":
    cfg_name = sys.argv[1]
    main("./config_" + cfg_name)
