#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import netCDF4

# dimensions:
#     category = 10 ;
#     level = 16 ;
#     char = 20 ;
# variables:
#     char category(category, nchar) ;
#           category:long_name = "name of category" ;
#     float layer_bot(level) ;
#         layer_bot:units = "m" ;
#         layer_bot:long_name = "bottom of layer above ground" ;
#     float layer_top(level) ;
#         layer_top:units = "m" ;
#         layer_top:long_name = "top of layer above ground" ;
#     float SNAP-*(level) ;
#         SNAP-*:units = "1" ;
#         SNAP-*:long_name = "vertical scale factor for sources of SNAP-* category";

def get_all_levels(levels): #levels are the top of the layer
    layer_top = levels
    layer_bot = [0]+levels[:-1]
    layer_mid = [(i+j)/2 for (i,j) in zip(layer_top,layer_bot)]
    
    return layer_bot,layer_mid,layer_top
        


def write_netcdf(filename, categories, cat_name, levels, scale_factors):
    layer_bot,layer_mid,layer_top = get_all_levels(levels)  

    with netCDF4.Dataset(filename, 'w') as nc:

        # global attributes (add input data)
        nc.setncatts({
            'DESCRIPTION':  'Vertical profiles for emissions',
            'DATAORIGIN':   'based on profiles developed for COST-728 action',
            'CREATOR':      'Jean-Matthieu Haussaire',
            'EMAIL':        'jean-matthieu.haussaire@empa.ch',
            'AFFILIATION':  'Empa Duebendorf, Switzerland',
            'DATE CREATED': time.ctime(time.time()),            
        })

        # create dimensions    
        nc.createDimension('level', size=len(levels))

    
        # create variables and attributes
        nc_bot = nc.createVariable('layer_bot', 'f4', ('level'))
        nc_bot.long_name = 'bottom of layer above ground'
        nc_bot.units = 'm'
        nc_bot[:] =  layer_bot

        nc_mid = nc.createVariable('layer_mid', 'f4', ('level'))
        nc_mid.long_name = 'middle of layer above ground'
        nc_mid.units = 'm'
        nc_mid[:] =  layer_mid

        nc_top = nc.createVariable('layer_top', 'f4', ('level'))
        nc_top.long_name = 'top of layer above ground'
        nc_top.units = 'm'
        nc_top[:] =  layer_top

        for (i,cat) in enumerate(categories):            
            nc_sca = nc.createVariable(cat_name+cat, 'f4', ('level'))
            nc_sca.long_name = 'vertical scale factor for sources of SNAP-%s category' % cat
            nc_sca.units = '1'
            nc_sca[:] = scale_factors[i]


def read_profiles(filename, nlevel=16):
    levels = []
    categories = []
    profiles = []
    
    with open(filename) as profile_file:
        levels = [int(i) for i in profile_file.readline().split("\t")[1:]]
    
        all_sevens= []
        for line in profile_file:            
            values = line.split()
            cat = values[0]
            profile = values[1:]            
            if cat=="34":
                categories.append("3")
                categories.append("4")
                profiles.append(profile)
                profiles.append(profile)
            elif cat=="7.01":
                categories.append("7")
                all_sevens.append([float(i) for i in profile])
            elif "7." in cat:
                all_sevens.append([float(i) for i in profile])
            elif cat=="F1":
                categories.append("F")
                all_sevens.append([float(i) for i in profile])
            elif "F" in cat:
                all_sevens.append([float(i) for i in profile])
            else:
                categories.append(cat)
                if len(all_sevens)>0:
                    profiles.append(np.array(all_sevens).mean(0))
                    all_sevens=[]                    
                profiles.append(profile)
                    

    return categories, np.array(profiles, 'f4'),levels




def main(filename):
    categories, profiles,levels = read_profiles('vert_profiles/vert_prof_che_gnfr.dat')
    write_netcdf(filename, categories, "GNFR_", levels, profiles)



if __name__ == "__main__":
    main('output/vertical_profiles.nc')
