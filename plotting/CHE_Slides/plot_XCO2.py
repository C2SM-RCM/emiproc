# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
#from shapely.geometry import Polygon
import netCDF4 as nc
from matplotlib.patches import Polygon
import matplotlib.colors as colors
import matplotlib
from itertools import product
from datetime import datetime


convert_unit = {
    "CO2_ALL" : 29/44.,
    "ch4" : 29/16.}

pole_lon = -170
pole_lat = 43

#/project/s862/CHE/CHE_Europe_output/cosmo2D_2015-01/cosmo_2d_2015010100.nc 
path = "/project/s862/CHE/CHE_Europe_output/cosmo2D_2015-01/"

var = "CO2_ALL"



bounds = [ 0.,0.001,  0.01,  0.1,  0.3,  0.5, 1. ]
norm   = matplotlib.colors.BoundaryNorm( bounds, ncolors=256 )

transform = ccrs.RotatedPole(pole_longitude=pole_lon, pole_latitude=pole_lat)

for i in [1]:#range(1,10):

    for j in [12]:#range(0,24,3):
        date = datetime(2015,1,i,j)
        date_str = date.strftime("%Y%m%d%H")
        date_disp = date.strftime("%Y-%m-%d %H:00")

        cosmo_1 = nc.Dataset(path+"cosmo_2d_"+date_str+".nc")
    
        # co2_all= (cosmo_1[var][0,-1,:])
        # co2_bg = (cosmo_1["CO2_BG"][0,-1,:])
        # co2 = co2_bg+co2_all
        
        to_plot = (
            cosmo_1['XCO2_BG'][0,:] + cosmo_1['XCO2_ALL'][0,:] +
                cosmo_1['XCO2_RA'][0,:] - cosmo_1['XCO2_GPP'][0,:]
        )

        
        cosmo_xlocs = cosmo_1["rlon"][:]
        cosmo_ylocs= cosmo_1["rlat"][:]

        ax = plt.axes(projection=transform)

        # plot borders
        ax.coastlines(resolution="110m")
        ax.add_feature(cartopy.feature.BORDERS)


        vmin=400 #4*pow(10,-4)
        vmax=410 #4.6*pow(10,-4)

        log=False
        if log:
            to_plot_mask = np.ma.masked_where(to_plot<=0, to_plot)
            mesh = ax.pcolormesh(cosmo_xlocs,cosmo_ylocs,to_plot_mask,norm=colors.LogNorm())#,vmin=vmin,vmax=vmax)
            plt.colorbar(mesh,ticks=[(6+0.1*i)*pow(10,-4) for i in range(6)])
        else:   
            #to_plot_mask = np.ma.masked_where(np.logical_not(np.isfinite(to_plot)), to_plot)
            mesh = ax.pcolormesh(cosmo_xlocs,cosmo_ylocs,to_plot,vmin=vmin,vmax=vmax)#, norm = norm)# 
            plt.colorbar(mesh)#,ticks=[(6+0.1*i)*pow(10,-4) for i in range(6)])#,norm=norm,boundaries = bounds)

        ax.set_extent([-17,21,-11,19.5],crs=transform)
        plt.tight_layout()
        plt.title("CO2 concentrations (in ppm) on %s" %date_disp)

        # corners= ccrs.PlateCarree().transform_points(transform,np.array([-17,-17,21,21]),np.array([-11,19.5,-11,19.5]))
        # ax.set_extent([min(corners[:,0]),max(corners[:,0]),min(corners[:,1]),max(corners[:,1])])

        # plt.savefig("COSMO_"+date_str+".png")
        # plt.clf()
        plt.show()

   
