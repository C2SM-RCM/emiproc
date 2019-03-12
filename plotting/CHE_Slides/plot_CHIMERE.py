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

path = "/project/s862/CHE/CHIMERE_simulation_1w/"

var = "CO2"



bounds = [ 0.,0.001,  0.01,  0.1,  0.3,  0.5, 1. ]
norm   = matplotlib.colors.BoundaryNorm( bounds, ncolors=256 )

transform = ccrs.RotatedPole(pole_longitude=pole_lon, pole_latitude=pole_lat)

for i in range(1,10):
    folder = path
    date = datetime(2015,1,i)
    date_str = date.strftime("%Y%m%d")
    date_disp = date.strftime("%Y-%m-%d %H:00")

    cosmo_1 = nc.Dataset(folder+"out."+date_str+"00.24.nc")


    for j in range(0,24,1):
        date = datetime(2015,1,i,j)
    
        date_str = date.strftime("%Y%m%d%H")
    
        # co2_all= (cosmo_1[var][0,-1,:])
        # co2_bg = (cosmo_1["CO2_BG"][0,-1,:])
        # co2 = co2_bg+co2_all
        
        co2 = (
            cosmo_1['CO2'][:]
        )



        to_plot = co2[j,0,:]*pow(10,-3)#*convert_unit[var]



            # Transform the lon,lat of the ECMWF grid into rotated-pole coordinates
        all_points = np.array([(x,y) for (x,y) in zip(cosmo_1["lon"][:].flatten(), cosmo_1["lat"][:].flatten())])  #[(x,y) for x in cosmo_1["lon"][:].flatten() for y in cosmo_1["lat"][:].flatten()])
        grid_points= transform.transform_points(ccrs.PlateCarree(),all_points[:,0],all_points[:,1])
        cosmo_xlocs = grid_points[:,0]
        cosmo_xlocs.shape = cosmo_1["lon"][:].shape
        cosmo_ylocs = grid_points[:,1]
        cosmo_ylocs.shape = cosmo_1["lat"][:].shape

        ax = plt.axes(projection=transform)

        # plot borders
        ax.coastlines(resolution="110m")
        ax.add_feature(cartopy.feature.BORDERS)


        vmin=400 #4*pow(10,-4)
        vmax=460 #4.6*pow(10,-4)

        log=False
        if log:
            to_plot_mask = np.ma.masked_where(to_plot<=0, to_plot)
            mesh = ax.pcolormesh(cosmo_xlocs,cosmo_ylocs,to_plot_mask,norm=colors.LogNorm(),vmin=vmin,vmax=vmax)
            plt.colorbar(mesh,ticks=[(6+0.1*i)*pow(10,-4) for i in range(6)])
        else:   
            #to_plot_mask = np.ma.masked_where(np.logical_not(np.isfinite(to_plot)), to_plot)
            mesh = ax.pcolormesh(cosmo_xlocs,cosmo_ylocs,to_plot,vmin=vmin,vmax=vmax)#, norm = norm)# ,vmin=0,vmax=0.8)
            plt.colorbar(mesh)#,ticks=[(6+0.1*i)*pow(10,-4) for i in range(6)])#,norm=norm,boundaries = bounds)

        ax.set_extent([-17,21,-11,19.5],crs=transform)
        plt.tight_layout()
        plt.title("CO2 concentrations (in ppm) on %s" %date_disp)

        # corners= ccrs.PlateCarree().transform_points(transform,np.array([-17,-17,21,21]),np.array([-11,19.5,-11,19.5]))
        # ax.set_extent([min(corners[:,0]),max(corners[:,0]),min(corners[:,1]),max(corners[:,1])])

        plt.savefig("Figures/CHIMERE/CHIMERE_"+date_str+".png")
        plt.clf()
        # plt.show()

   
