# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 17:01:33 2018

@author: hjm
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 15:28:01 2018

@author: hjm
"""


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

##################
##  COSMO GRID  ##
##################
# cosmo_dx = 0.1
# cosmo_dy = 0.1
# cosmo_lon= -1.4#-2*cosmo_dx
# cosmo_lat = 2.5#-2*cosmo_dy
# cosmo_nx = 70#+4
# cosmo_ny = 60#+4
# ############################
# cosmo_dx = 0.05
# cosmo_dy = 0.05
# cosmo_lon= -17#-2*cosmo_dx
# cosmo_lat = -11#-2*cosmo_dy
# cosmo_nx = 760#+4
# cosmo_ny = 610#+4

pole_lon = -170
pole_lat = 43



cosmo_1 = nc.Dataset("/store/empa/em05/dbrunner/che/icbc/cams_gvri_2015010612.nc")


var = "co2"
#var = "CO2_"
co2=(cosmo_1[var][0,-1,:])

to_plot = abs(co2.data)

# If i want discrete values
bounds = [ 0.,0.001,  0.01,  0.1,  0.3,  0.5, 1. ]
cmap   = matplotlib.colors.ListedColormap( [ 'b', 'g', 'y', 'r' ] )
norm   = matplotlib.colors.BoundaryNorm( bounds, ncolors=256 )

plot_map=True
if plot_map:
    transform = ccrs.RotatedPole(pole_longitude=pole_lon, pole_latitude=pole_lat)

    # Transform the lon,lat of the ECMWF grid into rotated-pole coordinates
    all_points = np.array([(x,y) for x in cosmo_1["longitude"][:] for y in cosmo_1["latitude"][:]])
    grid_points= transform.transform_points(ccrs.PlateCarree(),all_points[:,0],all_points[:,1])
    cosmo_xlocs = grid_points[:,0]
    cosmo_xlocs.shape = (len(cosmo_1["longitude"][:]),len(cosmo_1["latitude"][:]))
    cosmo_ylocs = grid_points[:,1]
    cosmo_ylocs.shape = (len(cosmo_1["longitude"][:]),len(cosmo_1["latitude"][:]))

    ax = plt.axes(projection=transform)
    
    # plot borders
    ax.coastlines(resolution="110m")
    ax.add_feature(cartopy.feature.BORDERS)
    
    log=True
    if log:
        to_plot_mask = np.ma.masked_where(to_plot<=0, to_plot)
        mesh = ax.pcolormesh(cosmo_xlocs,cosmo_ylocs,to_plot_mask.T,norm=colors.LogNorm(),vmin=6*pow(10,-4),vmax=6.5*pow(10,-4))
        plt.colorbar(mesh)#,ticks=[pow(10,i) for i in range(-4,-6)])
    else:   
        #to_plot_mask = np.ma.masked_where(np.logical_not(np.isfinite(to_plot)), to_plot)
        mesh = ax.pcolormesh(cosmo_xlocs,cosmo_ylocs,to_plot.T)#, norm = norm)# ,vmin=0,vmax=0.8)
        plt.colorbar(mesh)#,norm=norm,boundaries = bounds)

    ax.set_extent([-17,21,-11,19.5],crs=transform)
    # corners= ccrs.PlateCarree().transform_points(transform,np.array([-17,-17,21,21]),np.array([-11,19.5,-11,19.5]))
    # ax.set_extent([min(corners[:,0]),max(corners[:,0]),min(corners[:,1]),max(corners[:,1])])

    plt.savefig("cams.png")
    plt.clf()
    #plt.show()

   
