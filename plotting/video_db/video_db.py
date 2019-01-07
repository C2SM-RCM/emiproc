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
from datetime import datetime

pole_lon = -170
pole_lat = 43

date = datetime(2015,1,1,10)
date_str = date.strftime("%Y%m%d%H")
date_disp = date.strftime("%Y-%m-%d %H:00")
cosmo_1 = nc.Dataset("./lffd2015010110.nc")

var = "CH4_ALL"
co2=(cosmo_1[var][0,-1,:,:])

to_plot = (co2)

# If i want discrete values
bounds = [ 0.,0.001,  0.01,  0.1,  0.3,  0.5, 1. ]
cmap   = matplotlib.colors.ListedColormap( [ 'b', 'g', 'y', 'r' ] )
norm   = matplotlib.colors.BoundaryNorm( bounds, ncolors=256 )

if True:
    transform = ccrs.RotatedPole(pole_longitude=pole_lon, pole_latitude=pole_lat)
    
    ax = plt.axes(projection=transform)
    
    # plot borders
    ax.coastlines(resolution="110m")
    ax.add_feature(cartopy.feature.BORDERS)
    ax.set_extent((17,20.5,49,51))
    # plot the cosmo grid
    cosmo_xlocs = cosmo_1["rlon"][:]#np.arange(cosmo_lon,cosmo_lon+cosmo_dx*cosmo_nx,cosmo_dx)
    cosmo_ylocs= cosmo_1["rlat"][:]#np.arange(cosmo_lat,cosmo_lat+cosmo_dy*cosmo_ny,cosmo_dy)

    #ax.gridlines(crs= ccrs.PlateCarree(), xlocs=cosmo_xlocs,ylocs = cosmo_ylocs, color="r")
    # Add half a cell to be centered
    # cosmo_xlocs += cosmo_dx/2
    # cosmo_ylocs += cosmo_dy/2

    to_plot_mask = np.ma.masked_where(to_plot<=0, to_plot)
    mesh = ax.pcolor(cosmo_xlocs,cosmo_ylocs,to_plot_mask,vmin=pow(10,-9), vmax=pow(10,-6), norm=colors.LogNorm()) #norm=colors.LogNorm()
    plt.colorbar(mesh,ticks=[pow(10,i) for i in range(-10,-5)])
    plt.title("CH4 concentrations (in kg/kg) on %s" %date_disp)
    

    # plt.savefig("10.png")
    # plt.clf()
    plt.show()

   
