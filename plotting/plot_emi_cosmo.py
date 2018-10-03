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

# calculate 2D array of the areas (m^^2) of the COSMO grid
def gridbox_area(dx,dy,ymin,nx,ny):
    radius=6375000. #the earth radius in meters
    deg2rad=np.pi/180.
    dlat = dy*deg2rad
    dlon = dx*deg2rad

    # box area at equator
    dd=2.*pow(radius,2)*dlon*np.sin(0.5*dlat)
    areas = np.array([[dd*np.cos(deg2rad*ymin+j*dlat) for j in range(ny)] for foo in range(nx)])
    return areas 


##################
##  COSMO GRID  ##
##################
# Lets say these are the bottom corner
cosmo_lon= -1.6
cosmo_lat = 2.3
cosmo_dx = 0.1
cosmo_dy = 0.1
cosmo_nx = 74
cosmo_ny = 64
pole_lon = -170
pole_lat = 43



# Variables to convert unit
sec_per_year = 365.25*24*3600
cosmo_areas = gridbox_area(cosmo_dx,cosmo_dy,cosmo_lat,cosmo_nx,cosmo_ny)



cosmo_2 = nc.Dataset("../emis_2015_brd.nc")
cosmo_1 = nc.Dataset("../emis_2015_Berlin-coarse.nc")

co2= (cosmo_1["CO2_02_AREA"][:]-cosmo_2["CO2_02_AREA"][:])/np.ma.masked_where(cosmo_1["CO2_02_AREA"][:]<pow(10,-12),cosmo_1["CO2_02_AREA"][:])

to_plot = abs(co2 )
#to_plot = cosmo_1["CO2_02_AREA"][:]



# If i want discrete values
bounds = [ 0.,  0.01,  0.1,  0.3,  0.5 ]
cmap   = matplotlib.colors.ListedColormap( [ 'r', 'y', 'g', 'b' ] )
norm   = matplotlib.colors.BoundaryNorm( bounds, cmap.N )

plot_map=False
plot_hist=True
if plot_hist:
    bins=np.arange(0,0.5,0.01)
    ax = plt.axes()
    ax.hist(to_plot.flatten(), cumulative=True,bins=bins, facecolor='g', alpha=0.75)
    ax.set_xticks(np.arange(0,0.5,0.05))
    ax.set_xticklabels([str(i)+"%" for i in  np.arange(0,51,5)])
    ax.set_yticks(np.arange(0,101,10)/100 *64*74)
    ax.set_yticklabels([str(i)+"%" for i in  np.arange(0,101,10)])
    ax.grid()
    ax.set_ylabel("Amount of grid cells")
    ax.set_xlabel("Percentage of difference between me and brd")


if plot_map:
    transform = ccrs.RotatedPole(pole_longitude=pole_lon, pole_latitude=pole_lat)
    
    ax = plt.axes(projection=transform)
    
    # plot borders
    ax.coastlines(resolution="110m")
    ax.add_feature(cartopy.feature.BORDERS)
    
    # plot the cosmo grid
    cosmo_xlocs = np.arange(cosmo_lon,cosmo_lon+cosmo_dx*cosmo_nx,cosmo_dx)
    cosmo_ylocs= np.arange(cosmo_lat,cosmo_lat+cosmo_dy*cosmo_ny,cosmo_dy)
    #ax.gridlines(crs= ccrs.PlateCarree(), xlocs=cosmo_xlocs,ylocs = cosmo_ylocs, color="r")
    # Add half a cell to be centered
    cosmo_xlocs += cosmo_dx/2
    cosmo_ylocs += cosmo_dy/2
    log=False
    if log:
        to_plot_mask = np.ma.masked_where(to_plot<=0, to_plot)
        mesh = ax.pcolormesh(cosmo_xlocs,cosmo_ylocs,to_plot_mask,norm=colors.LogNorm())#,vmax=1,vmin=0.01)
        #plt.colorbar(mesh,ticks=[pow(10,i) for i in range(-15,-5)])
    else:   
        #to_plot_mask = np.ma.masked_where(np.logical_not(np.isfinite(to_plot)), to_plot)
        mesh = ax.pcolormesh(cosmo_xlocs,cosmo_ylocs,to_plot, cmap = cmap, norm = norm )#,vmax=0.5)#_mask)
        plt.colorbar(mesh)

   
