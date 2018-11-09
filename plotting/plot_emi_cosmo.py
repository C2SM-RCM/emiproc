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



# cosmo_2 = nc.Dataset("../testdata/emis_2015_Berlin-coarse_64_74.nc")
# cosmo_1 = nc.Dataset("../testdata/emis_2015_Berlin-coarse_60_70.nc")
#cosmo_2 = nc.Dataset("../hourly_emissions/output/CO2_CO_NOX_Berlin-coarse_2015010100.nc")
# cosmo_1 = nc.Dataset("../testdata/emi_int2lm_2d.nc")
# cosmo_2 = nc.Dataset("../testdata/emi_2d.nc")
#cosmo_1 = nc.Dataset("../testdata/oae_outputs/test_1.nc")
#cosmo_2 = nc.Dataset("../testdata/oae_outputs/test_2.nc")
cosmo_1 = nc.Dataset("../testdata/CHE_TNO/emis_2015_Europe.nc")


var = "CO2_G_AREA"
#var = "CO2_"
co2=(cosmo_1[var][:])#-cosmo_2[var][0,-1,:])#/np.ma.masked_where(cosmo_1[var][0,-1,:]<pow(10,-12),cosmo_1[var][0,-1,:])
#co2 = cosmo_2[var][0,-1,:]-cosmo_1[var][0,-1,:]

to_plot = abs(co2 )
#to_plot = cosmo_1["CO2_02_AREA"][:]


# If i want discrete values
bounds = [ 0.,0.001,  0.01,  0.1,  0.3,  0.5, 1. ]
cmap   = matplotlib.colors.ListedColormap( [ 'b', 'g', 'y', 'r' ] )
norm   = matplotlib.colors.BoundaryNorm( bounds, ncolors=256 )

plot_map=True
plot_hist=False
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
    cosmo_xlocs = cosmo_1["rlon"]#np.arange(cosmo_lon,cosmo_lon+cosmo_dx*cosmo_nx,cosmo_dx)
    cosmo_ylocs= cosmo_1["rlat"]#np.arange(cosmo_lat,cosmo_lat+cosmo_dy*cosmo_ny,cosmo_dy)

    #ax.gridlines(crs= ccrs.PlateCarree(), xlocs=cosmo_xlocs,ylocs = cosmo_ylocs, color="r")
    # Add half a cell to be centered
    # cosmo_xlocs += cosmo_dx/2
    # cosmo_ylocs += cosmo_dy/2

    log=True
    if log:
        to_plot_mask = np.ma.masked_where(to_plot<=0, to_plot)
        mesh = ax.pcolor(cosmo_xlocs,cosmo_ylocs,to_plot_mask,norm=colors.LogNorm(),vmin=pow(10,-15))#vmin=pow(10,-7),vmax=5*pow(10,-5))
        plt.colorbar(mesh,ticks=[pow(10,i) for i in range(-15,-3)])
    else:   
        #to_plot_mask = np.ma.masked_where(np.logical_not(np.isfinite(to_plot)), to_plot)
        mesh = ax.pcolor(cosmo_xlocs,cosmo_ylocs,to_plot)#, norm = norm)# ,vmin=0,vmax=0.8)
        plt.colorbar(mesh)#,norm=norm,boundaries = bounds)

    # plt.savefig("emi.png")
    # plt.clf()
    plt.show()

   
