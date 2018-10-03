# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 10:00:08 2018

@author: hjm
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import netCDF4 as nc

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

transform = ccrs.RotatedPole(pole_longitude=pole_lon, pole_latitude=pole_lat)

## TNO grid
# Lets say these are the bottom corner
tno_lon = -30
tno_dx  = 1/8.
tno_nx  = 90*8
tno_lat = 28
tno_dy  = 1./16
tno_ny = 42*16


ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(resolution="110m")
ax.add_feature(cartopy.feature.BORDERS)

tno_xlocs = np.arange(tno_lon,tno_lon+tno_dx*tno_nx,tno_dx)
tno_ylocs= np.arange(tno_lat,tno_lat+tno_dy*tno_ny,tno_dy)

#ax.gridlines(crs= ccrs.PlateCarree(), xlocs=tno_xlocs,ylocs = tno_ylocs, color="r")

# plot the cosmo grid
cosmo_xlocs = np.arange(cosmo_lon,cosmo_lon+cosmo_dx*cosmo_nx,cosmo_dx)
cosmo_ylocs = np.arange(cosmo_lat,cosmo_lat+cosmo_dy*cosmo_ny,cosmo_dy)[:-1]
#ax.gridlines(crs= transform,xlocs=cosmo_xlocs,ylocs=cosmo_ylocs)




mapping = np.load("mapping.npy")

map_truth = np.array([[len(mapping[i,j]) for j in range(672)] for i in range(720)])

mesh = ax.pcolormesh(tno_xlocs,tno_ylocs,map_truth.T)#,vmax=1,vmin=0.01)#_mask)
plt.colorbar(mesh)




