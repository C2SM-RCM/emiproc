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



##  TNO GRID  ##
################
# Lets say these are the bottom corner
tno_lon = -30
tno_dx  = 1/8.
tno_nx  = 90*8
tno_lat = 30
tno_dy  = 1./16
tno_ny = 42*16

# Variables to convert unit
sec_per_year = 365.25*24*3600
tno_areas = gridbox_area(tno_dx,tno_dy,tno_lat,tno_nx,tno_ny)

ax = plt.axes(projection=ccrs.PlateCarree())#projection=transform)

# plot borders
ax.coastlines(resolution="110m")
ax.add_feature(cartopy.feature.BORDERS)
ax.set_extent((7.15,20.8,49,55.6))

# plot the tno grid
tno_xlocs = np.arange(tno_lon,tno_lon+tno_dx*tno_nx,tno_dx)
tno_ylocs= np.arange(tno_lat,tno_lat+tno_dy*tno_ny,tno_dy)
#ax.gridlines(crs= ccrs.PlateCarree(), xlocs=tno_xlocs,ylocs = tno_ylocs, color="r")
# Add half a cell to be centered
#tno_xlocs += tno_dx
#tno_ylocs += tno_dy

tno = nc.Dataset("./TNO_CAMS_CO2_emissions_2011.nc")

co2= tno["co2_bf"][:]+tno["co2_ff"][:]


selection_area  = tno["source_type_index"][:]==1
snap_list=[2]
tno_snap = tno["emis_cat_shortsnap"][:].tolist() #[1,2,34,5,6,71,72,73,74,75,8,9,10]
selection_snap = np.array([tno["emission_category_index"][:] == tno_snap.index(i)+1 for i in snap_list])
selection_snap_area  = np.array([selection_snap.any(0),selection_area]).all(0)
to_plot = np.zeros((tno_nx,tno_ny))
for (i,c) in enumerate(co2):
#    if i/1000 == int(i/1000):
#        print(i)    
    if selection_snap_area[i]:
        lat = tno["latitude_index"][i]-1
        lon = tno["longitude_index"][i]-1
        to_plot[lon,lat]+= c/(tno_areas[lon,lat]*sec_per_year)
to_plot_mask = np.ma.masked_where(to_plot==0, to_plot)
mesh = ax.pcolormesh(tno_xlocs,tno_ylocs,to_plot_mask.T,norm=colors.LogNorm())
plt.colorbar(mesh,ticks=[pow(10,i) for i in range(-15,-5)])




    