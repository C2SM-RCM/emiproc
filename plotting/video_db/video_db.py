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
import itertools
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.io.shapereader as shpreader

pole_lon = -170
pole_lat = 43

var = "CH4_ALL"
path = "/project/s862/CHE/CHE_output_todel/CHE_Europe_output/"

transform = ccrs.RotatedPole(pole_longitude=pole_lon, pole_latitude=pole_lat)

for i,j in itertools.product(range(1,10),range(24)):
    if i==1:
        folder = path+"2015010100_0_24/cosmo_output/"
    else:
        folder = path+"2015010"+str(i-1)+"18_0_30/cosmo_output/"
        
    date = datetime(2015,1,i,j)
    date_str = date.strftime("%Y%m%d%H")
    date_disp = date.strftime("%Y-%m-%d %H:00")
    cosmo_1 = nc.Dataset(folder+"lffd"+date_str+".nc")


    co2=(cosmo_1[var][0,-1,:,:])

    to_plot = (co2)
    
    ax = plt.axes(projection=transform)
    
    # plot borders
    ax.coastlines(resolution="10m")
    lines = cartopy.feature.NaturalEarthFeature(category='cultural', scale='10m',                     
                                         name='admin_0_boundary_lines_land')
    ax.add_feature(lines, edgecolor='k', facecolor='none')


    cosmo_xlocs = cosmo_1["rlon"][:]
    cosmo_ylocs= cosmo_1["rlat"][:]

    to_plot_mask = np.ma.masked_where(to_plot<=0, to_plot)
    mesh = ax.pcolor(cosmo_xlocs,cosmo_ylocs,to_plot_mask,vmin=pow(10,-9), vmax=pow(10,-6), norm=colors.LogNorm())
    plt.colorbar(mesh,ticks=[pow(10,i) for i in range(-10,-5)])
    plt.title("CH4 concentrations (in kg/kg) on %s" %date_disp)

    ax.set_extent([17,20.5,49,51],crs=ccrs.PlateCarree())

    #ax.gridlines(xlocs=np.arange(17,21,0.5), ylocs=np.arange(49,51.5,0.5),crs=ccrs.PlateCarree())
    # ax.gridlines()
    #xticks = ccrs.transform
    # ax.set_xticks(np.arange(17,21,0.5),crs=ccrs.PlateCarree())
    #ax.set_yticks(np.arange(49,51,0.5),crs=ccrs.PlateCarree())

    # ax.xaxis.set_major_formatter(LongitudeFormatter())
    #ax.yaxis.set_major_formatter(LatitudeFormatter())



    # plot major city locations
    shp_fn = shpreader.natural_earth(resolution='10m', category='cultural', 
                                     name='populated_places')
    shp = shpreader.Reader(shp_fn)
    xy = [pt.coords[0] for pt in shp.geometries()]
    x, y = zip(*xy)
    points = transform.transform_points(ccrs.PlateCarree(),np.array(x),np.array(y))
    x=points[:,0]
    y=points[:,1]
    ax.scatter(x,y,25,marker='o',color="c",edgecolors='black',zorder=100)

    katto = transform.transform_point(19.1,50.3,ccrs.PlateCarree())
    ax.text(katto[0],katto[1],"Kattowitz")
    krak = transform.transform_point(20.1,50.1,ccrs.PlateCarree())
    ax.text(krak[0],krak[1],"Krakow")

    plt.savefig(date_str+".png")
    plt.clf()
    # plt.show()

   
