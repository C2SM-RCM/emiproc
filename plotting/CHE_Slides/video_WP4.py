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
from multiprocessing import Pool


pole_lon = -170
pole_lat = 43

transform = ccrs.RotatedPole(pole_longitude=pole_lon, pole_latitude=pole_lat)

path = "/project/s862/CHE/CHE_output_todel/CHE_Europe_output/"


def plot_2D_field(to_plot,proj,lon,lat,log=False,**kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection=proj)
    
    # plot borders
    ax.coastlines(resolution="10m")
    lines = cartopy.feature.NaturalEarthFeature(category='cultural', scale='10m',                     
                                         name='admin_0_boundary_lines_land')
    ax.add_feature(lines, edgecolor='k', facecolor='none')

    if log:
        to_plot_mask = np.ma.masked_where(to_plot<=0, to_plot)
        mesh = ax.pcolor(lon,lat,to_plot_mask, norm=colors.LogNorm(),**kwargs)#vmin=pow(10,-9), vmax=pow(10,-4)
        fig.colorbar(mesh,ticks=[pow(10,i) for i in range(-10,-3)])
    else:
        mesh = ax.pcolor(lon,lat,to_plot,**kwargs)
        fig.colorbar(mesh,ticks=[pow(10,i) for i in range(-10,-3)])
    return fig

def plot_from_file(folder,date):
    date_str = date.strftime("%Y%m%d%H")
    date_disp = date.strftime("%Y-%m-%d %H:00")

    cosmo_1 = nc.Dataset(folder+"lffd"+date_str+".nc")

    for var in ["CO2_ALL","CO_ALL"]:
        co2=(cosmo_1[var][0,-1,:,:])

        lon ="rlon"
        lat="rlat"
        cosmo_xlocs = cosmo_1[lon][:]
        cosmo_ylocs= cosmo_1[lat][:]

        fig = plot_2D_field(co2,transform,cosmo_xlocs,cosmo_ylocs,log=True,vmin=pow(10,-9), vmax=pow(10,-4-2*(var=="CO_ALL")))
        s = var.split("_")[0]
        plt.title(s+" concentrations (in kg/kg) on %s" %date_disp)

        fig.savefig(s+"_"+date_str+".png")
        plt.close(fig)
        
        # plt.show()



for i in range(1,2):#10):
    if i==1:
        folder = path+"2015010100_0_24/cosmo_output/"
    else:
        folder = path+"2015010"+str(i-1)+"18_0_30/cosmo_output/"

    with Pool(16) as pool: 
        pool.starmap(plot_from_file, [(folder,datetime(2015,1,i,j)) for j in range(5)])

    
    
        #ax.gridlines(xlocs=np.arange(17,21,0.5), ylocs=np.arange(49,51.5,0.5),crs=ccrs.PlateCarree())
        # ax.gridlines()
        #xticks = ccrs.transform
        # ax.set_xticks(np.arange(17,21,0.5),crs=ccrs.PlateCarree())
        #ax.set_yticks(np.arange(49,51,0.5),crs=ccrs.PlateCarree())

        # ax.xaxis.set_major_formatter(LongitudeFormatter())
        #ax.yaxis.set_major_formatter(LatitudeFormatter())



        # # plot major city locations
        # shp_fn = shpreader.natural_earth(resolution='10m', category='cultural', 
        #                                  name='populated_places')
        # shp = shpreader.Reader(shp_fn)
        # xy = [pt.coords[0] for pt in shp.geometries()]
        # x, y = zip(*xy)
        # points = transform.transform_points(ccrs.PlateCarree(),np.array(x),np.array(y))
        # x=points[:,0]
        # y=points[:,1]
        # ax.scatter(x,y,25,marker='o',color="c",edgecolors='black',zorder=100)
        
        # katto = transform.transform_point(19.1,50.3,ccrs.PlateCarree())
        # ax.text(katto[0],katto[1],"Kattowitz")
        # krak = transform.transform_point(20.1,50.1,ccrs.PlateCarree())
        # ax.text(krak[0],krak[1],"Krakow")


   
