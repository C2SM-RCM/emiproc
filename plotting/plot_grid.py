# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
from shapely.geometry import Polygon
import netCDF4 as nc
#from matplotlib.patches import Polygon

#tno_startlon= -10
#tno_dlon = 1./16
#tno_nx = 16*20+1
#
#tno_startlon= 35
#tno_dlon = 1./8
#tno_nx = 8*30+1

def plot_line(ax, ob):
    x, y = ob.xy
    ax.plot(x, y, color="k",alpha=0.7, solid_capstyle='round', zorder=2)

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

################
##  TNO GRID  ##
################
# Lets say these are the bottom corner
tno_lon = -30
tno_dx  = 1/8.
tno_nx  = 90*8
tno_lat = 30
tno_dy  = 1./16
tno_ny = 42*16

#ax = plt.axes(projection=ccrs.PlateCarree())
ax = plt.axes(projection=transform)

# plot borders
ax.coastlines(resolution="10m")
ax.add_feature(cartopy.feature.BORDERS)
# plot borders with the shapefile from Dominik
#shpfilename = "C:\\Users\\hjm\\Desktop\\shapefile\\cntry08.shp"
#reader = shpreader.Reader(shpfilename)
#shapes  = reader.geometries()
#for g in reader.geometries():    
#    for l in g.boundary:
#        plot_line(ax,l)    


#ax.set_extent([-10, 10, 35, 65], ccrs.PlateCarree())
#ax.set_extent([-15, 15, -15, 15], transform)



# plot the cosmo grid
cosmo_xlocs = np.arange(cosmo_lon,cosmo_lon+cosmo_dx*cosmo_nx,cosmo_dx)
cosmo_ylocs = np.arange(cosmo_lat,cosmo_lat+cosmo_dy*cosmo_ny,cosmo_dy)[:-1]
#ax.gridlines(crs= transform,xlocs=cosmo_xlocs,ylocs=cosmo_ylocs)

# plot the tno grid
tno_xlocs = np.arange(tno_lon,tno_lon+tno_dx*tno_nx,tno_dx)
tno_ylocs= np.arange(tno_lat,tno_lat+tno_dy*tno_ny,tno_dy)
#ax.gridlines(xlocs=tno_xlocs,ylocs = tno_ylocs, color="r")

#Knowing the grid, get the middle of cells
#This is what I would get from TNO
middle_cells =[]  
for x in tno_xlocs:
    for y in tno_ylocs:
        middle_cells.append((x+tno_dx/2,y+tno_dy/2))
        
        
        
# plot the country mask         
#me = nc.Dataset("./emis_2015_Berlin-coarse.nc")
##me = nc.Dataset("./emis_2015_brd.nc")
#mask = me["country_ids"][:]
#adapt = {0:"c",6:"k",19:"g",23:"b",46:"r",47:"y",60:"w",61:"m"}
#for (a,x) in enumerate(cosmo_xlocs):
#    for (b,y) in enumerate(cosmo_ylocs):
#        cosmo_cell_x = [x+cosmo_dx,x+cosmo_dx,x,x]
#        cosmo_cell_y = [y+cosmo_dy,y,y,y+cosmo_dy]
#            
#        ax.fill(cosmo_cell_x,cosmo_cell_y,color=adapt[mask[b,a]])
##        points = ccrs.PlateCarree().transform_points(transform,np.array(cosmo_cell_x),np.array(cosmo_cell_y))
##        ax.fill(points[:,0],points[:,1],color=adapt[mask[b,a]])
#        
        
        
for (x_tno,y_tno) in middle_cells:
    #middle_cell_tno_proj = transform.transform_point(x,y,ccrs.PlateCarree())
    #ax.scatter(middle_cell_tno_proj[0],middle_cell_tno_proj[1],color="b",zorder=100)

    # Get the corners of the tno cell
    tno_cell_x= np.array([x_tno+tno_dx/2,x_tno+tno_dx/2,x_tno-tno_dx/2,x_tno-tno_dx/2])
    tno_cell_y= np.array([y_tno+tno_dy/2,y_tno-tno_dy/2,y_tno-tno_dy/2,y_tno+tno_dy/2])
    
    # Make a polygon out of it, in cosmo grid.
    points = transform.transform_points(ccrs.PlateCarree(),tno_cell_x,tno_cell_y)
    polygon_tno = Polygon(points)
    
    if x_tno==-29.9375 and y_tno==36.59375:
        print(polygon_tno.area) 
    polygon_tno.area
    ax.fill(points[:,0],points[:,1])
    
    incr = 0
    for x in cosmo_xlocs:
        for y in cosmo_ylocs:
            # Get the corners of the cosmo cell
            cosmo_cell_x = [x+cosmo_dx,x+cosmo_dx,x,x]
            cosmo_cell_y = [y+cosmo_dy,y,y,y+cosmo_dy]
            polygon_cosmo = Polygon([i for i in zip(cosmo_cell_x,cosmo_cell_y)])
            
            if polygon_cosmo.intersects(polygon_tno):
                incr+=1
                print(incr)
                ax.fill(cosmo_cell_x,cosmo_cell_y)#,color="c")
        
                inter = polygon_cosmo.intersection(polygon_tno)
                inter.area
                #print(inter.area)                
                corners = inter.exterior.coords.xy    
                ax.fill(corners[0],corners[1])#,color="m")
        
    
        


