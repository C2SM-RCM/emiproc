# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 13:09:49 2018

@author: hjm
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
import netCDF4 as nc
#from matplotlib.patches import Polygon
from shapely.geometry import Polygon

# calculate 2D array of the areas (m^^2) of the COSMO grid
def gridcell_area(x,y,dx,dy): #bottom left corner of the cell
    radius=6375000. #the earth radius in meters
    deg2rad=np.pi/180.
    dlat = dy*deg2rad
    dlon = dx*deg2rad

    # box area at equator
    dd=2.*pow(radius,2)*dlon*np.sin(0.5*dlat)
    area = dd*np.cos(deg2rad*y+dlat)
    return area




##################
## The issue that script script is checking is the following:
##  
##   If I divide a tno cell in two halves, they have the same area in degree^2.
## -----
## ¦   ¦
## -----
## ¦   ¦
## -----
##   But in reality, in high latitude, they would look like this instead:
##   -
##  / \
##  ---
## /   \
## -----
##   which shows they don't have the same area at all.
##  What is the impact of calculating the area ratio of an intersection in degree^2 ?

## Other way to look at it:
##   ________
##   |  |    |
##   |  |____|
##   |  |    |
##   |__|____|
##
## In that case, they all have the same area (2*4 cells) in degree^2.
## But actually, at high latitude, 2*4 cells (left) is different than 4*2 (right)
## Because 1deg lon is much bigger than 1deg latitude


if True:
    ################
    ##  TNO GRID  ##
    ################
    molly = ccrs.Mollweide()
    # Lets say these are the bottom corner
    tno_lon = -30
    tno_dx  = 1/8.
    tno_nx  = 90*8
    tno_lat = 28
    tno_dy  = 1./16
    tno_ny = 42*16
    
    x_tno=15
    y_tno=80
    tno_cell_x= np.array([x_tno+tno_dx/2,x_tno+tno_dx/2,x_tno-tno_dx/2,x_tno-tno_dx/2])
    tno_cell_y= np.array([y_tno+tno_dy/2,y_tno-tno_dy/2,y_tno-tno_dy/2,y_tno+tno_dy/2])
    

    # cosmo cell 1, the left cell
    cosmo_cell_1_x = np.array([x_tno-tno_dx/6,x_tno-tno_dx/6,x_tno-tno_dx/2,x_tno-tno_dx/2])
    cosmo_cell_1_y = np.array([y_tno+tno_dy/2,y_tno-tno_dy/2,y_tno-tno_dy/2,y_tno+tno_dy/2])
    
    # cosmo cell 2, the bottom right cell
    cosmo_cell_2_x = np.array([x_tno+tno_dx/2,x_tno+tno_dx/2,x_tno-tno_dx/6,x_tno-tno_dx/6])
    cosmo_cell_2_y = np.array([y_tno,y_tno-tno_dy/2,y_tno-tno_dy/2,y_tno])

    # cosmo cell 3, the top right cell
    cosmo_cell_3_x = np.array([x_tno+tno_dx/2,x_tno+tno_dx/2,x_tno-tno_dx/6,x_tno-tno_dx/6])
    cosmo_cell_3_y = np.array([y_tno,y_tno-tno_dy/2,y_tno-tno_dy/2,y_tno])+tno_dy/2

    
    # Make a polygon out of it, 
    polygon_tno = Polygon([i for i in zip(tno_cell_x,tno_cell_y)])
    tno_area = polygon_tno.area
    
    polygon_cosmo_1 = Polygon([i for i in zip(cosmo_cell_1_x,cosmo_cell_1_y)])
    inter_1 = polygon_cosmo_1.intersection(polygon_tno)
    area_1 = inter_1.area
    polygon_cosmo_2 = Polygon([i for i in zip(cosmo_cell_2_x,cosmo_cell_2_y)])
    inter_2 = polygon_cosmo_2.intersection(polygon_tno)
    area_2 = inter_2.area
    polygon_cosmo_3 = Polygon([i for i in zip(cosmo_cell_3_x,cosmo_cell_3_y)])
    inter_3 = polygon_cosmo_3.intersection(polygon_tno)
    area_3 = inter_3.area

    print("left area:", area_1/tno_area, "bottom right area:",area_2/tno_area, "top right area:",area_3/tno_area)

    # With Molly
    polygon_tno = Polygon(molly.transform_points(ccrs.PlateCarree(),tno_cell_x,tno_cell_y))
    tno_area = polygon_tno.area
    polygon_cosmo_1 = Polygon(molly.transform_points(ccrs.PlateCarree(),cosmo_cell_1_x,cosmo_cell_1_y))
    polygon_cosmo_2 = Polygon(molly.transform_points(ccrs.PlateCarree(),cosmo_cell_2_x,cosmo_cell_2_y))
    polygon_cosmo_3 = Polygon(molly.transform_points(ccrs.PlateCarree(),cosmo_cell_3_x,cosmo_cell_3_y))
    area_1 = polygon_cosmo_1.intersection(polygon_tno).area
    area_2 = polygon_cosmo_2.intersection(polygon_tno).area
    area_3 = polygon_cosmo_3.intersection(polygon_tno).area

    print("left area:", area_1/tno_area, "bottom right area:",area_2/tno_area, "top right area:",area_3/tno_area)

    # With area
    tno_area = gridcell_area(x_tno,y_tno,tno_dx,tno_dy)
    area_1 = gridcell_area(x_tno, y_tno, tno_dx/3, tno_dy)
    area_2 = gridcell_area(x_tno, y_tno-tno_dy/4, 2/3*tno_dx, tno_dy/2)
    area_3 = gridcell_area(x_tno, y_tno+tno_dy/4, 2/3*tno_dx, tno_dy/2)
    
    print("left area:", area_1/tno_area, "bottom right area:",area_2/tno_area, "top right area:",area_3/tno_area)
