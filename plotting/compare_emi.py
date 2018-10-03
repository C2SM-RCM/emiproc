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




me = nc.Dataset("./emis_2015_Berlin-coarse.nc")
brd = nc.Dataset("./emis_2015_brd.nc")

tc_1 =  me["CO2_02_AREA"][:]
tc_2 = brd["CO2_02_AREA"][:]



if False:
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
    ################
    ##  TNO GRID  ##
    ################
    transform = ccrs.RotatedPole(pole_longitude=pole_lon, pole_latitude=pole_lat)
    molly = ccrs.Mollweide()
    # Lets say these are the bottom corner
    tno_lon = -30
    tno_dx  = 1/8.
    tno_nx  = 90*8
    tno_lat = 28
    tno_dy  = 1./16
    tno_ny = 42*16
    
    x_tno=15
    y_tno=60
    tno_cell_x= np.array([x_tno+tno_dx/2,x_tno+tno_dx/2,x_tno-tno_dx/2,x_tno-tno_dx/2])
    tno_cell_y= np.array([y_tno+tno_dy/2,y_tno-tno_dy/2,y_tno-tno_dy/2,y_tno+tno_dy/2])
    
    # cosmo cell 1, touches the bottom of the tno_cell
    cosmo_cell_1_x = np.array([x_tno+tno_dx/2,x_tno+tno_dx/2,x_tno-tno_dx/2,x_tno-tno_dx/2])
    cosmo_cell_1_y = np.array([y_tno+tno_dy/2,y_tno-tno_dy/2,y_tno-tno_dy/2,y_tno+tno_dy/2])-1./32.
    
    # cosmo cell 1, touches the bottom of the tno_cell
    cosmo_cell_2_x = np.array([x_tno+tno_dx/2,x_tno+tno_dx/2,x_tno-tno_dx/2,x_tno-tno_dx/2])
    cosmo_cell_2_y = np.array([y_tno+tno_dy/2,y_tno-tno_dy/2,y_tno-tno_dy/2,y_tno+tno_dy/2])+1./32.
    
    # Make a polygon out of it, 
    polygon_tno = Polygon([i for i in zip(tno_cell_x,tno_cell_y)])
    tno_area = polygon_tno.area
    
    polygon_cosmo_1 = Polygon([i for i in zip(cosmo_cell_1_x,cosmo_cell_1_y)])
    inter_1 = polygon_cosmo_1.intersection(polygon_tno)
    area_1 = inter_1.area
    polygon_cosmo_2 = Polygon([i for i in zip(cosmo_cell_2_x,cosmo_cell_2_y)])
    inter_2 = polygon_cosmo_2.intersection(polygon_tno)
    area_2 = inter_2.area
    
    print("bottom area:", area_1/tno_area,"top area:",area_2/tno_area)
    
    # With Molly
    polygon_tno = Polygon(molly.transform_points(ccrs.PlateCarree(),tno_cell_x,tno_cell_y))
    tno_area = polygon_tno.area
    polygon_cosmo_1 = Polygon(molly.transform_points(ccrs.PlateCarree(),cosmo_cell_1_x,cosmo_cell_1_y))
    polygon_cosmo_2 = Polygon(molly.transform_points(ccrs.PlateCarree(),cosmo_cell_2_x,cosmo_cell_2_y))
    area_1 = polygon_cosmo_1.intersection(polygon_tno).area
    area_2 = polygon_cosmo_2.intersection(polygon_tno).area
    print("bottom area:", area_1/tno_area,"top area:",area_2/tno_area)
    
    # With area
    tno_area = gridcell_area(x_tno,y_tno,tno_dx,tno_dy)
    area_1 = gridcell_area(x_tno,y_tno,tno_dx,tno_dy/2)
    area_2 = gridcell_area(x_tno, y_tno+tno_dy/2, tno_dx, tno_dy/2)
    print("bottom area:", area_1/tno_area,"top area:",area_2/tno_area)
