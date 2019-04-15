#!/usr/bin/env python

"""
Description: Functions related to the COSMO grid

Written by:  Michael Jaehn - michael.jaehn@empa.ch - 2018-07-04
"""

import netCDF4
import math
from math import pi, sin, cos, asin, atan2

def read_rotpole(filename):                                                    
    """ Read rotated pole coordinates from COSMO filename. """                 
                                                                                  
    with netCDF4.Dataset(filename) as nc:                                      
        rotpole = nc.variables['rotated_pole']                                 
        pollon = rotpole.getncattr('grid_north_pole_longitude')                
        pollat = rotpole.getncattr('grid_north_pole_latitude')                 

    return pollon, pollat 

def rotated_grid_transform(position, direction, north_pole):
    """
    position:   tuple(lon, lat) = input coordinate
    direction:  1 = Regular -> Rotated, 2 = Rotated -> Regular
    north_pole: tuple(lon, lat) = position of rotated north pole
    returns:    tuple(lon, lat) = output coordinate
    """
    lon = position[0]
    lat = position[1]

    # Convert degrees to radians
    lon = (lon * pi) / 180.0
    lat = (lat * pi) / 180.0

    NP_lon = north_pole[0] 
    NP_lat = north_pole[1]

    theta = 90 - NP_lat    # Rotation around y-axis
    phi = NP_lon - 180     # Rotation around z-axis


    # Convert degrees to radians
    phi = (phi * pi) / 180.0
    theta = (theta * pi) / 180.0

    # Convert from spherical to cartesian coordinates
    x = cos(lon) * cos(lat)
    y = sin(lon) * cos(lat)
    z = sin(lat)

    if direction == 1: # Regular -> Rotated
        
        x_new = cos(theta) * cos(phi) * x + cos(theta) * sin(phi) * y + sin(theta) * z
        y_new = -sin(phi) * x + cos(phi) * y
        z_new = -sin(theta) * cos(phi) * x - sin(theta) * sin(phi) * y + cos(theta) * z
        
    elif direction == 2: # Rotated -> Regular
        
        phi = -phi
        theta = -theta

        x_new = cos(theta) * cos(phi) * x + sin(phi) * y + sin(theta) * cos(phi) * z
        y_new = -cos(theta) * sin(phi) * x + cos(phi) * y - sin(theta) * sin(phi) * z
        z_new = -sin(theta) * x + cos(theta) * z
        
    else:
        raise Exception('Invalid direction, value must be either 1 or 2.')

    # Convert cartesian back to spherical coordinates
    lon_new = atan2(y_new, x_new)
    lat_new = asin(z_new)

    # Convert radians back to degrees
    lon_new = (lon_new * 180.0) / pi
    lat_new = (lat_new * 180.0) / pi

    return (lon_new, lat_new)
