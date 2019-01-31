#!/usr/bin/env python
# coding: utf-8

import importlib
import netCDF4 as nc
import numpy as np
import cartopy.crs as ccrs
from shapely.geometry import Polygon
import time
import itertools
import sys 
import cartopy.io.shapereader as shpreader
from multiprocessing import Pool
from country_code import country_codes
import os


# constants to convert from kg/yr to kg/s/m2
day_per_yr = 365.25
hr_per_yr = day_per_yr * 24.
sec_per_day = 86400
convfac = 1./(day_per_yr*sec_per_day)
npool = 18

def load_cfg(cfg_path):
    """Load config file"""
    try:
        sys.path.append(os.path.dirname(os.path.realpath(cfg_path)))
        cfg = importlib.import_module(os.path.basename(cfg_path))
    except IndexError:
        print('ERROR: no config file provided!')
        sys.exit(1)
    except ImportError:
        print('ERROR: failed to import config module "%s"!' % os.path.basename(cfg_path))
        sys.exit(1)

    return cfg

def gridbox_area(cfg):
    """Calculate 2D array of the areas (m^^2) of the output COSMO grid"""
    radius=6375000. #the earth radius in meters
    deg2rad=np.pi/180.
    dlat = cfg.dy*deg2rad
    dlon = cfg.dx*deg2rad

    """Box area at equator"""
    dd=2.*pow(radius,2)*dlon*np.sin(0.5*dlat)
    areas = np.array([[dd*np.cos(deg2rad*cfg.ymin+j*dlat) for j in range(cfg.ny)] for foo in range(cfg.nx)])
    return areas 

def prepare_output_file(cfg,out,country_mask):
    """Starts writing out the output file :
       - Dimensions and variables for longitude and latitude
       - Rotated pole
       - Country mask variable
    inputs:
       - cfg : the config file
       - out : the netcdf output file, already open
       - country_mask : the country_mask that has already been calculated prior
    outputs:
       None
    """       

    """Create the dimensions and the rotated pole"""
    lonname = "rlon"; latname="rlat"
    if cfg.pollon==180 and cfg.pollat==90:
        lonname = "lon"; latname="lat"
    else:
        out.createVariable("rotated_pole",str)
        out["rotated_pole"].grid_mapping_name = "rotated_latitude_longitude"
        out["rotated_pole"].grid_north_pole_latitude = cfg.pollat
        out["rotated_pole"].grid_north_pole_longitude = cfg.pollon
        out["rotated_pole"].north_pole_grid_longitude = 0.

    out.createDimension(lonname,cfg.nx)
    out.createDimension(latname,cfg.ny)
    
    """Create the variable associated to the dimensions"""
    out.createVariable(lonname,"float32",lonname)
    out[lonname].axis = "X"
    out[lonname].units = "degrees"
    out[lonname].standard_name = "longitude"
    out[lonname][:] = np.arange(cfg.xmin,cfg.xmin+cfg.dx*cfg.nx,cfg.dx)

    out.createVariable(latname,"float32",latname)
    out[latname].axis = "Y"
    out[latname].units = "degrees"
    out[latname].standard_name = "latitude"
    out[latname][:] = np.arange(cfg.ymin,cfg.ymin+cfg.dy*cfg.ny,cfg.dy)

    """Create the variable associated with the country_mask"""
    mask_name = "country_ids"
    out.createVariable(mask_name,"short",(latname,lonname))
    out[mask_name].long_name = "EMEP_country_code"
    out[mask_name][:] = country_mask.T



##################################
##  Regarding the country mask  ##
##################################
def check_country(country,points):
    """For a given country, return if the grid cell defined by points is within the country
    input : 
       - country
       - points : the (latitude, longitude) of the four corners of a cell
    output :
       - True if the grid cell is within the country
    """
    bounds = country.bounds #(minx, miny, maxx, maxy)
    if ((bounds[0]>max([k[0] for k in points])) or 
        (bounds[2]<min([k[0] for k in points])) or
        (bounds[1]>max([k[1] for k in points])) or 
        (bounds[3]<min([k[1] for k in points]))):
        return False
    else:
        return True

def compute_country_mask(cfg):
    """Returns the name of the country for each cosmo grid cell.
    If for a given grid cell, no country is found (Ocean for example), the country code 0 is assigned.
    input :
       - cfg : config file
    output :
       - Return a country mask. For each grid cell of the output domain, a single country code is determined.
    
    """
    start = time.time()
    print("Creating the country mask")
    natural_earth = True
    if natural_earth:
        shpfilename = shpreader.natural_earth(resolution='110m',
                                              category='cultural', name='admin_0_countries')
        iso3 = "ADM0_A3"
    else:
        shpfilename = "/usr/local/exelis/idl82/resource/maps/shape/cntry08.shp"
        iso3 = "ISO_3DIGIT"

    reader = shpreader.Reader(shpfilename)

    country_mask = np.empty((cfg.nx,cfg.ny))
   
 
    cosmo_xlocs = np.arange(cfg.xmin, cfg.xmin+cfg.dx*cfg.nx, cfg.dx)
    cosmo_ylocs = np.arange(cfg.ymin, cfg.ymin+cfg.dy*cfg.ny, cfg.dy)
    transform = ccrs.RotatedPole(pole_longitude=cfg.pollon, pole_latitude=cfg.pollat)
    incr=0
    no_country_code=[]

    european = []
    non_euro = []
    for country in reader.records():
        if country.attributes["CONTINENT"]=="Europe":
            european.append(country)
        else:
            non_euro.append(country)

    for (a,x) in enumerate(cosmo_xlocs):
        for (b,y) in enumerate(cosmo_ylocs):
            """Progress bar"""
            incr+=1
            sys.stdout.write('\r')
            sys.stdout.write(" {:.1f}%".format((100/((cfg.nx*cfg.ny)-1)*((a*cfg.ny)+b))))
            sys.stdout.flush()            

            mask = []

            """Get the corners of the cell in lat/lon coord"""
            # TO CHECK : is it indeed the bottom left corner ?
            # cosmo_cell_x = [x+cfg.dx,x+cfg.dx,x,x]
            # cosmo_cell_y = [y+cfg.dy,y,y,y+cfg.dy]
            # Or the center of the cell
            cosmo_cell_x = np.array([x+cfg.dx/2,x+cfg.dx/2,x-cfg.dx/2,x-cfg.dx/2])
            cosmo_cell_y = np.array([y+cfg.dy/2,y-cfg.dy/2,y-cfg.dy/2,y+cfg.dy/2])

            points = ccrs.PlateCarree().transform_points(transform,cosmo_cell_x,cosmo_cell_y)
            polygon_cosmo = Polygon(points)

            """To be faster, only check european countries at first"""
            for country in european:#reader.records():
                if check_country(country,points):
                    # if x+cfg.dx<bounds[0] or y+cfg.dy<bounds[1] or x>bounds[2] or y>bounds[3]:
                    #     continue
                    if polygon_cosmo.intersects(country.geometry):
                        mask.append(country.attributes[iso3])

            """If not found among the european countries, check elsewhere"""
            if len(mask)==0:
                for country in non_euro:#reader.records():
                    if check_country(country,points):
                        # if x+cfg.dx<bounds[0] or y+cfg.dy<bounds[1] or x>bounds[2] or y>bounds[3]:
                        #     continue
                        if polygon_cosmo.intersects(country.geometry):
                            mask.append(country.attributes[iso3])
                

            """If more than one country, assign the one which has the greatest area"""
            if len(mask)>1:
                area = 0
                for country in [rec for rec in reader.records() if rec.attributes[iso3] in mask]:
                    new_area = polygon_cosmo.intersection(country.geometry).area
                    if area <new_area:
                        area = new_area
                        new_mask = [country.attributes[iso3]]
                mask= new_mask

            """Convert the name to ID"""
            if len(mask)==1:
                try:
                    mask = [country_codes[mask[0]]]
                except KeyError:
                    no_country_code.append(mask[0])
                    mask=[-1]

            # If no country (ocean), then assign the ID 0            
            if len(mask)==0:
                mask=[0]
                            
            country_mask[a,b]=mask[0]

    print("\nCountry mask is done")
    end= time.time()
    print("it took",end-start,"seconds")
    if len(no_country_code)>0:
        print("The following countries were found, but didn't have a corresponding code")
        print(set(no_country_code))
    np.save(os.path.join(cfg.output_path,"country_mask.npy"),country_mask)
    return country_mask


def get_country_mask(cfg):
    """Calculate the country mask"""
    add_country_mask = True
    cmask_path = os.path.join(cfg.output_path,"country_mask.npy")
    if os.path.isfile(cmask_path):
        print("Do you wanna overwite the country mask found in %s ?" % cmask_path)
        s = input("y/[n] \n")
        add_country_mask = (s=="y")

    if add_country_mask:
        country_mask = compute_country_mask(cfg)
    else:
        country_mask = np.load(cmask_path)
    return country_mask


###################################
##  Regarding the point sources  ##
###################################
def interpolate_to_cosmo_point(lat_source,lon_source,cfg,proj=ccrs.PlateCarree()):
    """This function returns the indices of the cosmo grid cell that contains the point source
    input : 
       - lat_source  : The latitude of the point source
       - lon_source  : The longitude of the point source
       - cfg : The configuration file
       - proj : The cartopy projection of the lat/lon of the point source
    output :
       - (cosmo_indx,cosmo_indy) : the indices of the cosmo grid cell containing the source
""" 

    transform = ccrs.RotatedPole(pole_longitude=cfg.pollon, pole_latitude=cfg.pollat)
    point = transform.transform_point(lon_source,lat_source,proj)

    cosmo_indx = int(np.floor((point[0]-cfg.xmin)/cfg.dx))
    cosmo_indy = int(np.floor((point[1]-cfg.ymin)/cfg.dy))

    return (cosmo_indx,cosmo_indy)



##################################
##  Regarding the area sources  ##
##################################
def interpolate_single_cell(cfg,points):
    """ - points are the corner of the cell in the inventory"""

    """Information about the cosmo grid"""
    cosmo_xlocs = np.arange(cfg["xmin"], cfg["xmin"]+cfg["dx"]*cfg["nx"], cfg["dx"])
    cosmo_ylocs = np.arange(cfg["ymin"], cfg["ymin"]+cfg["dy"]*cfg["ny"], cfg["dy"])

    """This is the interpolation that will be returned"""
    """Initialization"""
    mapping=[]

    """Make a polygon out of the points, and get its area."""
    """Note : the unit/meaning of the area is in degree^2, but it doesn't matter since we only want the ratio. we assume the area is on a flat earth."""

    polygon_tno = Polygon(points)
    area_tno = polygon_tno.area

    """Find the cosmo cells that intersect it"""
    for (a,x) in enumerate(cosmo_xlocs):
        """Get the corners of the cosmo cell"""
        cosmo_cell_x = [x+cfg["dx"]/2,x+cfg["dx"]/2,x-cfg["dx"]/2,x-cfg["dx"]/2]
        if (min(cosmo_cell_x)>max([k[0] for k in points])) or (max(cosmo_cell_x)<min([k[0] for k in points])):
            continue

        for (b,y) in enumerate(cosmo_ylocs):                    
            cosmo_cell_y = [y+cfg["dy"]/2,y-cfg["dy"]/2,y-cfg["dy"]/2,y+cfg["dy"]/2]

            if (min(cosmo_cell_y)>max([k[1] for k in points])) or (max(cosmo_cell_y)<min([k[1] for k in points])):
                continue

            points_cosmo = [k for k in zip(cosmo_cell_x,cosmo_cell_y)]
            polygon_cosmo = Polygon(points_cosmo)

            if polygon_cosmo.intersects(polygon_tno):
                inter = polygon_cosmo.intersection(polygon_tno)
                mapping.append((a,b,inter.area/area_tno))
    return mapping

def cell_corners(lon_var,lat_var,inv_name,i,j,cfg):
    if inv_name == "tno":
        x_tno = lon_var[i]
        y_tno = lat_var[j]
        cell_x= np.array([
            x_tno+cfg.tno_dx/2,
            x_tno+cfg.tno_dx/2,
            x_tno-cfg.tno_dx/2,
            x_tno-cfg.tno_dx/2])
        cell_y= np.array([
            y_tno+cfg.tno_dy/2,
            y_tno-cfg.tno_dy/2,
            y_tno-cfg.tno_dy/2,
            y_tno+cfg.tno_dy/2])
        proj = ccrs.PlateCarree()
    elif inv_name == "vprm":
        globe = ccrs.Globe(ellipse=None,semimajor_axis = 6370000,semiminor_axis = 6370000)
        lambert = ccrs.LambertConformal(
            central_longitude = 12.5, central_latitude = 51.604,
            standard_parallels=[51.604],globe=globe)
        
        center_lambert = lambert.transform_point(lon_var[j,i],lat_var[j,i],ccrs.PlateCarree())
        cell_x = np.array([
            center_lambert[0]+cfg.tno_dx/2,
            center_lambert[0]+cfg.tno_dx/2,
            center_lambert[0]-cfg.tno_dx/2,
            center_lambert[0]-cfg.tno_dx/2])
        cell_y = np.array([
            center_lambert[1]+cfg.tno_dy/2,
            center_lambert[1]-cfg.tno_dy/2,
            center_lambert[1]-cfg.tno_dy/2,
            center_lambert[1]+cfg.tno_dy/2])
        proj = lambert
    elif inv_name == "edgar":
        x_tno = lon_var[i]
        y_tno = lat_var[j]
        cell_x= np.array([
            x_tno+cfg.edgar_dx,
            x_tno+cfg.edgar_dx,
            x_tno,
            x_tno])
        cell_y= np.array([
            y_tno+cfg.edgar_dy,
            y_tno-cfg.edgar_dy,
            y_tno,
            y_tno])
        proj = ccrs.PlateCarree()
    else:
        print("Inventory %s is not supported yet. Consider defining your own or using tno or vprm." % inv_name)
        


    return cell_x,cell_y,proj

def get_dim_var(inv,inv_name,cfg):
    if inv_name == "tno":
        lon_dim = inv.dimensions["longitude"].size
        lat_dim = inv.dimensions["latitude"].size
        lon_var = inv["longitude"][:]
        lat_var = inv["latitude"][:]
    elif inv_name == "vprm":
        lon_dim = inv.dimensions["west_east"].size
        lat_dim = inv.dimensions["south_north"].size
        lon_var = inv["lon"][:]
        lat_var = inv["lat"][:]
    elif inv_name == "edgar":
        lon_var = np.arange(cfg.edgar_xmin,cfg.edgar_xmax,cfg.edgar_dx)
        lat_var = np.arange(cfg.edgar_ymin,cfg.edgar_ymax,cfg.edgar_dy)
        lon_dim = len(lon_var)
        lat_dim = len(lat_var)
    else:
        print("Inventory %s is not supported yet. Consider defining your own or using tno or vprm." % inv_name)

    return lon_dim,lat_dim,lon_var,lat_var


def interpolate_to_cosmo_grid(tno,inv_name,cfg):
    print("Retrieving the interpolation between the cosmo and the inventory grids")
    start = time.time()
   
    transform = ccrs.RotatedPole(pole_longitude=cfg.pollon, pole_latitude=cfg.pollat)

    lon_dim,lat_dim,lon_var,lat_var = get_dim_var(tno,inv_name,cfg)

    """This is the interpolation that will be returned"""
    mapping = np.empty((lon_dim,lat_dim),dtype=object)

    # Annoying ...
    """Transform the cfg to a dictionary"""
    config = dict({
        "dx" : cfg.dx,
        "dy" : cfg.dy,
        "nx" : cfg.nx,
        "ny" : cfg.ny,
        "xmin" : cfg.xmin,
        "ymin" : cfg.ymin})


    with Pool(npool) as pool:
        for i in range(lon_dim):
            print("ongoing :",i)
            points = []
            for j in range(lat_dim):
                tno_cell_x,tno_cell_y,proj = cell_corners(lon_var,lat_var,inv_name,i,j,cfg)
                points.append(transform.transform_points(proj,tno_cell_x,tno_cell_y))

            mapping[i,:] = pool.starmap(interpolate_single_cell,[(config,points[j]) for j in range(lat_dim)])


    end = time.time()
    print("\nInterpolation is over")
    print("it took ",end-start,"seconds")

    np.save(os.path.join(cfg.output_path,"mapping.npy"),mapping)
    return mapping



def get_interpolation(cfg,tno,inv_name = "tno",filename="mapping.npy"):
    """retrieve the interpolation between the tno and cosmo grids."""
    make_interpolate = True            
    mapping_path = os.path.join(cfg.output_path,filename)
    if os.path.isfile(mapping_path):
        print("Do you wanna overwite the mapping found in %s ?" % mapping_path)
        s = input("y/[n] \n")
        make_interpolate = (s=="y")
        
    if make_interpolate:
        interpolation = interpolate_to_cosmo_grid(tno,inv_name,cfg)
    else:
        interpolation = np.load(mapping_path)

    return interpolation
