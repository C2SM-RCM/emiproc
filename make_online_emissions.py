#!/usr/bin/env python
# coding: utf-8

import importlib
import netCDF4 as nc
#import config as cfg
import numpy as np
import cartopy.crs as ccrs
from shapely.geometry import Polygon
import time
import itertools
import sys 
import cartopy.io.shapereader as shpreader

from country_code import country_codes
import os


# constants to convert from kg/yr to kg/s/m2
day_per_yr = 365.25
hr_per_yr = day_per_yr * 24.
sec_per_day = 86400
convfac = 1./(day_per_yr*sec_per_day)


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

def get_country_mask(cfg):
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


def var_name(s,snap,cat_kind):
    """Returns the name of a variable for a given species and snap
    input : 
       - s : species name ("CO2", "CH4" ...)
       - snap : Category number
       - cat_kind : Kind of category. must be "SNAP" or "NFR" 
    output :
       - returns a string which concatenate the species with the category number
    """
    out_var_name = s+"_"
    if cat_kind=="SNAP":
        if snap==70:                        
            out_var_name += "07_"
        else:
            if snap>9:
                out_var_name += str(snap)+"_"
            else:
                out_var_name += "0"+str(snap)+"_"
    elif cat_kind=="NFR":
        out_var_name += snap+"_"
    else:
        print("Wrong cat_kind in the config file. Must be SNAP or NFR")
        raise ValueError
    return out_var_name


def tno_file(species,cfg):
    """Returns the path to the TNO file for a given species.
    For some inputs, the CO2 is in a different file than the other species for instance.
    There are two parameters in the config file to distinguish them.
    Namely, tnoCamsPath and tnoMACCIIIPath    
    """

    # get the TNO inventory
    if species=="CO2":
        tno_path = cfg.tnoCamsPath
    else:
        tno_path = cfg.tnoMACCIIIPath

    if os.path.isfile(tno_path):
        return tno_path

    first_year = 2000
    last_year = 2011

    if cfg.year>last_year:
        base_year = str(last_year)
    elif cfg.year<first_year:
        base_year = str(first_year)
    else:
        base_year = str(cfg.year)
    tno_path += base_year+".nc"
    
    return tno_path



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



def interpolate_tno_to_cosmo_point(source,tno,cfg):
    """This function returns the indices of the cosmo grid cell that contains the point source
    input : 
       - source : The index of the  point source from the TNO inventory
       - tno : the TNO netCDF file, already open.
       - cfg : the configuration file
    output :
       - (cosmo_indx,cosmo_indy) : the indices of the cosmo grid cell containing the source
""" 

    lon_source = tno["longitude_source"][source]
    lat_source = tno["latitude_source"][source]

    transform = ccrs.RotatedPole(pole_longitude=cfg.pollon, pole_latitude=cfg.pollat)
    point = transform.transform_point(lon_source,lat_source,ccrs.PlateCarree())

    cosmo_indx = int(np.floor((point[0]-cfg.xmin)/cfg.dx))
    cosmo_indy = int(np.floor((point[1]-cfg.ymin)/cfg.dy))

    return (cosmo_indx,cosmo_indy)
    
def interpolate_tno_to_cosmo_grid(tno,out,cfg):
    """This function determines which COSMO cell coincides with a TNO cell
    input : 
       - tno : the TNO netCDF file, already open
       - out : the output netCDF file, already open
       - cfg  : the configuration file
    output : 
       It produces an array of dimension (tno_lon,tno_lat)
       Each element will be a list containing triplets (x,y,r) 
          - x : index of the longitude of cosmo grid cell 
          - y : index of the latitude of cosmo grid cell
          - r : ratio of the area of the intersection compared to the area of the tno cell.
       To avoid having a lot of r=0, we only keep the cosmo cells that intersect the tno cell.

    For a TNO grid of (720,672) and a Berlin domain with resolution 0.1°, it takes 5min to run
"""

    print("Retrieving the interpolation between the cosmo and the tno grids")
    start = time.time()
    transform = ccrs.RotatedPole(pole_longitude=cfg.pollon, pole_latitude=cfg.pollat)
    cosmo_xlocs = np.arange(cfg.xmin, cfg.xmin+cfg.dx*cfg.nx, cfg.dx)
    cosmo_ylocs = np.arange(cfg.ymin, cfg.ymin+cfg.dy*cfg.ny, cfg.dy)

    var_out = np.zeros((out.dimensions["rlat"].size,out.dimensions["rlon"].size))
    if var_out.shape != (len(cosmo_ylocs),len(cosmo_xlocs)):
        print("Wrong dimensions in the output file, compared to the configuration.")
        print(var_out.shape)
        print((len(cosmo_ylocs),len(cosmo_xlocs)))
        return np.zeros(1)

    tno_lon = tno.dimensions["longitude"].size
    tno_lat = tno.dimensions["latitude"].size
    
    """This is the interpolation that will be returned"""
    mapping = np.empty((tno_lon,tno_lat),dtype=object)
            
    incr = 0
    start = time.time()
    for i in range(tno_lon):
        for j in range(tno_lat):
            """Initialization"""
            mapping[i,j]=[]

            """Progress bar"""
            incr+=1
            if int(incr/100) == incr/100.:
                sys.stdout.write('\r')
                sys.stdout.write(" {:.1f}%".format((100/((tno_lon*tno_lat)-1)*((i*tno_lat)+j))))
                sys.stdout.flush()            

            """Get the middle of the tno cell"""
            # x_tno=tno["longitude_source"][i]
            # y_tno=tno["latitude_source"][i]
            x_tno = tno["longitude"][i]
            y_tno = tno["latitude"][j]
            
            """Get the corners of the tno cell"""
            tno_cell_x= np.array([x_tno+cfg.tno_dx/2,x_tno+cfg.tno_dx/2,x_tno-cfg.tno_dx/2,x_tno-cfg.tno_dx/2])
            tno_cell_y= np.array([y_tno+cfg.tno_dy/2,y_tno-cfg.tno_dy/2,y_tno-cfg.tno_dy/2,y_tno+cfg.tno_dy/2])

            """Make a polygon out of it, and get its area."""
            """in cosmo grid"""
            """Note : the unit/meaning of the area is in degree^2, but it doesn't matter since we only want the ratio. we assume the area is on a flat earth."""
            points = transform.transform_points(ccrs.PlateCarree(),tno_cell_x,tno_cell_y)

            polygon_tno = Polygon(points)
            area_tno = polygon_tno.area

            """Find the cosmo cells that intersect it"""
            for (a,x) in enumerate(cosmo_xlocs):
                """Get the corners of the cosmo cell"""
                cosmo_cell_x = [x+cfg.dx/2,x+cfg.dx/2,x-cfg.dx/2,x-cfg.dx/2]
                if (min(cosmo_cell_x)>max([k[0] for k in points])) or (max(cosmo_cell_x)<min([k[0] for k in points])):
                    continue

                for (b,y) in enumerate(cosmo_ylocs):                    
                    cosmo_cell_y = [y+cfg.dy/2,y-cfg.dy/2,y-cfg.dy/2,y+cfg.dy/2]

                    if (min(cosmo_cell_y)>max([k[1] for k in points])) or (max(cosmo_cell_y)<min([k[1] for k in points])):
                        continue

                    points_cosmo = [k for k in zip(cosmo_cell_x,cosmo_cell_y)]
                    polygon_cosmo = Polygon(points_cosmo)

                    if polygon_cosmo.intersects(polygon_tno):
                        inter = polygon_cosmo.intersection(polygon_tno)
                        mapping[i,j].append((a,b,inter.area/area_tno))

    end = time.time()
    print("\nInterpolation is over")
    print("it took ",end-start,"seconds")
    #return var_out
    np.save(os.path.join(cfg.output_path,"mapping.npy"),mapping)
    return mapping
        

def main(cfg_path):
    """ The main script, taking a configuration file as input"""
    
    """try to load config file"""
    try:
        sys.path.append(os.path.dirname(os.path.realpath(cfg_path)))
        cfg = importlib.import_module(os.path.basename(cfg_path))
    except IndexError:
        print('ERROR: no config file provided!')
        sys.exit(1)
    except ImportError:
        print('ERROR: failed to import config module "%s"!' % os.path.basename(cfg_path))
        sys.exit(1)


    output_path = cfg.output_path+"emis_"+str(cfg.year)+"_"+cfg.gridname+".nc"
    with nc.Dataset(output_path,"w") as out:
        """Create the dimensions and the rotated pole"""
        lonname = "rlon"; latname="rlat"
        if cfg.pollon==180 and cfg.pollat==90:
            lonname = "lon"; latname="lat"
        else:
            out.createVariable("rotated_pole",str)
            out["rotated_pole"].grid_mapping_name = "rotated_latitude_longitude"
            out["rotated_pole"].grid_north_pole_latitude = 43.
            out["rotated_pole"].grid_north_pole_longitude = -170.
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

        
        """Calculate the country mask"""
        add_country_mask = True
        cmask_path = os.path.join(cfg.output_path,"country_mask.npy")
        if os.path.isfile(cmask_path):
            print("Do you wanna overwite the country mask found in %s ?" % cmask_path)
            s = input("y/[n] \n")
            add_country_mask = (s=="y")
       
        if add_country_mask:
            country_mask = get_country_mask(cfg)
        else:
            country_mask = np.load(cmask_path)

        mask_name = "country_ids"
        out.createVariable(mask_name,"short",(latname,lonname))
        out[mask_name].long_name = "EMEP_country_code"
        out[mask_name][:] = country_mask.T

        list_input_files = set([tno_file(s,cfg) for s in cfg.species])
        for f in list_input_files:
            print(f)            
            with nc.Dataset(f) as tno:
                """retrieve the interpolation between the tno and cosmo grids."""
                make_interpolate = True            
                mapping_path = os.path.join(cfg.output_path,"mapping.npy")
                if os.path.isfile(mapping_path):
                    print("Do you wanna overwite the mapping found in %s ?" % mapping_path)
                    s = input("y/[n] \n")
                    make_interpolate = (s=="y")
                 
                if make_interpolate:
                    interpolation = interpolate_tno_to_cosmo_grid(tno,out,cfg)
                else:
                    interpolation = np.load(mapping_path)

                ## FOR DEBUGGING
                # interpolation = np.empty((720,672),dtype=object)
                # for i in range(720):
                #     for j in range(672):
                #         interpolation[i,j]=[(int(74*np.random.rand()),int(64*np.random.rand()),1)]
                # print(interpolation[0,101])
                
                """mask corresponding to the area/point sources"""
                selection_area  = tno["source_type_index"][:]==1
                selection_point = tno["source_type_index"][:]==2

                # SNAP ID 
                #tno_snap = tno[cfg.tno_cat_var][:].tolist() 
                tno_snap = cfg.tno_snap
                    
                for snap in cfg.snap:
                    """In emission_category_index, we have the index of the category, starting with 1.
                    It means that if the emission is of SNAP1, it will have index 1, SNAP34 index 3"""
                    if snap==70:
                        snap_list=[i for i in range(70,80) if i in tno_snap]
                    elif snap=="F":
                        snap_list=["F1","F2","F3"]
                    else:
                        snap_list=[snap]
                    
                    print(snap_list, tno_snap)
                    """mask corresponding to the given snap category"""
                    selection_snap = np.array([tno["emission_category_index"][:] == tno_snap.index(i)+1 for i in snap_list])
                    
                    """mask corresponding to the given snap category for area/point"""
                    selection_snap_area  = np.array([selection_snap.any(0),selection_area]).all(0)
                    selection_snap_point = np.array([selection_snap.any(0),selection_point]).all(0)
                    
                    species_list = [s for s in cfg.species if tno_file(s,cfg)==f]
                    for s in species_list:
                        print("Species",s,"SNAP",snap)
                        out_var_area = np.zeros((cfg.ny,cfg.nx))
                        out_var_point = np.zeros((cfg.ny,cfg.nx))

                        if s=="CO2":
                            """add fossil and bio fuel CO2"""
                            var = tno["co2_ff"][:]+tno["co2_bf"][:]
                        elif s=="CO":
                            """add fossil and bio fuel CO"""
                            var = tno["co_ff"][:]+tno["co_bf"][:]                      
                        elif s=="PM2.5":
                            var = tno["pm2_5"]
                        else:
                            var = tno[s.lower()][:]

                        start = time.time()
                        for (i,source) in enumerate(var):
                            if selection_snap_area[i]:
                                lon_ind = tno["longitude_index"][i]-1
                                lat_ind = tno["latitude_index"][i]-1
                                for (x,y,r) in interpolation[lon_ind,lat_ind]:
                                        out_var_area[y,x]+=var[i]*r
                            if selection_snap_point[i]:
                                (indx,indy) = interpolate_tno_to_cosmo_point(i,tno,cfg)
                                if indx>=0 and indx<cfg.nx and indy>=0 and indy<cfg.ny:
                                    out_var_point[indy,indx]+=var[i]

                        end = time.time()
                        print("it takes ",end-start,"sec")                     
                        ## TO DO : 
                        ## - Add the factor from 2011 to 2015
                        

                        """convert unit from kg.year-1.cell-1 to kg.m-2.s-1"""

                        """calculate the areas (m^^2) of the COSMO grid"""
                        cosmo_area = 1./gridbox_area(cfg)
                        out_var_point*= cosmo_area.T*convfac
                        out_var_area *= cosmo_area.T*convfac

                        out_var_name = var_name(s,snap,cfg.cat_kind)
                        for (t,sel,out_var) in zip(["AREA","POINT"],
                                           [selection_snap_area,selection_snap_point],
                                           [out_var_area,out_var_point]):
                            if sel.any():
                                out.createVariable(out_var_name+t,float,(latname,lonname))
                                out[out_var_name+t].units = "kg m-2 s-1"
                                out[out_var_name+t][:] = out_var

                        

    return interpolation

if __name__ == "__main__":
    mapping = main("./config_CHE")
    #mask = get_country_mask()
