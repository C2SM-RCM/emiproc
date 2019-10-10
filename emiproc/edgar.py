import os
import time
from netCDF4 import Dataset
import numpy as np
from glob import glob
from . import utilities as util
import pandas

def get_emi(filename, edgar_grid):
    lon_var, lat_var = edgar_grid.lon_range(), edgar_grid.lat_range()
    gridded_emi = np.zeros((len(lon_var), len(lat_var)))
    f = pandas.read_csv(filename,header=2,sep=';',names=['lat','lon','emi']) 
    lats = f['lat']
    lons = f['lon']
    emis = f['emi']
    for lat,lon,emi in zip(lats,lons,emis):
        if (
            (lat <= lat_var[-1])
            and (lon <= lon_var[-1])
            and (lat >= lat_var[0])
            and (lon >= lon_var[0])
        ):
            lon_idx = round((lon - edgar_grid.xmin) / edgar_grid.dx)
            lat_idx = round((lat - edgar_grid.ymin) / edgar_grid.dy)
            gridded_emi[lon_idx, lat_idx] = emi
    return gridded_emi
    

def process_edgar(cfg, interpolation, country_mask, out, latname, lonname):
    """
    Process EDGAR inventory.
    """
    out_var = np.zeros((cfg.cosmo_grid.ny, cfg.cosmo_grid.nx))  # sum of all sources
    for s in cfg.species: #Only tested for CO2
        for cat in cfg.categories:
            path = os.path.join(cfg.input_path, cat)
            files = glob(os.path.join(path, "*"+s+"*"+str(cfg.input_year)+"*")) 
            if len(files) != 1:
                print("There are too many or too few files")
                print(files)
            else:
                filename = files[0]
            print(filename)

            start = time.time()

            emi = get_emi(filename, cfg.input_grid)
            for lon_idx, lat_idx in np.ndindex(emi.shape):
                for (x, y, r) in interpolation[lon_idx, lat_idx]:
                    # EDGAR inventory is in tons per grid cell
                    out_var[y, x] += emi[lon_idx, lat_idx] * r
            end = time.time()
            print("it takes ", end - start, "sec")

            # convert unit from ton.year-1.cell-1 to kg.m-2.s-1

            # calculate the areas (m^^2) of the COSMO grid
            cosmo_area = 1.0 / cfg.cosmo_grid.gridcell_areas()
            out_var *= cosmo_area.T / util.SEC_PER_YR * 1000

            out_var_name = util.get_out_varname(s, cat, cfg)            
            print('Write as variable:', out_var_name)
            unit = "kg m-2 s-1"
            util.write_variable(out, out_var, out_var_name,
                                latname, lonname, unit)
