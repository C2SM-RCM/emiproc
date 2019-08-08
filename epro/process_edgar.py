import os
import time
from netCDF4 import Dataset
import numpy as np
from glob import glob
from . import utilities as util

def get_emi(filename, edgar_grid):
    lon_var, lat_var = edgar_grid.lon_range(), edgar_grid.lat_range()
    emi = np.zeros((len(lon_var), len(lat_var)))
    with open(filename) as f:
        for l in f:
            data = l.split(";")
            try:
                lat = float(data[0])
                lon = float(data[1])
                if (
                    (lat <= lat_var[-1])
                    and (lon <= lon_var[-1])
                    and (lat >= lat_var[0])
                    and (lon >= lon_var[0])
                ):
                    emi_lon = round((lon - edgar_grid.xmin) / edgar_grid.dx)
                    emi_lat = round((lat - edgar_grid.ymin) / edgar_grid.dy)

                    emi[emi_lon, emi_lat] = float(data[2])
            except ValueError:
                continue
    return emi

def process_edgar(cfg, interpolation, country_mask, out, latname, lonname):
    from . import get_out_varname
    """Starts writing out the output file"""
    output_path = ( os.path.join(cfg.output_path,cfg.output_name))
    
    """ EDGAR specific"""
    out_var = np.zeros((cfg.cosmo_grid.ny, cfg.cosmo_grid.nx))  # sum of all sources
    for cat in cfg.categories:
        path = os.path.join(cfg.input_path, cat)
        files = glob(path + "/*_2015_*")
        if len(files) > 1 or len(files) < 0:
            print("There are too many or too few files")
            print(files)
        else:
            filename = files[0]
        print(filename)

        start = time.time()

        emi = get_emi(filename, cfg.input_grid)
        for lon in range(emi.shape[0]):
            for lat in range(emi.shape[1]):
                for (x, y, r) in interpolation[lon, lat]:
                    # EDGAR inventory is in tons per grid cell
                    out_var[y, x] += emi[lon, lat] * r
        end = time.time()
        print("it takes ", end - start, "sec")

    """convert unit from ton.year-1.cell-1 to kg.m-2.s-1"""

    """calculate the areas (m^^2) of the COSMO grid"""
    cosmo_area = 1.0 / cfg.cosmo_grid.gridcell_areas()
    out_var *= cosmo_area.T / util.SEC_PER_YR * 1000

    out_var_name = get_out_varname(cfg.species,'',cfg)
    out.createVariable(out_var_name, float, (latname, lonname))
    out[out_var_name].units = "kg m-2 s-1"
    out[out_var_name][:] = out_var
