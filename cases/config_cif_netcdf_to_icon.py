import os
from pathlib import Path
import time

from emiproc.grids import COSMOGrid, TNOGrid, ICONGrid, LatLonNcGrid

# inventory comes from a cif regular grid
inventory = 'CIF_regular'

# model either "cosmo-art", "cosmo-ghg" or "icon" (affects the
# output units and handling of the output grid)
model = 'icon'

# path to input inventory
#input_path = '/mnt/c/Users/coli/Documents/CIF_interpolation/fluxes/original/flux_CH4_original.nc'
input_path = '/mnt/c/Users/coli/Documents/CIF_interpolation/fluxes/factor10/flux.s10.201804.nc'

# input grid
input_grid = LatLonNcGrid(input_path, lat_name='lat', lon_name='lon')

# input species
species = ['spec']


# mapping from input to output species (input is used for missing keys)
in2out_species = {
    'spec': 'Emiproc'
}

# mapping from input to output species (input is used for missing keys)
in2out_category = {}

# output variables are written in the following format using species and
# category after applying mapping as well as source_type (AREA or POINT) for
# TNO inventories
varname_format = '{species}_{category}' # not providing source_type will add up
                                        # point and area sources

# ICON domain
output_grid = ICONGrid('/mnt/c/Users/coli/Documents/CIF_interpolation/dyn_grid.nc')

# output path and filename
output_path = str(Path(input_path).parent / 'emiproc')
output_name = 'out_' + Path(input_path).name

shpfile_resolution = "10m" 
# number of processes computing the mapping inventory->COSMO-grid
nprocs = 6

# metadata added as global attributes to netCDF output file
nc_metadata = {
    "DESCRIPTION": "Regular grid emissions to ICON grid",
    "DATAORIGIN": f"{input_path}",
    "CREATOR": "Constantin Lionel",
    "EMAIL": "lionel.constantin@empa.ch",
    "AFFILIATION": "Empa Duebendorf, Switzerland",
    "DATE CREATED": time.ctime(time.time()),
}
