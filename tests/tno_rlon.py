import os
import time

from emiproc.grids import COSMOGrid, TNOGrid, ICONGrid

# inventory
inventory = 'TNO'

# model either "cosmo-art", "cosmo-ghg" or "icon" (affects the
# output units and handling of the output grid)
model = 'cosmo-ghg'

# path to input inventory
input_path = "test_emissions.nc"

# input grid
input_grid = TNOGrid(input_path)

# input species
species = ['co2_ff']

# input categories
categories = [
  "A",
]

# mapping from input to output species (input is used for missing keys)
in2out_species = {
    'co2_ff': 'CO2',
}

# mapping from input to output species (input is used for missing keys)
in2out_category = {
}

# output variables are written in the following format using species and
# category after applying mapping as well as source_type (AREA or POINT) for
# TNO inventories
varname_format = '{species}_{category}_{source_type}' # not providing source_type will add up
                                                      # point and area sources

# COSMO domain
output_grid = COSMOGrid(
    nx=26, #900
    ny=26, #600
    dx=0.01, #0.01
    dy=0.01, #0.01
    xmin=-0.13,
    ymin=-0.13,
    pollon=-169.0,
    pollat=43.0,
)


# output path and filename
output_path = os.path.join('outputs', '{online}')
output_name = "outgrid.nc"

# resolution of shape file used for country mask
shpfile_resolution = "110m" 

# number of processes computing the mapping inventory->COSMO-grid
nprocs = 18

# metadata added as global attributes to netCDF output file
nc_metadata = {
    "DESCRIPTION": "Gridded annual emissions",
    "DATAORIGIN": "TNO",
    "CREATOR": "Erik Koene",
    "EMAIL": "erik.koene@empa.ch",
    "AFFILIATION": "Empa Duebendorf, Switzerland",
    "DATE CREATED": time.ctime(time.time()),
}
