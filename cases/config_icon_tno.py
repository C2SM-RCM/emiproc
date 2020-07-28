import os
import time

from emiproc.grids import COSMOGrid, TNOGrid, ICONGrid

# inventory
inventory = 'TNO'

# model either "cosmo-art", "cosmo-ghg" or "icon" (affects the
# output units and handling of the output grid)
model = 'icon'

# path to input inventory
input_path = "/input/TNOMACC/TNO_GHGco/TNO_6x6_GHGco_v1_1/TNO_GHGco_v1_1_year2015.nc"

# input grid
input_grid = TNOGrid(input_path)

# input species
species = ['co2_ff', 'co2_bf']

# input categories
categories = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F1",
    "F2",
    "F3",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
]

# mapping from input to output species (input is used for missing keys)
in2out_species = {
    'co2_ff': 'CO2',
    'co2_bf': 'CO2'
}

# mapping from input to output species (input is used for missing keys)
in2out_category = {}

# output variables are written in the following format using species and
# category after applying mapping as well as source_type (AREA or POINT) for
# TNO inventories
varname_format = '{species}_{category}' # not providing source_type will add up
                                        # point and area sources

# path to ICON output grid
icon_path = "/newhome/stem/git/C2SM-RCM/domain1_DOM01.nc"

# output ICON grid
output_grid = ICONGrid(icon_path)

# output path and filename
output_path = os.path.join('outputs', '{online}')
output_name = "tno.nc"

# resolution of shape file used for country mask
shpfile_resolution = "10m" 

# number of processes computing the mapping inventory->COSMO-grid
nprocs = 18

# metadata added as global attributes to netCDF output file
nc_metadata = {
    "DESCRIPTION": "Gridded annual emissions",
    "DATAORIGIN": "TNO",
    "CREATOR": "Jean-Matthieu Haussaire",
    "EMAIL": "jean-matthieu.haussaire@empa.ch",
    "AFFILIATION": "Empa Duebendorf, Switzerland",
    "DATE CREATED": time.ctime(time.time()),
}
