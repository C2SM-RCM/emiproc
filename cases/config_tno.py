import os
import time

from epro.grids import COSMOGrid, TNOGrid

# inventory
inventory = 'TNO'

# model either "cosmo-art" or "cosmo-ghg" (affect the output units)
model = 'cosmo-ghg'

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

# COSMO domain
cosmo_grid = COSMOGrid(
    nx=900,
    ny=600,
    dx=0.01,
    dy=0.01,
    xmin=-4.92,
    ymin=-3.18,
    pollon=-170.0,
    pollat=43.0,
)

offline = False
if offline:
    cosmo_grid.xmin -= 2 * cosmo_grid.dx
    cosmo_grid.ymin -= 2 * cosmo_grid.dy
    cosmo_grid.nx += 4
    cosmo_grid.ny += 4

# output path and filename
output_path = "TNO"
output_name = "tno.nc"

# resolution of shape file used for country mask
shpfile_resolution = "110m" # TODO: set back to 10m

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
