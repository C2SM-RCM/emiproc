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

# species and categories read from input file
species = ["co2_ff", "co2_bf"]
categories = [
    "A",
#    "B",
#    "C",
#    "D",
#    "E",
    "F1",
    "F2",
    "F3",
#    "G",
#    "H",
#    "I",
#    "J",
#    "K",
#    "L",
]

# mapping from input to output species (input is used for missing keys)
in2out_species = {
    'co2_ff': 'CO2',
    'co2_bf': 'CO2'
}

# mapping from input to output category (input is used for missing keys)
in2out_category = {'F1': 'F', 'F2': 'F', 'F3': 'F'}


# output variables are written in the following format using species and
# category after applying the mapping
varname_format = '{species}_{category}_{source_type}'

# COSMO domain
cosmo_grid = COSMOGrid(
    nx=90,
    ny=60,
    dx=0.1,
    dy=0.1,
    xmin=-4.92,
    ymin=-3.18,
    pollon=-170.0,
    pollat=43.0,
)

offline = True
if offline:
    cosmo_grid.xmin -= 2 * cosmo_grid.dx
    cosmo_grid.ymin -= 2 * cosmo_grid.dy
    cosmo_grid.nx += 4
    cosmo_grid.ny += 4


# output path and filename
if offline:
    output_path = os.path.join("TNO-test", 'offline')
else:
    output_path = os.path.join("TNO-test", 'online')

output_name = "test-tno.nc"


# resolution of shapefile used for country mask
shpfile_resolution = "110m"

# number of processes computing the mapping inventory->COSMO-grid
nprocs = 18

# metadata added as global attributes to netCDF output file
nc_metadata = {
    "DESCRIPTION": "Gridded annual emissions",
    "DATAORIGIN": "TNO",
    "CREATOR": "Gerrit Kuhlmann",
    "EMAIL": "gerrit.kuhlmann@empa.ch",
    "AFFILIATION": "Empa Duebendorf, Switzerland",
    "DATE CREATED": time.ctime(time.time()),
}

# Add total emissions (only for swiss inventory)
add_total_emissions = False

