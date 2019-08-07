import os
import time

from epro.grids import COSMOGrid, SwissGrid

# inventory
inventory = 'swiss-cc'

# model either "cosmo-art" or "cosmo-ghg" (affects the output units)
model = 'cosmo-ghg'

# path to input inventory
input_path = "/input/CH_EMISSIONS/CarboCountCO2/einzelgrids/"

# input grid (for Swiss inventory, unit m, x is easterly, y is northly)
input_grid = SwissGrid(
    name="carbocount",
    nx=760,
    ny=500,
    dx=500,
    dy=500,
    xmin=470000,
    ymin=60000
)

# species and categories read from input files
species = ["co2"]

categories = [
    "bm",
    "cf",
]

# mapping from input to output species (input is used for missing keys)
in2out_species = {'co2': 'CO2'}

# mapping from input to output categories (input is used missing keys)
in2out_category = {
    "bm": "B",
    "cf": "B",
    "df": "C",
    "hf": "C",
    "hk": "C",
    "if": "B",
    "ka": "J",
    "ki": "J",
    "ks": "J",
    "kv": "J",
    "la": "J",
    "lf": "L",
    "lw": "L",
    "mi": "B",
    "mt": "C",
    "nf": "B",
    "pf": "B",
    "pq": "B",
    "rf": "B",
    "vk": "F",
    "zp": "B",
    "zv": "F",
}

# output variables are written in the following format using species and
# categeory after applying the mapping
varname_format = '{species}_{category}'

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

# output path
output_path = "Swiss-test"

# output filename
output_name = 'test_swiss-cc.nc'

# resolution of shapefile used for country mask
shpfile_resolution = "110m"

# number of processes 
nprocs = 18

# metadata added as global attributes to netCDF output file
nc_metadata = {
    "DESCRIPTION": "Gridded annual emissions",
    "DATAORIGIN": "carbocount-CH",
    "CREATOR": "Gerrit Kuhlmann",
    "EMAIL": "gerrit.kuhlmann@empa.ch",
    "AFFILIATION": "Empa Duebendorf, Switzerland",
    "DATE CREATED": time.ctime(time.time()),
}

# Add total emissions
add_total_emissions = True


