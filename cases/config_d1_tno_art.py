# "constant" paths and values for TNO, regular lat/lon
# for MeteoTest Swiss inventory, use calculated regular domain in the code

import os
import time

from epro.grids import COSMOGrid, TNOGrid

# inventory
inventory = 'TNO'

# model either "cosmo-art" or "cosmo-ghg" (affects the output units)
model = 'cosmo-art'

# input filename
input_path = "/input/TNOMACC/CAMS-REG-AP_v2_2/CAMS-REG-AP_v2_2_1_emissions_year2015.nc"

# input grid
input_grid = TNOGrid(input_path)

# input species
species = ['co', 'nox', 'nmvoc', 'so2', 'nh3', 'pm10', 'pm2_5']

# input categories
categories = ["A", "B", "C", "D", "E", "F1","F2","F3","F4",
              "G", "H", "I", "J", "K", "L" ]

# mapping from input to output species (input is used for missing keys)
in2out_species = {
    'co':    'CO',
    'nox':   'NOX',
    'nmvoc': 'NMVOC',
    'so2':   'SO2',
    'nh3':   'NH3',
    'pm10':  'PM10',
    'pm2_5': 'PM25'
}

# mapping from input to output categories (input is used for missing keys)
in2out_category = {
    'F1': 'F',
    'F2': 'F',
    'F3': 'F',
    'F4': 'F'
}

# output variables are written in the following format using species and
# category after applying the mapping
varname_format = '{species}_{category}_{source_type}'

# output path and filename
output_path = os.path.join('oae-art-example', '{online}', 'tno')
output_name = 'tno-art.nc'

# Output grid is European domain (rotated pole coordinates)
cosmo_grid = COSMOGrid(
    nx=192,
    ny=164,
    dx=0.12,
    dy=0.12,
    xmin=-16.08,
    ymin=-9.54,
    pollon=-170.0,
    pollat=43.0,
)

# resolution of shapefile used for country mask
shpfile_resolution = "110m"

# number of processes
nprocs = 16

# metadata added as global attributes to netCDF output file
nc_metadata = {
    "DESCRIPTION": "Gridded annual emissions",
    "DATAORIGIN": "TNO-CAMS",
    "CREATOR": "Qing Mu and Gerrit Kuhlmann",
    "EMAIL": "gerrit.kuhlmann@empa.ch",
    "AFFILIATION": "Empa Duebendorf, Switzerland",
    "DATE CREATED": time.ctime(time.time()),
}


