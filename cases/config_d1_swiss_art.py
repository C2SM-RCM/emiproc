import os
import time

from emiproc.grids import COSMOGrid, SwissGrid

# inventory
inventory = 'swiss-art'

# model is "cosmo-art" or "cosmo-ghg" (affects the output units)
model = 'cosmo-art'

# add total emissions (only for swiss inventory)
add_total_emissions = False

# for MeteoTest Swiss inventory, unit m, x is easterly, y is northly
input_path = "/input/CH_EMISSIONS/emiskat15/"

# input grid
input_grid = SwissGrid(
    name="swiss-2015",
    nx=1800,
    ny=1200,
    dx=200,
    dy=200,
    xmin=480000,
    ymin=60000,
)

# species and categories read from input files
species = ['bc','co','nh3','nmvoc','nox','pm10','pm25','so2']
categories = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m" ]

# mapping from input to output species (input is used for missing keys)
in2out_species = {
    'bc':    'BC',
    'co':    'CO',
    'nh3':   'NH3',
    'nmvoc': 'NMVOC',
    'nox':   'NOX',
    'pm10':  'PM10',
    'pm25':  'PM25',
    'so2':   'SO2'
}

# mapping from input to output categories (input is used for missing keys)
in2out_category = dict((c, c.upper()) for c in categories)

# Output variables are written in the following format using species and
# category after applying the mapping
varname_format = '{species}_{category}_ch'

# Online or offline emissions (offline emissions have grid with 2-cell boundary)
offline = False

# Europe domain (rotated pole coordinates)
xmin = -16.08
ymin =  -9.54
nx = 192
ny = 164

cosmo_grid = COSMOGrid(
    nx=nx,
    ny=ny,
    dx=0.12,
    dy=0.12,
    xmin=xmin,
    ymin=ymin,
    pollon=-170.0,
    pollat=43.0,
)

# output filename
output_path = os.path.join('oae-art-example', '{online}', 'swiss')
output_name = 'swiss-art.nc'

# resolution of shape file used for country mask
shpfile_resolution = "10m"

# number of processes used
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
