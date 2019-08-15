
import time
import os
from epro.grids import COSMOGrid, TNOGrid

inv_1 = os.path.join('outputs', '{online}', 'tno.nc')
inv_name_1 = '_TNO'
inv_2 = os.path.join('outputs', '{online}', 'carbocount.nc')
inv_name_2 = '_Carbocount'

inv_out = os.path.join('outputs', '{online}', 'all_emissions.nc')


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

nc_metadata = {
    "DESCRIPTION": "Gridded annual emissions",
    "DATAORIGIN": "TNO and carbocount-CH",
    "CREATOR": "Jean-Matthieu Haussaire",
    "EMAIL": "jean-matthieu.haussaire@empa.ch",
    "AFFILIATION": "Empa Duebendorf, Switzerland",
    "DATE CREATED": time.ctime(time.time()),
}
