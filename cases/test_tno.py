import time

from epro.grids import COSMOGrid, TNOGrid

# TNO inventory
inventory = 'TNO'

tnofile = "/input/TNOMACC/TNO_GHGco/TNO_6x6_GHGco_v1_1/TNO_GHGco_v1_1_year2015.nc"
input_grid = TNOGrid(tnofile)

cat_kind = "NFR"

tno_cat = [
    "A",
#    "B",
#    "C",
#    "D",
#    "E",
#    "F1",
#    "F2",
#    "F3",
#    "G",
#    "H",
#    "I",
#    "J",
#    "K",
#    "L",
]

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

offline = False
if offline:
    cosmo_grid.xmin -= 2 * cosmo_grid.dx
    cosmo_grid.ymin -= 2 * cosmo_grid.dy
    cosmo_grid.nx += 4
    cosmo_grid.ny += 4


species = ["co2_ff"] #,"co2_bf"]
output_cat = tno_cat

output_path = "."
output_name = "tno.nc"

shpfile_resolution = "110m"

# number of processes computing the mapping inventory->COSMO-grid
nprocs = 18

nc_metadata = {
    "DESCRIPTION": "Gridded annual emissions",
    "DATAORIGIN": "TNO",
    "CREATOR": "Jean-Matthieu Haussaire",
    "EMAIL": "jean-matthieu.haussaire@empa.ch",
    "AFFILIATION": "Empa Duebendorf, Switzerland",
    "DATE CREATED": time.ctime(time.time()),
}
