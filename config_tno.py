import time

from grids import COSMOGrid, TNOGrid

# TNO inventory
tnofile = "/input/TNOMACC/TNO_GHGco/Future_years_emissions/TNO_GHGco_v1_1_CIRCE_BAU_year2030.nc"
tno_grid = TNOGrid(tnofile)

cat_kind = "NFR"

tno_cat = [
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

# COSMO domain
cosmo_grid = COSMOGrid(
    nx=760,
    ny=610,
    dx=0.05,
    dy=0.05,
    xmin=-17.0,
    ymin=-11.0,
    pollon=-170.0,
    pollat=43.0,
)

offline = False
if offline:
    cosmo_grid.xmin -= 2 * cosmo_grid.dx
    cosmo_grid.ymin -= 2 * cosmo_grid.dy
    cosmo_grid.nx += 4
    cosmo_grid.ny += 4


species = ["co2_ff"]
output_cat = ["A", "B"]

output_path = "./testdata/oae_paper/"
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
