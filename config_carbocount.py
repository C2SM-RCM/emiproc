import time

from grids import COSMOGrid, SwissGrid

# for Swiss inventory, unit m, x is easterly, y is northly
input_path = "/input/CH_EMISSIONS/CarboCountCO2/einzelgrids/"
swiss_grid = SwissGrid(
    name="carbocount",
    nx=760,
    ny=500,
    dx=500,
    dy=500,
    xmin=470_000,
    ymin=60_000,
    I_HAVE_UNDERSTOOD_THE_CONVENTION_SWITCH_MADE_IN_THIS_METHOD=True,
)

gridname = swiss_grid.name + "_CO2_FLEXPART_main"

species = ["CO2"]

ch_cat = [
    "bm",
    "cf",
    "df",
    "hf",
    "hk",
    "if",
    "ka",
    "ki",
    "ks",
    "kv",
    "la",
    "lf",
    "lw",
    "mi",
    "mt",
    "nf",
    "pf",
    "pq",
    "rf",
    "vk",
    "zp",
    "zv",
]

mapping = {
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

year = 2018

output_path = "./testdata/flexpart"

# COSMO domain
cosmo_grid = COSMOGrid(
    nx=207,
    ny=179,
    dx=0.16,
    dy=0.12,
    xmin=-11.92,
    ymin=36.06,
    pollon=180.0,
    pollat=90.0,
)

offline = False
if offline:
    cosmo_grid.xmin -= 2 * cosmo_grid.dx
    cosmo_grid.ymin -= 2 * cosmo_grid.dy
    cosmo_grid.nx += 4
    cosmo_grid.ny += 4

shpfile_resolution = "110m"
nprocs = 18

nc_metadata = {
    "DESCRIPTION": "Gridded annual emissions",
    "DATAORIGIN": "carbocount-CH",
    "CREATOR": "Michael Jaehn",
    "EMAIL": "michael.jaehn@empa.ch",
    "AFFILIATION": "Empa Duebendorf, Switzerland",
    "DATE CREATED": time.ctime(time.time()),
}
