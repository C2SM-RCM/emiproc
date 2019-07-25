
from epro.grids import COSMOGrid, TNOGrid

inv_1 = './testdata/oae_paper/online/tno.nc'
inv_name_1 = 'TNO'
inv_2 = './testdata/oae_paper/online/emis_2018_carbocount_CO2_FLEXPART_main.nc'
inv_name_2 = 'Carbocount'

inv_out = './All_emissions.nc'


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

nc_metadata = {
    "DESCRIPTION": "Gridded annual emissions",
    "DATAORIGIN": "TNO and carbocount-CH",
    "CREATOR": "Jean-Matthieu Haussaire",
    "EMAIL": "jean-matthieu.haussaire@empa.ch",
    "AFFILIATION": "Empa Duebendorf, Switzerland",
    "DATE CREATED": time.ctime(time.time()),
}
