
import time
from epro.grids import COSMOGrid, TNOGrid


inv_1 = 'oae-art-example/online/tno/tno-art.nc'
inv_name_1 = ''

inv_2 = 'oae-art-example/online/swiss/swiss-art.nc'
inv_name_2 = ''

inv_out = 'oae-art-example/online/emis_2015_d1.nc'


# Output grid is European domain (rotated pole coordinates)
xmin = -16.08
ymin =  -9.54
nx = 192
ny = 164

offline = False

if offline:
    xmin -= 2 * dx
    ymin -= 2 * dy
    nx += 4
    ny += 4

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


# metadata added as global attributes to netCDF output file
nc_metadata = {
    "DESCRIPTION": "Gridded annual emissions",
    "DATAORIGIN": "TNO-CAMS",
    "CREATOR": "Qing Mu and Gerrit Kuhlmann",
    "EMAIL": "gerrit.kuhlmann@empa.ch",
    "AFFILIATION": "Empa Duebendorf, Switzerland",
    "DATE CREATED": time.ctime(time.time()),
}
