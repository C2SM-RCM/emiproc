
import time
from emiproc.grids import COSMOGrid, TNOGrid, ICONGrid



inv_1 = 'oae-art-example/{online}/tno/tno-art.nc'
inv_name_1 = ''

inv_2 = 'oae-art-example/{online}/swiss/swiss-art.nc'
inv_name_2 = ''

inv_out = 'oae-art-example/{online}/emis_2015_d1.nc'


# Output grid is European domain (rotated pole coordinates)
output_grid = COSMOGrid(
    nx=192,
    ny=164,
    dx=0.12,
    dy=0.12,
    xmin=-16.08,
    ymin=-9.54,
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
