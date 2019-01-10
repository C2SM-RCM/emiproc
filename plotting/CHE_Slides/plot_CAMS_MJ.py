import xarray
import amrs.models.cosmo
import amrs.vis
import amrs.misc.misc
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs

data = xarray.open_dataset("/store/empa/em05/dbrunner/che/icbc/cams_gvri_2015010612.nc")
#data = xarray.open_dataset("/project/s862/CHE/CHE_output_todel/CHE_Europe_output/2015010100_0_24/cosmo_output/lffd2015010106.nc")

startlon = -17
stoplon = 21
startlat = -11
stoplat = 19.5
pollon, pollat = (-170,43) #amrs.misc.misc.read_rotpole(filename)

domain = amrs.models.cosmo.Domain('Domain', startlon, startlat, stoplon, stoplat, pollon=pollon, pollat=pollat)

# Transform the lon,lat of the ECMWF grid into rotated-pole coordinates
all_points = np.array([(x,y) for x in data.longitude for y in data.latitude])
transform = ccrs.RotatedPole(pole_longitude=pollon, pole_latitude=pollat)
grid_points= transform.transform_points(ccrs.PlateCarree(),all_points[:,0],all_points[:,1])
cosmo_xlocs = grid_points[:,0]
cosmo_xlocs.shape = (len(data.longitude),len(data.latitude))
cosmo_ylocs = grid_points[:,1]
cosmo_ylocs.shape = (len(data.longitude),len(data.latitude))


fig = amrs.vis.make_field_map(cosmo_xlocs, cosmo_ylocs, data.co2[0,-1,:,:], domain=domain)

plt.show()
