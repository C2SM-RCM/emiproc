import xarray
import amrs.models.cosmo
import amrs.vis
import amrs.misc.misc
import matplotlib.pyplot as plt

#data = xarray.open_dataset("/store/empa/em05/dbrunner/che/icbc/cams_gvri_2015010612.nc")
data = xarray.open_dataset("/project/s862/CHE/CHE_output_todel/CHE_Europe_output/2015010100_0_24/cosmo_output/lffd2015010106.nc")

startlon = -17
stoplon = 21
startlat = -11
stoplat = 19.5
pollon, pollat = (-170,43) #amrs.misc.misc.read_rotpole(filename)

domain = amrs.models.cosmo.Domain('Domain', startlon, startlat, stoplon, stoplat, pollon=pollon, pollat=pollat)
fig = amrs.vis.make_field_map(data.rlon, data.rlat, data.CO2_ALL[0,-1,:,:], domain=domain)

plt.show()
