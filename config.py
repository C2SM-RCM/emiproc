# "constant" paths and values for TNO
tnoCamsPath = "/input/TNOMACC/CO2/TNO_CAMS_CO2_emissions_"
tnoMACCIIIPath = "/input/TNOMACC/MACCIII/TNO_MACC_III_emissions_"
tno_xmin = -30. 
tno_xmax = 60.
tno_ymin = 30.
tno_ymax = 72. 
tno_dx = 1/8.
tno_dy = 1/16.
#tno_lons = np.arange(tno_startlon,tno_endlon+tno_dlon,tno_dlon)
#tno_lats = np.arange(tno_startlat,tno_endlat+tno_dlat,tno_dlat)
#tno_nx = round((tno_endlon-tno_startlon)/tno_dlon) + 1.
#tno_ny = round((tno_endlat-tno_startlat)/tno_dlat) + 1.



#case specific parameters
species = ['CO2']#, 'PM10', 'CH4', 'SO2', 'NMVOC', 'NH3', 'NOx'] #among 'CO2', 'PM2.5', 'CO', 'PM10', 'CH4', 'SO2', 'NMVOC', 'NH3', 'NOx'

snap = [1,2,34,5,6,70,8,9,10] #70 corresponds to all 7*
maccversion = 'III'           # use this version for TNO/MACC data
year = 2015
gridname = 'Berlin-coarse'
output_path ="./testdata/"
#invs = ['CH4_TNO','CO2_TNO','CO_TNO','NOx_TNO','Berlin']


# Domain
#Berlin-coarse
dx = 0.1
dy = 0.1
xmin = -1.4-2*dx
ymin = 2.5-2*dy
nx = 70+4
ny = 60+4
pollon = -170.0
pollat = 43.0

