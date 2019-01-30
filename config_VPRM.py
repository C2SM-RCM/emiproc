# "constant" paths and values for TNO
#tnoCamsPath = "/store/empa/em05/haussaij/CHE/VPRM/final/gpp_2015010100.nc"
EU="1"
tnoCamsPath = "/store/empa/em05/haussaij/CHE/VPRM/20150101/vprm_fluxes_EU"+EU+"_GPP_2015010110.nc"
tnoMACCIIIPath = tnoCamsPath
tno_xmin = -30. 
tno_xmax = 60.
tno_ymin = 30.
tno_ymax = 72. 
tno_dx = 1000
tno_dy = 1000
#tno_lons = np.arange(tno_startlon,tno_endlon+tno_dlon,tno_dlon)
#tno_lats = np.arange(tno_startlat,tno_endlat+tno_dlat,tno_dlat)
#tno_nx = round((tno_endlon-tno_startlon)/tno_dlon) + 1.
#tno_ny = round((tno_endlat-tno_startlat)/tno_dlat) + 1.



#case specific parameters
species = ['CO2_GPP']#, 'PM10', 'CH4', 'SO2', 'NMVOC', 'NH3', 'NOx'] #among 'CO2', 'PM2.5', 'CO', 'PM10', 'CH4', 'SO2', 'NMVOC', 'NH3', 'NOx'

cat_kind="NFR"
# snap = ["A", "B", "C", "D", "E", 
#         "F1", "F2", "F3", "G", "H", 
#         "I", "J", "K", "L" ]
snap = ["F"]
#tno_cat_var = "emis_cat_code" # "emis_cat_shortsnap"
tno_snap = [ ]
year = 2015
gridname = 'Europe'
output_path ="./testdata/VPRM/EU"+EU+"/"
#invs = ['CH4_TNO','CO2_TNO','CO_TNO','NOx_TNO','Berlin']

offline=True

# Domain
#Berlin-coarse
dx = 0.05
dy = 0.05
pollon = -170.0
pollat = 43.0

if not offline:
    xmin = -17#-2*dx
    ymin = -11#-2*dy
    nx = 760#+4
    ny = 610#+4
else:
    xmin = -17-2*dx
    ymin = -11-2*dy
    nx = 760+4
    ny = 610+4

