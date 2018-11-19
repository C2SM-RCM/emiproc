# "constant" paths and values for TNO
tnoCamsPath = "/project/hjm/CHE/Anthropogenic_emissions/TNO_GHGco_v1_0_year2015.nc"
tnoMACCIIIPath = tnoCamsPath
tno_xmin = -30. 
tno_xmax = 60.
tno_ymin = 30.
tno_ymax = 72. 
tno_dx = 1/10.
tno_dy = 1/20.
#tno_lons = np.arange(tno_startlon,tno_endlon+tno_dlon,tno_dlon)
#tno_lats = np.arange(tno_startlat,tno_endlat+tno_dlat,tno_dlat)
#tno_nx = round((tno_endlon-tno_startlon)/tno_dlon) + 1.
#tno_ny = round((tno_endlat-tno_startlat)/tno_dlat) + 1.



#case specific parameters
species = ['CO2','CH4','CO','NOX']#, 'PM10', 'CH4', 'SO2', 'NMVOC', 'NH3', 'NOx'] #among 'CO2', 'PM2.5', 'CO', 'PM10', 'CH4', 'SO2', 'NMVOC', 'NH3', 'NOx'

cat_kind="NFR"
# snap = ["A", "B", "C", "D", "E", 
#         "F1", "F2", "F3", "G", "H", 
#         "I", "J", "K", "L" ]
snap = [ "A", "B", "C", "D", "E", "F", 
        "G", "H", "I", "J", "K", "L" ]
#tno_cat_var = "emis_cat_code" # "emis_cat_shortsnap"
tno_snap = [ "A", "B", "C", "D", "E", "F1","F2","F3",
        "G", "H", "I", "J", "K", "L" ]
year = 2015
gridname = 'Europe'
output_path ="./testdata/CHE_TNO_offline/"
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

