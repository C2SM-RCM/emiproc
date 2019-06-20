# "constant" paths and values for TNO
#tnoCamsPath = "/project/hjm/CHE/TNO_Anthropogenic_emissions/v1_1_2018_12/TNO_6x6_GHGco_v1_1/TNO_GHGco_v1_1_year2015.nc"
tnoCamsPath = "/input/TNOMACC/TNO_GHGco/Future_years_emissions/TNO_GHGco_v1_1_CIRCE_BAU_year2030.nc"
tnoMACCIIIPath = tnoCamsPath
tno_xmin = -30. 
tno_xmax = 60.
tno_ymin = 30.
tno_ymax = 72. 
tno_dx = 1/10.
tno_dy = 1/20.

#case specific parameters
species = ['CO2']#,'CH4','CO','NOX','NMVOC']

cat_kind="NFR_BAU"
snap = [ "A", "B", "C", "D", "E", "F", 
        "G", "H", "I", "J", "K", "L" ]
# snap = [ "J", "L" ]
#tno_snap = [ "A", "B", "C", "D", "E", "F1","F2","F3",
#        "G", "H", "I", "J", "K", "L" ]
tno_snap = [ "A", "B", "C", "D", "E", "F1","F2","F3",
        "G", "H", "I", "J", "L" ]
year = 2030
gridname = 'Europe_BAU'
output_path ="./testdata/CHE_TNO_v1_1_2018_12/CHE_TNO_online/"

offline=False

# Domain
#CHE_Europe domain
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

