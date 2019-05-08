# "constant" paths and values for TNO
tnoCamsPath = "/input/TNOMACC/TNO_GHGco/TNO_6x6_GHGco_v1_1/TNO_GHGco_v1_1_year2015.nc"
tnoMACCIIIPath = tnoCamsPath
tno_xmin = -30.
tno_xmax = 60.
tno_ymin = 30.
tno_ymax = 72.
tno_dx = 1/10.
tno_dy = 1/20.

#case specific parameters
species = ['CO2']

cat_kind="NFR"                                                                 

snap = [ "A", "B", "C", "D", "E", "F",                                         
         "G", "H", "I", "J", "K", "L" ]                                         

tno_snap = [ "A", "B", "C", "D", "E", "F1", "F2", "F3",                          
             "G", "H", "I", "J", "K", "L" ]

year = 2018
gridname = 'tno_CO2_FLEXPART_Nest'
output_path ="./testdata"

offline=False

# FLEXPART Nest
dx = 0.02
dy = 0.015
pollon = 180.0 # non-rotated grid
pollat = 90.0 # non-rotated grid 
xmin = 4.97
ymin = 45.4875
nx = 305
ny = 205

if offline:
    xmin -= 2 * dx
    ymin -= 2 * dy
    nx += 4
    ny += 4

