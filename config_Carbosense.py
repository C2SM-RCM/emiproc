# "constant" paths and values for TNO
tnoCamsPath = "/input/TNOMACC/CO2/TNO_CAMS_CO2_emissions_"
tnoMACCIIIPath = "/input/TNOMACC/MACCIII/TNO_MACC_III_emissions_"
tno_xmin = -30.
tno_xmax = 60.
tno_ymin = 30.
tno_ymax = 72.
tno_dx = 1/8.
tno_dy = 1/16.

#case specific parameters
species = ['CO2', 'CH4', 'CO']
#species = ['CO2']

cat_kind="SNAP"
snap = [1,2,34,5,6,70,8,9,10] #70 corresponds to all 7*
tno_snap = [ 1,2,34,5,6,71,72,73,8,9,10]
year = 2018
gridname = 'CH_1km'

output_path ="./testdata"

offline=False

# Domain
# CH-1
dx = 0.01
dy = 0.01
pollon = -170.0
pollat = 43.0
xmin = -4.92
ymin = -3.18
nx = 900
ny = 600

if offline:
    xmin -= 2 * dx
    ymin -= 2 * dy
    nx += 4
    ny += 4

