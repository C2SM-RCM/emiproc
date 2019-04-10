# For CarboCount Swiss inventory, unit m, x is easterly, y is northly
input_path = "/input/CH_EMISSIONS/CarboCountCO2/summengrids/"
ch_xn = 760 
ch_yn = 500
ch_xll = 470000 
ch_yll = 60000
ch_cell = 500
nodata_value = -9999

origin = 'carbocount'
gridname = origin + '_CO2_1km'

species = ['CO2']

ch_cat = ['']






year = 2018

output_path ="./testdata"

offline=False

# Carbosense COSMO-1 Domain
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

