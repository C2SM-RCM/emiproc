# for MeteoTest Swiss inventory, unit m, x is easterly, y is northly
origin = 'meteotest'
input_path = "/input/CH_EMISSIONS/emiskat15/"
ch_xll = 480000 
ch_yn = 1200
ch_yll = 60000
ch_xn = 1800 
ch_cell = 200
nodata_value = -9999



species = ['CO']

cat_kind="NFR"
#output cat
gnfr = [ "A", "B", "C", "D", "E", "F", 
        "G", "H", "I", "J", "K", "L", "M" ]
#input cat
ch_cat = [ "A", "B", "C", "D", "E", "F",
        "G", "H", "I", "J", "K", "L", "M" ]

year = 2018
gridname = 'CH_1km'

output_path ="./testdata/"

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

