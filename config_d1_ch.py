# for MeteoTest Swiss inventory, unit m, x is easterly, y is northly
input_path = "/input/CH_EMISSIONS/emiskat15/"
ch_xll = 480000 
ch_yn = 1200
ch_yll = 60000
ch_xn = 1800 
ch_cell = 200
nodata_value = -9999

species = ['BC','CO','NH3','NMVOC','NOX','PM10','PM25','SO2']#, 'PM10', 'CH4', 'SO2', 'NMVOC', 'NH3', 'NOx'] #among 'CO2', 'PM2.5', 'CO', 'PM10', 'CH4', 'SO2', 'NMVOC', 'NH3', 'NOx'

cat_kind="NFR"
#output cat
gnfr = [ "A", "B", "C", "D", "E", "F", 
        "G", "H", "I", "J", "K", "L", "M" ]
#input cat
ch_cat = [ "A", "B", "C", "D", "E", "F",
        "G", "H", "I", "J", "K", "L", "M" ]
year = 2015
gridname = 'd1_ch'
#output_path ="./testdata/d1_offline/ch/"
output_path ="./testdata/d1_online/ch/"

#offline=True
offline=False

# Domain
#Europe domain, rotated pole coordinate
dx = 0.12
dy = 0.12
pollon = -170.0
pollat = 43.0

if not offline:
    xmin = -16.08#-2*dx
    ymin = -9.54#-2*dy
    nx = 192#+4
    ny = 164#+4
else:
    xmin = -16.08-2*dx
    ymin = -9.54-2*dy
    nx = 192+4
    ny = 164+4

