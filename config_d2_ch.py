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
gridname = 'd2_ch'
#output_path ="./testdata/d2_offline/ch/"
output_path ="./testdata/d2_online/ch/"

#offline=True
offline=False

# Domain
#Europe domain, rotated pole coordinate
dx = 0.02
dy = 0.02
pollon = -170.0
pollat = 43.0

if not offline:
    xmin = -3.48#-2*dx
    ymin = -1.74#-2*dy
    nx = 220#+4
    ny = 142#+4
else:
    xmin = -3.48-2*dx
    ymin = -1.74-2*dy
    nx = 220+4
    ny = 142+4

