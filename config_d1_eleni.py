tnoCamsPath = "/input/TNOMACC/MACCIII/TNO_MACC_III_emissions_2011.nc"
tno_xmin = -30. 
tno_xmax = 60.
tno_ymin = 30.
tno_ymax = 72. 
tno_dx = 1/8.
tno_dy = 1/16.

species = ['CO','NOX','NMVOC','SO2','NH3','PM10','PM25']#, 'PM10', 'CH4', 'SO2', 'NMVOC', 'NH3', 'NOx'] #among 'CO2', 'PM2.5', 'CO', 'PM10', 'CH4', 'SO2', 'NMVOC', 'NH3', 'NOx'

cat_kind="SNAP"
#output cat
snap = [ 1, 2, 34, 5, 6, 71, 72, 73, 74, 75, 8, 9, 10]
#input cat
tno_snap = [1, 2, 34, 5, 6, 71, 72, 73, 74, 75, 8, 9, 10]
year = 2013
gridname = 'd1_eleni'
output_path ="./testdata/eleni/d1_online/"
#output_path ="./testdata/d1_online/tno/"

#offline=True
offline=False

# Domain
#Europe domain, rotated pole coordinate
dx = 0.025
dy = 0.025
pollon = -156.5
pollat = 52.3

if not offline:
    xmin = -4.8#-2*dx
    ymin = -4.0#-2*dy
    nx = 384#+4
    ny = 330#+4
else:
    xmin = -4.8-2*dx
    ymin = -4.0-2*dy
    nx = 384+4
    ny = 330+4

