# "constant" paths and values for TNO, regular lat/lon
# for MeteoTest Swiss inventory, use calculated regular domain in the code
tnoCamsPath = (
    "/input/TNOMACC/CAMS-REG-AP_v2_2/CAMS-REG-AP_v2_2_1_emissions_year2015.nc"
)
tno_xmin = -30.0
tno_xmax = 60.0
tno_ymin = 30.0
tno_ymax = 72.0
tno_dx = 1 / 10.0
tno_dy = 1 / 20.0

species = [
    "CO",
    "NOX",
    "NMVOC",
    "SO2",
    "NH3",
    "PM10",
    "PM25",
]  # , 'PM10', 'CH4', 'SO2', 'NMVOC', 'NH3', 'NOx'] #among 'CO2', 'PM2.5', 'CO', 'PM10', 'CH4', 'SO2', 'NMVOC', 'NH3', 'NOx'

cat_kind = "NFR"
# output cat
snap = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
# input cat
tno_snap = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F1",
    "F2",
    "F3",
    "F4",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
]
year = 2015
gridname = "d1_tno"
# output_path ="./testdata/d1_offline/tno/"
output_path = "./testdata/d1_online/tno/"

# offline=True
offline = False

# Domain
# Europe domain, rotated pole coordinate
dx = 0.12
dy = 0.12
pollon = -170.0
pollat = 43.0

if not offline:
    xmin = -16.08  # -2*dx
    ymin = -9.54  # -2*dy
    nx = 192  # +4
    ny = 164  # +4
else:
    xmin = -16.08 - 2 * dx
    ymin = -9.54 - 2 * dy
    nx = 192 + 4
    ny = 164 + 4
