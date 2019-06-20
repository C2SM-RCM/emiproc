# "constant" paths and values for TNO, regular lat/lon
# for MeteoTest Swiss inventory, use calculated regular domain in the code
tnoCamsPath = (
    "/input/TNOMACC/CAMS-REG-AP_v2_2/CAMS-REG-AP_v2_2_1_emissions_year2015.nc"
)
tnoMACCIIIPath = tnoCamsPath
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
gridname = "d2_tno"
output_path = "./testdata/d2_offline/tno/"
# output_path ="./testdata/d2_online/tno/"

offline = True
# offline=False

# Domain
# Europe domain, rotated pole coordinate
dx = 0.02
dy = 0.02
pollon = -170.0
pollat = 43.0

if not offline:
    xmin = -3.48  # -2*dx
    ymin = -1.74  # -2*dy
    nx = 220  # +4
    ny = 142  # +4
else:
    xmin = -3.48 - 2 * dx
    ymin = -1.74 - 2 * dy
    nx = 220 + 4
    ny = 142 + 4
