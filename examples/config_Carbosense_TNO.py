# "constant" paths and values for TNO
tnoCamsPath = (
    "/input/TNOMACC/TNO_GHGco/TNO_6x6_GHGco_v1_1/TNO_GHGco_v1_1_year2015.nc"
)
tnoMACCIIIPath = tnoCamsPath
tno_xmin = -30.0
tno_xmax = 60.0
tno_ymin = 30.0
tno_ymax = 72.0
tno_dx = 1 / 10.0
tno_dy = 1 / 20.0

# case specific parameters
species = ["CO2", "CH4", "CO"]

cat_kind = "NFR"

snap = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

tno_snap = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F1",
    "F2",
    "F3",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
]

year = 2018
gridname = "tno_CO2_CO_CH4_1km_CH_only"
output_path = "./testdata"

offline = False

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