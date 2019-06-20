# TNO inventory
tnofile = "/input/TNOMACC/TNO_GHGco/Future_years_emissions/TNO_GHGco_v1_1_CIRCE_BAU_year2030.nc"

tno_xmin = -30.0
tno_xmax = 60.0
tno_ymin = 30.0
tno_ymax = 72.0
tno_dx = 1 / 10.0
tno_dy = 1 / 20.0

cat_kind = "NFR"

tno_cat = [
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

# COSMO domain
xmin = -17
ymin = -11
nx = 760
ny = 610
dx = 0.05
dy = 0.05
pollon = -170.0
pollat = 43.0

offline = False
if offline:
    xmin -= 2 * dx
    ymin -= 2 * dy
    nx += 4
    ny += 4


species = ["co2_ff"]
output_cat = ["A", "B"]

output_path = "./testdata/oae_paper/"
output_name = "tno.nc"
