# "constant" paths and values for TNO
input_path = "/input/EDGAR/v432_FT_CHE/"
edgar_xmin = -30.0
edgar_xmax = 60.0
edgar_ymin = 30.0
edgar_ymax = 69.0
edgar_dx = 0.1
edgar_dy = 0.1
edgar_nx = (edgar_xmax - edgar_xmin) / edgar_dx
edgar_ny = (edgar_ymax - edgar_ymin) / edgar_dy

# case specific parameters
species = "CO2"

gnfr = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
edgar_cat = [
    "AGS",
    "CHE",
    "ENE",
    "FFF",
    "IND",
    "IRO",
    "NEU",
    "NFE",
    "NMM",
    "PRO",
    "PRU_SOL",
    "RCO",
    "REF_TRF",
    "SWD_INC",
    "TNR_Aviation_CDS",
    "TNR_Aviation_CRS",
    "TNR_Aviation_LTO",
    "TNR_Other",
    "TNR_Ship",
    "TRO",
]
year = 2015
gridname = "Europe"
output_path = "./testdata/EDGAR/"

offline = True

# Domain
# CHE_Europe domain
dx = 0.05
dy = 0.05
pollon = -170.0
pollat = 43.0

if not offline:
    xmin = -17  # -2*dx
    ymin = -11  # -2*dy
    nx = 760  # +4
    ny = 610  # +4
else:
    xmin = -17 - 2 * dx
    ymin = -11 - 2 * dy
    nx = 760 + 4
    ny = 610 + 4
