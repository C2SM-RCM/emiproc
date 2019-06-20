# "constant" paths and values for TNO
vprm_path = "/scratch/snx3000/haussaij/VPRM/single_files/vprm_fluxes_EU%s_GPP_2015010110.nc"
tno_xmin = -30.0
tno_xmax = 60.0
tno_ymin = 30.0
tno_ymax = 72.0
tno_dx = 1000
tno_dy = 1000

output_path = "./testdata/VPRM/output/"

offline = True

# Domain
# Berlin-coarse
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
