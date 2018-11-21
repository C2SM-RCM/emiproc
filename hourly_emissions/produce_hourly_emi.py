import netCDF4 as nc
import numpy as np
import datetime as dt
import time as tm

def extract_to_grid(grid_to_index, country_vals):
    """Extract the values from each country_vals [1D] on gridpoints given
    by grid_to_index (2D)

    Parameters
    ----------
    grid_to_index : np.array(dtype=int)
        grid_to_index has at every gridpoint an integer which serves as an
        index into country_vals.
        grid_to_index.shape == (x, y)
        grid_to_index[i,j] < n_k forall i < x, j < y, k < len(country_vals)
    *country_vals : list(np.array(dtype=float))
        country_vals[i].shape == (n_i,)

    Returns
    -------
    tuple(np.array(dtype=float))
        Tuple of np.arrays, each of them the containing at (i,j) the value
        in the corresponding country_vals-array indicated by the grid_to_index-
        array.

        >>> res1, res2 = extract_to_grid(x, y, z)
        >>> y.dtype == res1.dtype
        True
        >>> x.shape == res2.shape
        >>> res1[i,j] == y[x[i,j]]
        True
        >>> res2[i,j] == z[x[i,j]]
        True
    """
    res = np.zeros_like(grid_to_index)
    for i in range(grid_to_index.shape[0]):
        for j in range(grid_to_index.shape[1]):
            res[i,j] = country_vals[grid_to_index[i,j]]
    return res


#################
# Berlin testcase
# path_emi = "../testdata/emis_2015_Berlin-coarse_64_74.nc"
# path_org = "../testdata/hourly_emi_brd/CO2_CO_NOX_Berlin-coarse_2015010110.nc"
# output_path = "./output/"
# output_name = "CO2_CO_NOX_Berlin-coarse_"
# prof_path = "./input_profiles/"
# rlon = 74
# rlat = 64
# var_list = ["CO2_A_E","CO2_07_E"]
# catlist = [['CO2_02_AREA','CO2_34_AREA','CO2_05_AREA','CO2_06_AREA','CO2_07_AREA','CO2_08_AREA','CO2_09_AREA','CO2_10_AREA'],
#            ["CO2_07_AREA"]]
# tplist = [['CO2_2','CO2_4','CO2_5','CO2_6','CO2_7','CO2_8','CO2_9','CO2_10'],
#           ["CO2_7"]]
# vplist = [['SNAP-2','SNAP-4','SNAP-5','SNAP-6','SNAP-7','SNAP-8','SNAP-9','SNAP-10'],
#           ["SNAP-7"]]
################

##########
# CHE
#path_emi = "../testdata/CHE_TNO_offline/emis_2015_Europe.nc"
path_emi = "../testdata/CHE_TNO_offline/emis_2015_Europe.nc"
path_org = "../testdata/hourly_emi_brd/CO2_CO_NOX_Berlin-coarse_2015010110.nc"
output_path = "./output_CHE/"
output_name = "Europe_CHE_"
prof_path = "./input_profiles_CHE/"
rlon = 760+4
rlat = 610+4
# TESTING
# rlon = 10
# rlat = 10

var_list= []
for (s,nfr) in [(s,nfr) for s in ["CO2","CO","CH4"] for nfr in ["A","B","C","F","O","ALL"]]:
    var_list.append(s+"_"+nfr+"_E") #["CO2_ALL_E","CO2_A_E"]
catlist= [
    ["CO2_A_AREA","CO2_A_POINT"], # for CO2_A_E
    ["CO2_B_AREA","CO2_B_POINT"], # for CO2_B_E
    ["CO2_C_AREA"],               # for CO2_C_E
    ["CO2_F_AREA"],               # for CO2_F_E
    ["CO2_D_AREA","CO2_D_POINT",
     "CO2_E_AREA",
     "CO2_G_AREA",
     "CO2_H_AREA","CO2_H_POINT", 
     "CO2_I_AREA",     
     "CO2_J_AREA","CO2_J_POINT",  # for CO2_O_E
 ],
    ["CO2_A_AREA","CO2_A_POINT",
     "CO2_B_AREA","CO2_B_POINT",
     "CO2_C_AREA",
     "CO2_F_AREA",
     "CO2_D_AREA","CO2_D_POINT",
     "CO2_E_AREA",
     "CO2_G_AREA",
     "CO2_H_AREA","CO2_H_POINT", 
     "CO2_I_AREA",     
     "CO2_J_AREA","CO2_J_POINT",  # for CO2_ALL_E
],
    ["CO_A_AREA","CO_A_POINT"],   # for CO_A_E
    ["CO_B_AREA","CO_B_POINT"],   # for CO_B_E
    ["CO_C_AREA"],                # for CO_C_E
    ["CO_F_AREA"],                # for CO_F_E
    ["CO_D_AREA","CO_D_POINT",
     "CO_E_AREA",
     "CO_G_AREA",
     "CO_H_AREA","CO_H_POINT", 
     "CO_I_AREA",     
     "CO_J_AREA","CO_J_POINT",    # for CO_O_E
],
    ["CO_A_AREA","CO_A_POINT",
     "CO_B_AREA","CO_B_POINT",
     "CO_C_AREA",
     "CO_F_AREA",
     "CO_D_AREA","CO_D_POINT",
     "CO_E_AREA",
     "CO_G_AREA",
     "CO_H_AREA","CO_H_POINT", 
     "CO_I_AREA",     
     "CO_J_AREA","CO_J_POINT",    # for CO_ALL_E
],
    ["CH4_A_AREA","CH4_A_POINT"], # for CH4_A_E
    ["CH4_B_AREA","CH4_B_POINT"], # for CH4_B_E
    ["CH4_C_AREA"],               # for CH4_C_E
    ["CH4_F_AREA"],               # for CH4_F_E
    ["CH4_D_AREA","CH4_D_POINT",
     "CH4_E_AREA",
     "CH4_G_AREA",
     "CH4_H_AREA","CH4_H_POINT", 
     "CH4_I_AREA",     
     "CH4_J_AREA","CH4_J_POINT",  # for CH4_O_E
],
    ["CH4_A_AREA","CH4_A_POINT",
     "CH4_B_AREA","CH4_B_POINT",
     "CH4_C_AREA",
     "CH4_F_AREA",
     "CH4_D_AREA","CH4_D_POINT",
     "CH4_E_AREA",
     "CH4_G_AREA",
     "CH4_H_AREA","CH4_H_POINT", 
     "CH4_I_AREA",     
     "CH4_J_AREA","CH4_J_POINT", # for CH4_ALL_E
 ]
]

tplist_1 = [
    ['GNFR_A','GNFR_A'], # for s_A_E
    ['GNFR_B','GNFR_B'], # for s_B_E
    ['GNFR_C'],          # for s_C_E
    ['GNFR_F'],          # for s_F_E
    ['GNFR_D','GNFR_D', 
     'GNFR_E',
     'GNFR_G',
     'GNFR_H','GNFR_H',
     'GNFR_I',
     'GNFR_J','GNFR_J',
     'GNFR_K',
     'GNFR_L',],         # for s_O_E
    ['GNFR_A','GNFR_A',
     'GNFR_B','GNFR_B',
     'GNFR_C',
     'GNFR_F',
     'GNFR_D','GNFR_D',
     'GNFR_E',
     'GNFR_G',
     'GNFR_H','GNFR_H',
     'GNFR_I',
     'GNFR_J','GNFR_J',
 ]          # for s_ALL_E
]
tplist= tplist_1+tplist_1+tplist_1

vplist = tplist

###########


levels = 7  # Removed if-statement in inner loop
# apply_prof = True  # Removed if-statement in inner loop

# catlist = ['CO2_01_POINT','CO2_02_AREA','CO2_34_AREA','CO2_34_POINT','CO2_05_AREA','CO2_05_POINT','CO2_06_AREA','CO2_07_AREA','CO2_08_AREA','CO2_08_POINT','CO2_09_AREA','CO2_09_POINT','CO2_10_AREA']
# tplist = ['CO2_1','CO2_2','CO2_4','CO2_4','CO2_5','CO2_5','CO2_6','CO2_7','CO2_8','CO2_8','CO2_9','CO2_9','CO2_10']
# vplist = ['SNAP-1','SNAP-2','SNAP-4','SNAP-4','SNAP-5','SNAP-5','SNAP-6','SNAP-7','SNAP-8','SNAP-8','SNAP-9','SNAP-9','SNAP-10']

dow = nc.Dataset(prof_path + "dayofweek.nc")
hod = nc.Dataset(prof_path + "hourofday.nc")
moy = nc.Dataset(prof_path + "monthofyear.nc")
ver  = nc.Dataset(prof_path + "vertical_profiles.nc")

month=0
with nc.Dataset(path_emi) as emi:
    # Mapping country_ids (from emi) to country-indices (from moy)
    # Only gives minor speedboost
    country_ids = np.array(
        [
            [
                np.where(moy["country"][:] == emi["country_ids"][i,j])[0][0]
                for j in range(rlon)
            ]
            for i in range(rlat)
        ],
        dtype = np.int16)
    print(type(country_ids))

    for day in range(3,7):
        for hour in range(24):
            time = dt.datetime(year=2015,
                               day=day-2,
                               hour=hour,
                               month=month+1)
            print(time.strftime("%D"))
            with nc.Dataset(output_path +
                            output_name +
                            time.strftime(format="%Y%m%d%H") +
                            ".nc","w") as of:
                of.createDimension("time")
                of.createDimension("rlon", rlon)
                of.createDimension("rlat", rlat)
                of.createDimension("bnds", 2)
                of.createDimension("level", levels)
                with nc.Dataset(path_org) as inf:
                    for var in ["time","rotated_pole","level","level_bnds"]:
                        of.createVariable(var,
                                          inf[var].datatype,
                                          inf[var].dimensions)
                        of[var].setncatts(inf[var].__dict__)
                        if var in ["time"]:
                            of[var][:] = inf[var][:]

                with nc.Dataset(path_emi) as inf:
                    for var in ["rlon","rlat"]:
                        of.createVariable(var,
                                          inf[var].datatype,
                                          inf[var].dimensions)
                        of[var].setncatts(inf[var].__dict__)

                for var in var_list:
                    of.createVariable(var,
                                      "float32",
                                      ("time","level","rlat","rlon"))

                of["level"][:] = ver["layer_mid"][:]
                of["level_bnds"][:] = np.array([ver["layer_bot"][:],
                                                ver["layer_top"][:]])

                start = tm.time()
                for v, var in enumerate(var_list):
                    oae_vals = np.zeros((rlat, rlon, levels))
                    for (cat, tp, vp) in zip(catlist[v],
                                             tplist[v],
                                             vplist[v]):

                        # Adding dimension to allow numpy to broadcast
                        emi_mat = np.expand_dims(emi[cat][:rlat, :rlon], -1)
                        hod_mat = np.expand_dims(
                            extract_to_grid(grid_to_index, hod[tp][hour, :]), -1)
                        dow_mat = np.expand_dims(
                            extract_to_grid(grid_to_index, dow[tp][day, :]), -1)
                        moy_mat = add_trail_dim(
                            extract_to_grid(grid_to_index, moy[tp][hour, :]), -1)
                        ver_mat = ver[vp]

                        oae_vals += emi_mat * hod_mat * dow_mat * moy_mat * ver_mat

                    of[var][0,:] = oae_vals
                stop = tm.time()
                print("Processed {} datapoints in {}"
                      .format(rlon * rlat, stop-start))
