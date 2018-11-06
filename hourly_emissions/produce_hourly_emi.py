import netCDF4 as nc
import numpy as np
import datetime as dt

path_emi = "../testdata/emis_2015_Berlin-coarse_64_74.nc"
path_org = "../testdata/hourly_emi_brd/CO2_CO_NOX_Berlin-coarse_2015010110.nc"
output_path = "./output/"
output_name = "CO2_CO_NOX_Berlin-coarse_"

dow = nc.Dataset("dayofweek.nc")
hod = nc.Dataset("hourofday.nc")
moy = nc.Dataset("monthofyear.nc")
ver  = nc.Dataset("vertical_profiles.nc")


# catlist = ['CO2_01_POINT','CO2_02_AREA','CO2_34_AREA','CO2_34_POINT','CO2_05_AREA','CO2_05_POINT','CO2_06_AREA','CO2_07_AREA','CO2_08_AREA','CO2_08_POINT','CO2_09_AREA','CO2_09_POINT','CO2_10_AREA']
# tplist = ['CO2_1','CO2_2','CO2_4','CO2_4','CO2_5','CO2_5','CO2_6','CO2_7','CO2_8','CO2_8','CO2_9','CO2_9','CO2_10']
# vplist = ['SNAP-1','SNAP-2','SNAP-4','SNAP-4','SNAP-5','SNAP-5','SNAP-6','SNAP-7','SNAP-8','SNAP-8','SNAP-9','SNAP-9','SNAP-10']


var_list = ["CO2_A_E","CO2_07_E"]
catlist = [['CO2_02_AREA','CO2_34_AREA','CO2_05_AREA','CO2_06_AREA','CO2_07_AREA','CO2_08_AREA','CO2_09_AREA','CO2_10_AREA'],
           ["CO2_07_AREA"]]
tplist = [['CO2_2','CO2_4','CO2_5','CO2_6','CO2_7','CO2_8','CO2_9','CO2_10'],
          ["CO2_7"]]
vplist = [['SNAP-2','SNAP-4','SNAP-5','SNAP-6','SNAP-7','SNAP-8','SNAP-9','SNAP-10'],
          ["SNAP-7"]]



levels = 7

apply_prof = True


month=0
with nc.Dataset(path_emi) as emi:
    for day in range(3,7):
        for hour in range(24):
            time = dt.datetime(year=2015,day=day-2,hour=hour,month=month+1)
            print(time.strftime("%D"))
            with nc.Dataset(output_path+output_name+time.strftime(format="%Y%m%d%H")+".nc","w") as of:
                of.createDimension("time")
                of.createDimension("rlon",74)
                of.createDimension("rlat",64)
                of.createDimension("bnds",2)
                of.createDimension("level",levels)
                with nc.Dataset(path_org) as inf:
                    for var in ["time","rlon","rlat","rotated_pole","level","level_bnds","CO2_A_E"]:
                        of.createVariable(var,inf[var].datatype,inf[var].dimensions)
                        of[var].setncatts(inf[var].__dict__)
                        if var in ["time","rlon","rlat"]:
                            of[var][:] = inf[var][:]

                        if var=="CO2_A_E":
                            of.createVariable("CO2_07_E","float32",("time","level","rlat","rlon"))
                            of["CO2_07_E"].setncatts(inf[var].__dict__)

                if levels==1:
                    of["level"][:] = [10]
                    of["level_bnds"][:] = [0,20]
                else:
                    of["level"][:] = ver["layer_mid"][:]
                    of["level_bnds"][:] = np.array([ver["layer_bot"][:],ver["layer_top"][:]])

                for i in range(64): #range(2,62): #
                    print(i)
                    for j in range(74): #range(2,72): 
                        country_id = emi["country_ids"][i,j]
                        country_index = np.where(moy["country"][:]==country_id)[0][0]
                        
                        for k in range(levels):
                            for v,var in enumerate(var_list):
                                oae_to_add = 0
                                for (cat,tp,vp) in zip(catlist[v],tplist[v],vplist[v]):
                                    if apply_prof :
                                        if levels==1:
                                            oae_to_add += emi[cat][i,j] * hod[tp][hour,country_index] * dow[tp][day,country_index] * moy[tp][month,country_index]#*ver[vp][k]
                                            # if tp=="CO2_2":
                                            #     print(i-1,j-1,country_index,
                                            #           hour,hod[tp][hour,country_index],
                                            #           day,dow[tp][day,country_index],
                                            #           month,moy[tp][month,country_index],
                                            #           hod[tp][hour,country_index] * dow[tp][day,country_index] * moy[tp][month,country_index])
                                            #     print(caca)
                                        else:
                                            oae_to_add += emi[cat][i,j] * hod[tp][hour,country_index] * dow[tp][day,country_index] * moy[tp][month,country_index]*ver[vp][k]
                                    else:
                                        oae_to_add += emi[cat][i,j]

                                of[var][0,k,i,j] = oae_to_add #6-k ?
                                #of["CO2_07_E"][0,k,i,j] = emi["CO2_07_AREA"][i,j]



