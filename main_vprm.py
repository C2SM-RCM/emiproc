from make_online_emissions import *

from netCDF4 import Dataset
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import importlib
import config_VPRM_5km as cfg
from datetime import datetime,timedelta
from multiprocessing import Pool



def doit(i,interp_tot,cosmo_area):
    inidate = datetime(2015,1,i)
    print("taking care of ",inidate.strftime("%Y%m%d"))
    for s,sname in zip(["gpp","ra"],["CO2_GPP_F","CO2_RA_F"]):
        for h in range(24):
            time_ = inidate + timedelta(hours=h)
            time_str = time_.strftime("%Y%m%d%H")
            print("   ",s,time_str)

            #output_path = "/scratch/snx3000/haussaij/VPRM/int2lm/"+s+"_"+time_str+".nc"
            filename_out = s+"_"+time_str+".nc"
            output_path = "/scratch/snx3000/haussaij/VPRM/output/"+filename_out
            data_path = "/scratch/snx3000/haussaij/VPRM/input/"+ inidate.strftime("%Y%m%d")+"/"+s+"_"+time_str+".nc"
            data = Dataset(data_path)[sname][:]
            out_var_area = np.zeros((cfg.ny,cfg.nx))

            for lon in range(data.shape[1]):
                for lat in range(data.shape[0]):
                    for (x,y,r) in interp_tot[lat,lon]:
                        #vprm_dat is in umol/m^2/s. Each cell is exactly 1km^2
                        out_var_area[y,x] += data[lat,lon]*r*cfg.tno_dx*cfg.tno_dy #now out_var_area is in umol.cell-1.s-1

            with Dataset(output_path,"w") as outf:
                prepare_output_file(cfg,outf)
                # Add the time variable
                example_file = "/store/empa/em05/haussaij/CHE/input_che/vprm_old/vprm_smartcarb/processed/"+filename_out
                with Dataset(example_file) as ex :
                    outf.createDimension("time")
                    outf.createVariable(varname="time",
                                           datatype=ex["time"].datatype,
                                           dimensions=ex["time"].dimensions)
                    outf["time"].setncatts(ex["time"].__dict__)


                """convert unit from umol.cell-1.s-1 to kg.m-2.s-1"""
                """calculate the areas (m^^2) of the COSMO grid"""
                m_co2 = 44.01

                out_var_area *= cosmo_area.T*1e-9*m_co2
                out_var_name = sname
                outf.createVariable(out_var_name,float,("rlat","rlon"))
                outf[out_var_name].units = "kg m-2 s-1"
                outf[out_var_name][:] = out_var_area



def get_interp_tot(cfg):
    for n,i in enumerate([6,2,4,1,3,5]):#[1,3,5,6,2,4]): # Weird order, following the README of Julia
        input_path = cfg.vprm_path %str(i) 
        with Dataset(input_path) as inf:
            interpolation = get_interpolation(cfg,inf,filename="EU"+str(i)+"_mapping.npy",inv_name="vprm")
      
            if n==0:
                x,y = inf["lat"][:].shape
                interp_tot = np.empty((x*2,y*3),dtype=object)

            if n<3:
                interp_tot[:x,n*y:(n+1)*y] = interpolation.T
            else:
                interp_tot[x:,(n-3)*y:(n-2)*y] = interpolation.T
    return interp_tot

def get_interp_5km(cfg):
    input_path = cfg.vprm_path
    with Dataset(input_path) as inf:
        interpolation = get_interpolation(cfg,inf,inv_name="vprm")
        
    return interpolation.T



def main(cfg_path):
    """ The main script for processing the VPRM inventory. 
    Takes a configuration file as input"""

    print("Combining the interpolation matrices")
    if "5km" in cfg_path:
        interp_tot = get_interp_5km(cfg)
    else:
        interp_tot = get_interp_tot(cfg)

    cosmo_area = 1./gridbox_area(cfg)

    with Pool(9) as pool:
        pool.starmap(doit,[(i,interp_tot,cosmo_area) for i in range(1,10)])
        

if __name__=="__main__":
    main("./config_VPRM_5km")
    
