from make_online_emissions import interpolate_tno_to_cosmo_grid,initialize_output,gridbox_area
from netCDF4 import Dataset
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import importlib
import config_VPRM as cfg
from datetime import datetime,timedelta
from multiprocessing import Pool


input_path = cfg.tnoCamsPath

def doit(i,interp_tot,cosmo_area):
    inidate = datetime(2015,1,i)
    print("taking care of ",inidate.strftime("%Y%m%d"))
    for s,sname in zip(["gpp","ra"],["CO2_GPP_F","CO2_RA_F"]):
        for h in range(24):
            time_ = inidate + timedelta(hours=h)
            time_str = time_.strftime("%Y%m%d%H")
            print("   ",s,time_str)

            output_path = "/scratch/snx3000/haussaij/VPRM/int2lm/"+s+"_"+time_str+".nc"

            data_path = "/scratch/snx3000/haussaij/VPRM/"+ inidate.strftime("%Y%m%d")+"/"+s+"_"+time_str+".nc"
            data = Dataset(data_path)[sname][:]
            out_var_area = np.zeros((cfg.ny,cfg.nx))

            for lon in range(data.shape[1]):
                for lat in range(data.shape[0]):
                    for (x,y,r) in interp_tot[lat,lon]:
                        #vprm_dat is in umol/m^2/s. Each cell is exactly 1km^2
                        out_var_area[y,x] += data[lat,lon]*r*1000*1000 #now out_var_area is in umol

            with Dataset(output_path,"w") as outf:
                initialize_output(outf,cfg)
                print("I need to add time. Are you sure we're going on?")
                s = input("hein ?")

                """convert unit from umol.cell-1.s-1 to kg.m-2.s-1"""
                """calculate the areas (m^^2) of the COSMO grid"""
                m_co2 = 44.01

                out_var_area *= cosmo_area.T*1e-9*m_co2
                out_var_name = sname
                outf.createVariable(out_var_name,float,("rlat","rlon"))
                outf[out_var_name].units = "kg m-2 s-1"
                outf[out_var_name][:] = out_var_area



def get_interp_tot():
    for n,i in enumerate([6,2,4,1,3,5]):#[1,3,5,6,2,4]): # Weird order, following the README of Julia
        input_path = "/scratch/snx3000/haussaij/VPRM/20150101/vprm_fluxes_EU"+str(i)+"_GPP_2015010110.nc"
        with Dataset(input_path) as inf:
            interpolation = get_interpolation(cfg,inf,filename="EU"+str(i)+"_mapping.npy")
      
            if n==0:
                x,y = inf["lat"][:].shape
                interp_tot = np.empty((x*2,y*3),dtype=object)

            if n<3:
                interp_tot[:x,n*y:(n+1)*y] = interpolation.T
            else:
                interp_tot[x:,(n-3)*y:(n-2)*y] = interpolation.T
    return interp_tot


def main(cfg_path):
    """ The main script for processing the VPRM inventory. 
    Takes a configuration file as input"""

    """Load the configuration file"""
    cfg = load_cfg(cfg_path)

    print("Combining the interpolation matrices")
    interp_tot = get_interp_tot()
    cosmo_area = 1./gridbox_area(cfg)

    with Pool(8) as pool:
        pool.starmap(doit,[(i,interp_tot,cosmo_area) for i in range(2,10)])
        

if __name__=="__main__":
    main("./config_VPRM")
    
