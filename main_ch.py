from make_online_emissions import *
from glob import glob
import numpy as np

def get_ch_emi(filename):
    """read in meteotest swiss inventory data
     
       output: emi_trans[lon][lat] (emission data per species per category)
    """
    no_data= -9999
    emi = np.loadtxt(filename, skiprows=6)
    
    emi_new = np.empty((np.shape(emi)[0],np.shape(emi)[1]))   # create the same dimention as emi
    for i in range(np.shape(emi)[0]):
        for j in range(np.shape(emi)[1]):            
            emi_new[i,j]=emi[np.shape(emi)[0]-1-i,j] 
            if emi_new[i,j]==no_data:
               emi_new[i,j]=0 
    
    emi_trans = np.transpose(emi_new)
    
 
    return emi_trans

def main(cfg_path):
    """ The main script for processing TNO inventory. 
    Takes a configuration file as input"""

    """Load the configuration file"""
    cfg = load_cfg(cfg_path)

    """Load or compute the country mask"""
    country_mask = get_country_mask(cfg)

    """Load or compute the interpolation map"""
    interpolation = get_interpolation(cfg,None,inv_name="meteotest",
                                      filename='mapping_meteotest.npy')


    """Starts writing out the output file"""
    output_path = cfg.output_path+"emis_"+str(cfg.year)+"_"+cfg.gridname+".nc"
    with nc.Dataset(output_path,"w") as out:
        prepare_output_file(cfg,out,country_mask)


        """ meteotest swiss inventory specific"""
        for cat in cfg.ch_cat:            
            for var in cfg.species:                  
                constfile= ''.join([cfg.input_path,'e',cat.lower(),'15_',var.lower(),'*'])   
                emi= np.zeros((cfg.ch_xn,cfg.ch_yn))
                for filename in sorted(glob(constfile)):                          
                    print(filename)
                    emi += get_ch_emi(filename) #(lon,lat)                                       
                start = time.time()
                out_var = np.zeros((cfg.ny,cfg.nx))           
                for lon in range(np.shape(emi)[0]):
                    for lat in range(np.shape(emi)[1]):
                        for (x,y,r) in interpolation[lon,lat]:
                            out_var[y,x] += emi[lon,lat]*r
                end = time.time()
                print("it takes ",end-start,"sec")    

                """convert unit from kg.year-1.cell-1 to g.h-1.cell-1"""
                out_var *= 1./(day_per_yr*24)/1000.                  

                out_var_name = var+"_"+cat+"_ch"
                out.createVariable(out_var_name,float,("rlat","rlon"))
                out[out_var_name].units = "kg h-1 cell-1"
                out[out_var_name].grid_mapping = "rotated_pole"
                out[out_var_name][:] = out_var

    


if __name__ == "__main__":
    cfg_name = sys.argv[1]
    main("./config_" + cfg_name)
