from make_online_emissions import *
from glob import glob

def get_lat_lon(filename):
    with open(filename) as f:
        lat = []
        lon = []
        for l in f:
            data = l.split(";")
            try:
                lat.append(float(data[0]))
                lon.append(float(data[1]))
            except ValueError:
                continue
    return lat,lon

def get_emi(filename,cfg):
    lon_dim,lat_dim,lon_var,lat_var = get_dim_var(None,"edgar",cfg)
    emi = np.zeros((lon_dim,lat_dim))
    with open(filename) as f:
        for l in f:
            data = l.split(";")
            try:
                lat = float(data[0])
                lon = float(data[1])
                if (lat <=lat_var[-1]) and (lon <= lon_var[-1]) and \
                   (lat >=lat_var[0]) and (lon >=lon_var[0]):
                    emi_lon = round((lon - cfg.edgar_xmin)/cfg.edgar_dx)
                    emi_lat = round((lat - cfg.edgar_ymin)/cfg.edgar_dy)

                    emi[emi_lon,emi_lat] = float(data[2])
            except ValueError:
                continue
    return emi

def main(cfg_path):
    """ The main script for processing TNO inventory. 
    Takes a configuration file as input"""

    """Load the configuration file"""
    cfg = load_cfg(cfg_path)

    """Load or compute the country mask"""
    country_mask = get_country_mask(cfg)

    """Load or compute the interpolation map"""
    interpolation = get_interpolation(cfg,None,inv_name="edgar")

    """Starts writing out the output file"""
    output_path = cfg.output_path+"emis_"+str(cfg.year)+"_"+cfg.gridname+".nc"
    with nc.Dataset(output_path,"w") as out:
        prepare_output_file(cfg,out,country_mask)


        """ EDGAR specific"""
        for cat in cfg.edgar_cat:
            path = os.path.join(cfg.input_path,cat)
            files = glob(path+"/*_2015_*")            
            if len(files)>1 or len(files)<0:
                print("There are two many or two few files")
                print(files)
            else:
                filename = files[0]
            print(filename)
                        
            start = time.time()
            out_var = np.zeros((cfg.ny,cfg.nx))
            print(out_var.shape)
            #lat,lon = get_lat_lon(filename)
            emi = get_emi(filename,cfg) #(lon,lat)
            print(emi.shape)
            for lon in range(emi.shape[0]):
                for lat in range(emi.shape[1]):
                    for (x,y,r) in interpolation[lon,lat]:
                        # Maybe add a conversion of some kind.
                        out_var[y,x] += emi[lon,lat]*r
            end = time.time()
            print("it takes ",end-start,"sec")                     

            out_var_name = cfg.species+"_"+cat
            out.createVariable(out_var_name,float,("rlat","rlon"))
            out[out_var_name].units = "kg m-2 s-1"
            out[out_var_name][:] = out_var

    


if __name__ == "__main__":
    main("./config_EDGAR")

