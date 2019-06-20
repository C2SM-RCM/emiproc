from make_online_emissions import *
import country_code


#species = ['CO2', 'CO', 'CH4']
species = ['CO2']
gnfr_cat = [ "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M" ]

exclude_CH = False
only_CH = False


def interpolate_tno_to_cosmo_grid(tno,cfg):
    return interpolate_to_cosmo_grid(tno,"tno",cfg)

def interpolate_tno_to_cosmo_point(source,tno,cfg):
    """This function returns the indices of the cosmo grid cell that contains the point source
    input : 
       - source : The index of the  point source from the TNO inventory
       - tno : the TNO netCDF file, already open.
       - cfg : the configuration file
    output :
       - (cosmo_indx,cosmo_indy) : the indices of the cosmo grid cell containing the source
""" 
    lon_source = tno["longitude_source"][source]
    lat_source = tno["latitude_source"][source]
    
    return interpolate_to_cosmo_point(lat_source,lon_source,cfg)
    
    # transform = ccrs.RotatedPole(pole_longitude=cfg.pollon, pole_latitude=cfg.pollat)
    # point = transform.transform_point(lon_source,lat_source,ccrs.PlateCarree())

    # cosmo_indx = int(np.floor((point[0]-cfg.xmin)/cfg.dx))
    # cosmo_indy = int(np.floor((point[1]-cfg.ymin)/cfg.dy))

    # return (cosmo_indx,cosmo_indy)



def var_name(s,snap,cat_kind):
    """Returns the name of a variable for a given species and snap
    input : 
       - s : species name ("CO2", "CH4" ...)
       - snap : Category number
       - cat_kind : Kind of category. must be "SNAP" or "NFR" 
    output :
       - returns a string which concatenate the species with the category number
    """
    out_var_name = s+"_"
    if cat_kind=="SNAP":
        if snap==70:                        
            out_var_name += "07_"
        else:
            if snap>9:
                out_var_name += str(snap)+"_"
            else:
                out_var_name += "0"+str(snap)+"_"
    elif cat_kind=="NFR":
        out_var_name += snap
    elif cat_kind=="NFR_BAU":
        out_var_name += snap+"_BAU_"
    elif cat_kind=="NFR_CC":
        out_var_name += snap+"_CC_"
    else:
        print("Wrong cat_kind in the config file. Must be SNAP or NFR")
        raise ValueError
    return out_var_name

def tno_file(species,cfg):
    """Returns the path to the TNO file for a given species.
    For some inputs, the CO2 is in a different file than the other species for instance.
    There are two parameters in the config file to distinguish them.
    Namely, tnoCamsPath and tnoMACCIIIPath    
    """

    # get the TNO inventory
    if species=="CO2":
        tno_path = cfg.tnoCamsPath
    else:
        tno_path = cfg.tnoMACCIIIPath

    if os.path.isfile(tno_path):
        return tno_path

    first_year = 2000
    last_year = 2011

    if cfg.year>last_year:
        base_year = str(last_year)
    elif cfg.year<first_year:
        base_year = str(first_year)
    else:
        base_year = str(cfg.year)
    tno_path += base_year+".nc"
    
    return tno_path



def main(cfg_path):
    """ The main script for processing TNO inventory. 
    Takes a configuration file as input"""

    """Load the configuration file"""
    cfg = load_cfg(cfg_path)
    
    """Load or compute the country mask"""
    country_mask = get_country_mask(cfg)
    country_mask = np.transpose(country_mask)
    if exclude_CH:
        mask = country_mask == country_codes['CH']
        print('Exclude country "CH" (country code %d)' % country_codes['CH'])
    if only_CH:
        mask = country_mask != country_codes['CH']
        print('Only include "CH" (country code %d)' % country_codes['CH'])

    """Set names for longitude and latitude"""
    if (cfg.pollon==180 or cfg.pollon==0) and cfg.pollat==90:
        lonname = "lon"; latname="lat"
        print('Non-rotated grid: pollon = %f, pollat = %f' % (cfg.pollon, cfg.pollat))
    else:
        lonname = "rlon"; latname="rlat"
        print('Rotated grid: pollon = %f, pollat = %f' % (cfg.pollon, cfg.pollat))

    """Starts writing out the output file"""
    output_path = os.path.join(cfg.output_path, 
                               "emis_" + str(cfg.year) + "_" + cfg.gridname + ".nc")
    with nc.Dataset(output_path,"w") as out:
        prepare_output_file(cfg,out,country_mask)

        """Load or compute the interpolation maps"""
        list_input_files = set([tno_file(s,cfg) for s in cfg.species])
        for f in list_input_files:
            print(f)            
            with nc.Dataset(f) as tno:
                interpolation = get_interpolation(cfg,tno,
                                    filename='mapping_tno.npy')

                """From here onward, quite specific for TNO"""        
                
                """mask corresponding to the area/point sources"""
                selection_area  = tno["source_type_index"][:]==1
                selection_point = tno["source_type_index"][:]==2

                # SNAP ID 
                #tno_snap = tno[cfg.tno_cat_var][:].tolist() 
                tno_snap = cfg.tno_snap
                    
                """Initialize total flux"""
                total_flux = {}
                species_list = [s for s in cfg.species if tno_file(s,cfg)==f]
                for s in species_list:
                    total_flux[s] = np.zeros((cfg.ny,cfg.nx))

                for snap in cfg.snap:
                    """In emission_category_index, we have the index of the category, starting with 1.
                    It means that if the emission is of SNAP1, it will have index 1, SNAP34 index 3"""
                    if snap==70:
                        snap_list=[i for i in range(70,80) if i in tno_snap]
                    elif snap=="F":
                        snap_list=["F1","F2","F3"]
                    else:
                        snap_list=[snap]
                    
                    print(snap_list, tno_snap)
                    """mask corresponding to the given snap category"""
                    selection_snap = np.array([tno["emission_category_index"][:] == tno_snap.index(i)+1 for i in snap_list])
                    
                    """mask corresponding to the given snap category for area/point"""
                    selection_snap_area  = np.array([selection_snap.any(0),selection_area]).all(0)
                    selection_snap_point = np.array([selection_snap.any(0),selection_point]).all(0)
                    
                    for s in species_list:
                        print("Species",s,"SNAP",snap)
                        out_var_area = np.zeros((cfg.ny,cfg.nx))
                        out_var_point = np.zeros((cfg.ny,cfg.nx))

                        if s=="CO2":
                            """add fossil and bio fuel CO2"""
                            var = tno["co2_ff"][:]+tno["co2_bf"][:]
                        elif s=="CO":
                            # ignore problems here to get Qings case to work.
                            # She just commented it out, so I guess she doesn't
                            # need CO?
                            try:
                                """add fossil and bio fuel CO"""
                                var = tno["co_ff"][:]+tno["co_bf"][:]
                            except:
                                # just do what 'else' would have done
                                var = tno[s.lower()][:]
                        elif s=="PM25":
                            var = tno["pm2_5"]
                        else:
                            var = tno[s.lower()][:]

                        start = time.time()
                        for (i,source) in enumerate(var):
                            if selection_snap_area[i]:
                                lon_ind = tno["longitude_index"][i]-1
                                lat_ind = tno["latitude_index"][i]-1
                                for (x,y,r) in interpolation[lon_ind,lat_ind]:
                                        out_var_area[y,x]+=var[i]*r
                            if selection_snap_point[i]:
                                (indx,indy) = interpolate_tno_to_cosmo_point(i,tno,cfg)
                                if indx>=0 and indx<cfg.nx and indy>=0 and indy<cfg.ny:
                                    out_var_point[indy,indx]+=var[i]

                        end = time.time()
                        print("it takes ",end-start,"sec")                     
                        ## TO DO : 
                        ## - Add the factor from 2011 to 2015
                        

                        """convert unit from kg.year-1.cell-1 to kg.m-2.s-1"""

                        """calculate the areas (m^^2) of the COSMO grid"""
                        cosmo_area = 1./gridbox_area(cfg)
                        out_var_point*= cosmo_area.T*convfac
                        out_var_area *= cosmo_area.T*convfac
                        out_var_name = var_name(s,snap,cfg.cat_kind)

                        for (t,sel,out_var) in zip(["AREA","POINT"],
                                           [selection_snap_area,selection_snap_point],
                                           [out_var_area,out_var_point]):
                            if sel.any():
                                if exclude_CH or only_CH:
                                    out_var[mask] = 0
                                if not out_var_name in out.variables.keys():
                                    out.createVariable(out_var_name,float,
                                                       (latname,lonname))
                                    out[out_var_name].units = "kg m-2 s-1"
                                    if lonname == "rlon" and latname == "rlat":
                                        out[out_var_name].grid_mapping = "rotated_pole"
                                    out[out_var_name][:] = out_var
                                else:
                                    out[out_var_name][:] += out_var

                                total_flux[s] += out_var

            
                """Calcluate total emission/flux per species"""
                for s in species_list:
                    out.createVariable(s,float,(latname,lonname))
                    out[s].units = "kg m-2 s-1"
                    if lonname == "rlon" and latname == "rlat":
                        out[s].grid_mapping = "rotated_pole"
                    out[s][:] = total_flux[s]

        """Create dummy variables for merging inventories"""
        for s in species:
            if not s in out.variables.keys():
                out.createVariable(s,float,(latname,lonname))
                if lonname == "rlon" and latname == "rlat":
                    out[s].grid_mapping = "rotated_pole"
                out[s].units = "kg m-2 s-1"
                out[s][:] = 0
            for gnfr in gnfr_cat:
                varname = s + '_' + gnfr
                if not varname in out.variables.keys():
                    out.createVariable(varname,float,(latname,lonname))
                    if lonname == "rlon" and latname == "rlat":
                        out[varname].grid_mapping = "rotated_pole"
                    out[varname].units = "kg m-2 s-1"
                    out[varname][:] = 0

                        
if __name__ == "__main__":
    cfg_name = sys.argv[1]
    main("./config_" + cfg_name)
    
