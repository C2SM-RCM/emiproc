
import netCDF4
import os

from emiproc import utilities as util 
from emiproc.hourly_emissions import speciation as spec 


def split_gnfr_f(filename):
    """
    Split GNFR category F into F1, F2, F3 and F4 for NMVOC and PM25.
    """
    with netCDF4.Dataset(filename, 'a') as nc:

        # NMVOC
        if 'NMVOC_F_AREA' in nc.variables:
            var = nc.variables['NMVOC_F_AREA']
            suffix = ''
        else:
            var = nc.variables['NMVOC_F_ch']
            suffix = '_ch'

        latname, lonname = var.dimensions

        for name, factor in [('NMVOC_F1', 0.7867), ('NMVOC_F2', 0.1426),
                             ('NMVOC_F3', 0.0010), ('NMVOC_F4', 0.0697)]:

            util.write_variable(nc, factor * var[:], name + suffix, latname,
                                lonname, var.units, overwrite=True)

        # PM25
        if 'PM25_F_AREA' in nc.variables:
            var = nc.variables['PM25_F_AREA']
            suffix = ''
        else:
            var = nc.variables['PM25_F_ch']
            suffix = '_ch'

        latname, loname = var.dimensions

        for name, factor in [('PM25_F1', 0.0304), ('PM25_F2', 0.6071),
                             ('PM25_F3', 0.0000), ('PM25_F4', 0.3624)]:

            util.write_variable(nc, factor * var[:], name + suffix, latname,
                                lonname, var.units, overwrite=True)

def create_input_tracers(cfg, output_path):
    """
    Creates the namelist file ``INPUT_OAE`` for COSMO-ART (resp. ``INPUT_GHG``
    for COSMO-GHG) from a cfg module.
    """

    catlist = cfg.catlist
    var_list = cfg.var_list
    tplist = cfg.tplist
    vplist = cfg.vplist
    contribution_list = spec.create_mapping()

    if cfg.model == 'cosmo-ghg':                                       
        input_nml = 'INPUT_GHG'
        ctl_name = '&GHGCTL'
    elif cfg.model == 'cosmo-art':                                     
        input_nml = 'INPUT_OAE'
        ctl_name = '&OAECTL'
    else:
        raise RuntimeError 

    # Make output folder
    os.makedirs(output_path, exist_ok=True)

    nml_filename = os.path.join(output_path, input_nml)

    # Write &OAECTL group
    with open(nml_filename, 'w+') as nml_file:
        oaectl_vals = [ctl_name,
                       "    in_tracers = %d," % len(catlist),
                       "    vertical_profile_nc = '../input/oae/vertical_profiles.nc',",
                       "    hour_of_day_nc = '../input/oae/hourofday.nc',",
                       "    day_of_week_nc = '../input/oae/dayofweek.nc',",
                       "    month_of_year_nc = '../input/oae/monthofyear.nc',",
                       "    gridded_emissions_nc = '../input/oae/emissions.nc',",
                       ]
        nml_file.write('\n'.join(oaectl_vals))
        nml_file.write('\n/\n')
    
    for i in range(len(catlist)):
        var = var_list[i]
        cats = catlist[i]
        
        values = []
        for cat in cats: 
            values.append(contribution_list[var].get_wildcard(cat))

        # Write &TRACER groups
        with open(nml_filename, 'a') as nml_file:
            group = {
                    'yshort_name': var[:-1],
                      'ycatl': ', '.join("'{}'".format(cat) for cat in cats),
                      'ytpl': ', '.join("'{}'".format(tp) for tp in tplist[i]),
                      'yvpl': ', '.join("'{}'".format(vp) for vp in vplist[i]),
                      'contribl': ', '.join("{}".format(value) for value in values),
                    }
            nml_file.write(group2text(group)) 

    
def group2text(group):
    """
    Returns formatted content of namelist group ``&TRACER``.
    """

    lines = ['&TRACER']
    for key, value in group.items():

        if key == '' or value == '':
            continue

        if key == 'yshort_name':
            value = "'%s'" % value

        lines.append('    %s = %s,' % (key, value))
    lines.append('/\n')

    return '\n'.join(lines)


