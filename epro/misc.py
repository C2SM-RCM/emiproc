
import netCDF4

from . import utilities as util


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

