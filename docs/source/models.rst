Models
======


Various models are already supported by emiproc.


ICON-ART
--------

The ICON-ART model is an atmospheric transport model that uses 
a triangular grid. 

:py:func:`emiproc.exports.icon.export_icon_oem`
:py:class:`emiproc.grids.ICONGrid`


WRF-Chem
--------

WRF-Chem is the Weather Research and Forecasting (WRF) model coupled with Chemistry.  

:py:func:`emiproc.exports.wrf.export_wrf_hourly_emissions`
:py:class:`emiproc.exports.wrf.WRF_Grid`


NetCDF Rasters
--------------

NetCDF is a common format for storing gridded data.
emiproc can export emissions to NetCDF files, which can then be used as input for models.

:py:func:`emiproc.exports.rasters.export_raster_netcdf`
:py:func:`emiproc.exports.netcdf.nc_cf_attributes`