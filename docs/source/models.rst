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


Fluxy 
-----

Fluxy is a Python package for visualizing inversion model results. 
It also supports inventories in the form of NetCDF files.

See the
`following script <https://github.com/C2SM-RCM/emiproc/blob/master/scripts/exports/edgar_2_fluxy.ipynb>`_ 
to learn how to export inventories to 
fluxy and how to visualize them with fluxy.

:py:func:`emiproc.exports.fluxy.export_fluxy`