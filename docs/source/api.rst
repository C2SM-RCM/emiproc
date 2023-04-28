
API 
===

Inventory Class
---------------

.. autoclass:: emiproc.inventories.Inventory

Emissions Informations
----------------------

.. autoclass:: emiproc.inventories.EmissionInfo

Grid Class
----------

.. autoclass:: emiproc.grids.Grid

.. autoclass:: emiproc.grids.RegularGrid

Inventory Operators
-------------------

.. autofunction:: emiproc.inventories.utils.add_inventories

.. autofunction:: emiproc.inventories.utils.combine_inventories

.. autofunction:: emiproc.inventories.utils.scale_inventory

.. autofunction:: emiproc.inventories.utils.get_total_emissions

Geometric Transformations 
-------------------------

.. autofunction:: emiproc.inventories.utils.crop_with_shape


Remapping
---------

.. autofunction:: emiproc.regrid.remap_inventory

Categories Manipulations 
------------------------

.. autofunction:: emiproc.inventories.utils.validate_group

.. autofunction:: emiproc.inventories.utils.group_categories


Speciation
----------

.. autofunction:: emiproc.speciation.speciate_inventory

.. autofunction:: emiproc.speciation.speciate_nox

Exporting 
---------

.. autofunction:: emiproc.exports.netcdf.nc_cf_attributes

.. autofunction:: emiproc.exports.icon.export_icon_oem
    
.. autofunction:: emiproc.exports.rasters.export_raster_netcdf
    