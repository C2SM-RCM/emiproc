
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


Profiles
--------

.. autofunction:: emiproc.profiles.operators.get_weights_of_gdf_profiles

.. autofunction:: emiproc.profiles.operators.weighted_combination

.. autofunction:: emiproc.profiles.operators.combine_profiles

.. autofunction:: emiproc.profiles.operators.group_profiles_indexes

.. autofunction:: emiproc.profiles.utils.ratios_to_factors

.. autofunction:: emiproc.profiles.utils.factors_to_ratios




Vertical Profiles 
-----------------

.. autoclass:: emiproc.profiles.vertical_profiles.VerticalProfile

.. autoclass:: emiproc.profiles.vertical_profiles.VerticalProfiles

Operators On Vertical Profiles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: emiproc.profiles.vertical_profiles.resample_vertical_profiles

.. autofunction:: emiproc.profiles.vertical_profiles.check_valid_vertical_profile


input/output 
^^^^^^^^^^^^

.. autofunction:: emiproc.profiles.vertical_profiles.from_csv


Temporal Profiles
-----------------

.. autoclass:: emiproc.profiles.temporal_profiles.TemporalProfile

Cyclic Profiles 
^^^^^^^^^^^^^^^

.. autoclass:: emiproc.profiles.temporal_profiles.DailyProfile

.. autoclass:: emiproc.profiles.temporal_profiles.SpecificDayProfile

.. autoclass:: emiproc.profiles.temporal_profiles.WeeklyProfile

.. autoclass:: emiproc.profiles.temporal_profiles.MounthsProfile

.. autoclass:: emiproc.profiles.temporal_profiles.HourOfWeekProfile

.. autoclass:: emiproc.profiles.temporal_profiles.HourOfYearProfile

.. autoclass:: emiproc.profiles.temporal_profiles.HourOfLeapYearProfile



Utilities
^^^^^^^^^

.. autofunction:: emiproc.profiles.temporal_profiles.create_scaling_factors_time_serie

.. autofunction:: emiproc.profiles.temporal_profiles.profile_to_scaling_factors

.. autoclass:: emiproc.profiles.temporal_profiles.SpecificDay

input/output
^^^^^^^^^^^^

.. autofunction:: emiproc.profiles.temporal_profiles.read_temporal_profiles

.. autofunction:: emiproc.profiles.temporal_profiles.from_csv

.. autofunction:: emiproc.profiles.temporal_profiles.from_yaml

.. autofunction:: emiproc.profiles.temporal_profiles.to_yaml


Exporting 
---------

.. autofunction:: emiproc.exports.netcdf.nc_cf_attributes

.. autofunction:: emiproc.exports.icon.export_icon_oem
    
.. autofunction:: emiproc.exports.rasters.export_raster_netcdf
    