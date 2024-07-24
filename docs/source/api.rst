
API 
===

Inventory Class
---------------

.. autoclass:: emiproc.inventories.Inventory
    :members:

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

.. autofunction:: emiproc.inventories.utils.drop

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

Speciation in emiproc means splitting a substance in multiple sub-substances. 

This can be used for example to split NOx in NO and NO2 or to split 
the anthropogenic and biogenic part of a CO2.

.. autofunction:: emiproc.speciation.speciate

.. autofunction:: emiproc.speciation.read_speciation_table

.. autofunction:: emiproc.speciation.merge_substances

Input/Output
^^^^^^^^^^^^

.. autofunction:: emiproc.speciation.read_speciation_table

Utilities
---------

.. autofunction:: emiproc.utilities.get_country_mask

.. autofunction:: emiproc.utilities.get_natural_earth

.. autofunction:: emiproc.utilities.get_timezone_mask

.. autofunction:: emiproc.utilities.get_timezones


.. _profiles_api:

Profiles
--------

.. autofunction:: emiproc.profiles.operators.get_weights_of_gdf_profiles

.. autofunction:: emiproc.profiles.operators.weighted_combination

.. autofunction:: emiproc.profiles.operators.combine_profiles

.. autofunction:: emiproc.profiles.operators.group_profiles_indexes

.. autofunction:: emiproc.profiles.operators.remap_profiles

.. autofunction:: emiproc.profiles.utils.ratios_to_factors

.. autofunction:: emiproc.profiles.utils.factors_to_ratios

.. autofunction:: emiproc.profiles.utils.ratios_dataarray_to_profiles




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

.. autofunction:: emiproc.exports.icon.get_constant_time_profile

input/output
^^^^^^^^^^^^

.. autofunction:: emiproc.profiles.temporal_profiles.read_temporal_profiles

.. autofunction:: emiproc.profiles.temporal_profiles.from_csv

.. autofunction:: emiproc.profiles.temporal_profiles.from_yaml

.. autofunction:: emiproc.profiles.temporal_profiles.to_yaml


Data Generation
---------------

Functions that can be used to generate some parts of inventory data.

Heating Degree Days (HDD)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: emiproc.profiles.hdd.create_HDD_scaling_factor


Human Respiration 
^^^^^^^^^^^^^^^^^ 

.. autofunction:: emiproc.human_respiration.people_to_emissions

.. autoclass:: emiproc.human_respiration.EmissionFactor


VPRM 
^^^^

.. automodule:: emiproc.profiles.vprm
    :members:



Exporting 
---------

.. autofunction:: emiproc.exports.netcdf.nc_cf_attributes

.. autofunction:: emiproc.exports.icon.export_icon_oem

.. autofunction:: emiproc.exports.icon.make_icon_time_profiles

.. autofunction:: emiproc.exports.icon.make_icon_vertical_profiles

.. autoenum:: emiproc.exports.icon.TemporalProfilesTypes
    
.. autofunction:: emiproc.exports.rasters.export_raster_netcdf

.. autofunction:: emiproc.exports.profiles.export_inventory_profiles
    