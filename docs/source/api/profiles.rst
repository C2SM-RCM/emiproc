.. _profiles_api:

Profiles
========

.. autofunction:: emiproc.profiles.operators.get_weights_of_gdf_profiles

.. autofunction:: emiproc.profiles.operators.weighted_combination

.. autofunction:: emiproc.profiles.operators.combine_profiles

.. autofunction:: emiproc.profiles.operators.group_profiles_indexes

.. autofunction:: emiproc.profiles.operators.remap_profiles

.. autofunction:: emiproc.profiles.utils.ratios_to_factors

.. autofunction:: emiproc.profiles.utils.factors_to_ratios

.. autofunction:: emiproc.profiles.utils.ratios_dataarray_to_profiles




Vertical Profiles 
=================

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
=================

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
