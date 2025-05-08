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

.. autoclass:: emiproc.profiles.temporal.profiles.TemporalProfile


Cyclic Profiles 
^^^^^^^^^^^^^^^

Profiles that can be repeated and applied at any time.

.. autoclass:: emiproc.profiles.temporal.profiles.DailyProfile

.. autoclass:: emiproc.profiles.temporal.profiles.SpecificDayProfile

.. autoclass:: emiproc.profiles.temporal.profiles.WeeklyProfile

.. autoclass:: emiproc.profiles.temporal.profiles.MounthsProfile

.. autoclass:: emiproc.profiles.temporal.profiles.HourOfWeekProfile

Year Covering Profiles 
^^^^^^^^^^^^^^^^^^^^^^

Profiles that cannot be repeated, but apply specifically to a given time.

.. autoclass:: emiproc.profiles.temporal.profiles.HourOfYearProfile

.. autoclass:: emiproc.profiles.temporal.profiles.HourOfLeapYearProfile


Composite Profiles
^^^^^^^^^^^^^^^^^^^

.. autoclass:: emiproc.profiles.temporal.composite.CompositeTemporalProfiles

.. autofunction:: emiproc.profiles.temporal.composite.make_composite_profiles


Utilities
^^^^^^^^^

.. autofunction:: emiproc.profiles.temporal.operators.interpolate_profiles_hour_of_year

.. autofunction:: emiproc.profiles.temporal.operators.create_scaling_factors_time_serie

.. autofunction:: emiproc.profiles.temporal.operators.profile_to_scaling_factors


Specific days 
^^^^^^^^^^^^^^^

Utilites to create profiles for specific days of the week.

.. autoclass:: emiproc.profiles.temporal.specific_days.SpecificDay

.. autofunction:: emiproc.profiles.temporal.specific_days.days_of_specific_day

.. autofunction:: emiproc.profiles.temporal.specific_days.get_days_as_ints

.. autofunction:: emiproc.exports.icon.get_constant_time_profile

input/output
^^^^^^^^^^^^

.. autofunction:: emiproc.profiles.temporal.io.read_temporal_profiles

.. autofunction:: emiproc.profiles.temporal.io.from_csv

.. autofunction:: emiproc.profiles.temporal.io.from_yaml

.. autofunction:: emiproc.profiles.temporal.io.to_yaml
