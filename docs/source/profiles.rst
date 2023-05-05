Emission Profiles
=================

Most of the inventories provide annual values for the emissions of a specific region.
For simulations we need to know how they vary with time and elevation.

For that we downscale the annual values using profiles.


The total emission :math:`E` is the sum of the emission for an emission source.
It is given in :math:`kg/year/source`.
Sources can be assigned profiles to describe how the emission varies with time and height.

We then can split the profiles in :math:`N` intervals of time or height. 

.. math::

    E = \sum_{i}^{N} E_i

Scaling factors vs Ratios 
-------------------------

There are two ways of describing the profiles:

1. Scaling factors
2. Ratios


Ratios signify the proportion of the total emission :math:`E`
that is emitted at a specific interval of time or height.

:math:`r_i` the ratio at index :math:`i`
we have 

.. math::
    
        \sum_{i}^{N} r_i = 1

        E_i = E * r_i

    
Scaling factor give a value for which we can multiply the average emmission
to get the emission at a specific time or height.

:math:`f_i` is the scaling factor at index :math:`i` 

.. math::

        \frac{\sum_{i}^{N} f_i}{N} = 1

        E_i = \frac{E * f_i}{N}


Based on these formulas, we can easily convert from one to the other.

.. math::

        f_i = N * r_i

        r_i = \frac{f_i}{N}




Ratios sum up to one.
Scaling factors have an average of one.

Emiproc stores internally ratios for the profiles, but uses and exports sometimes
scaling factors.

scaling factors have the advantage that you can easily use them to combine profiles 
together.


Vertical Profiles 
-----------------

Vertical profiles in emiproc are handled by the class
:py:class:`~emiproc.profiles.vertical_profiles.VerticalProfile`

Each inventory is assigned a set of profile, which can be attributed
to a specific substance, category, gridcell or time .

Depending on how the data is gathered, different emission sources will
have different profiles. In exemple a powerplant will emit at just one height.

When applying operations (regridding, groupping, ...) on the inventories,
the profiles might need to be changed as well.

Vertical data in emiproc is always the height over the ground.


Creating profiles
-----------------

Create the profiles is simple, if you have many profiles on the same
vertical grid use :py:class:`~emiproc.profiles.vertical_profiles.VerticalProfiles`
If you have different vertical grids create a 
:py:class:`~emiproc.profiles.vertical_profiles.VerticalProfile`
for each of your vertical grid.

The two kind of profiles can be combined in the inventory.

Assigning Profiles to inventories
---------------------------------

In the inventory, the profiles will be stored as a 
:py:class:`~emiproc.profiles.vertical_profiles.VerticalProfiles` for vertical profiles.

The temporal profiles are stored as a list of list of :py:class:`~emiproc.profiles.temporal_profiles.TemporalProfile`.

The following paragraph will explain how to assign the profiles to emissions.

Assigning profiles to emissions
-------------------------------

We can add vertical profiles the following way:

1. Adding profiles to gridded emissions.
2. Specify a profile to a shape in the gdfs.

The second method is the simplest. One column of each gdfs can be 
called `__v_profile__` and `__t_profile__` for the vertical and time profiles.
Each shape of the gdfs can then be assigned to a desired profile.
These columns contain the index of the profile assigned to that shape.
If no profile is assigned, a value of -1 can be used as the index.


The first option requires to create a data array containing the profile index
to use for any combination of the 4 coordinates ( category / substance / cell / time ).
The coordinates don't need to all be present in the file, one could simply
put one of them, and emiproc assumes the vertical profiles are the same 
no matter the other coordinates.

TODO: put an example

Behaviour on Operations
-----------------------

Operations on inventories can be tricky.
The principle is to always weight correctly the different ratios.
Sometimes arbitrary decisions have to be done.
For example when adding two inventories, we need to decide if we 
use the vertical scales of on of the two, or if we want to go 
for a fancy merging. 

Add inventories (should scale each grid cell total values to do a weighted sum of the profiles)

Temporal Profiles
-----------------

Fundamentally there are 2 ways of defining the time profile.

1. Deterministic
2. Periodic 

Either you have a specific value at a certain time . (similarly to vertical 
where you have a value at a specific height)

Or you have periodic patterns that define the behaviour
(hour of day, day of week, mounth of year)

Profiles from data 
------------------

Some power plant give deterministic profiles .

Traffic is usually modelled as a periodic pattern.

Merging Deterministic and Periodic
----------------------------------

The output of that should be choosable by the user.

Merging Different Periodic profiles
-----------------------------------

Sometimes the frequency at which profiles are given will be different 
for categories.


This seems that we have to either assume the lower frequency behaves as 
the higher. Or resample the lowest on the highest.


Uncertainty on time profiles 
----------------------------

Sometimes profiles are given uncertainty values.
This is currently not handled in emiproc.

One would have to make sure the uncertainty propagate correctly while merging.



Examples
========

Vertical profile based on roof heights.
---------------------------------------

Adding an elevation to each source.
-----------------------------------