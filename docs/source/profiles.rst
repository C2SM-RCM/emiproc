Emission Profiles
=================

Most of the inventories provide annual values for the emissions of a specific region.
For simulations we need to know how they vary with time and elevation.

For that we downscale the annual values using profiles.


Scaling factors vs Ratios 
-------------------------

Here we clarify the terminology as profiles can be defined in 2 ways.

1. Scaling factors
2. Ratios

Ratios signify the proportion of the total emission that is emitted
at a specific time or height.

Scaling factor give a value for which we can multiply the average emmission
to get the emission at a specific time or height.

Ratios must sum up to one.
Scaling factors must have an average of one.

Emiproc uses internally ratios for the profiles.


Vertical Profiles 
=================

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

In the inventory, the profiles will be stored in a list where every 
profile has an index in the list.
Vertical profiles cover a range of indices.
TODO: add example.

Assigning profiles to emissions
-------------------------------

We can add vertical profiles the following way:

1. Adding profiles to category / substance / cell / time .
2. Specify a profile to a shape in the gdfs.

The second method is the simplest. One column of each gdfs can be 
called `__v_profile__` containing for each shape the index of the matching
vertical profile.


The first option requires to create a data array containing the profile index
to use for any combination of the 4 coordinates.
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
=================

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