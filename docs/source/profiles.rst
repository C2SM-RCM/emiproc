
.. warning::
    This is not yet implemented


Vertical Profiles 
=================

Vertical information can be added in many ways.

Vertical data in emiproc is always the height over the ground.

We can distinguish the cases as follow:
1. a point in the vertical domain
2. an area in the vertical domain

Operations to implement 
-----------------------

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