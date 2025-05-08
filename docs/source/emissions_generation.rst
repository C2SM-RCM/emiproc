Emission Models 
===============

emiproc can also genereate emissions using various well known methods 


VPRM 
---- 

The VPRM simulates the vegetation respiration and photosynthesis using the following 
inputs:

* temperature
* radiation
* satellite data

see :py:mod:`emiproc.profiles.vprm`


Heating Degree Days (HDD)
-------------------------


Calculates the demand of heating based on the temperature.

see :py:mod:`emiproc.profiles.hdd`


Human Respiration
-----------------

Human respirations can be generated using maps of population density and 
emission factors per person.

see :py:mod:`emiproc.human_respiration`
