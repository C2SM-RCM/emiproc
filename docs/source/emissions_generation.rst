Emission Models 
===============

emiproc can also genereate emissions using various well known methods 


.. _vprm:

VPRM 
---- 

The VPRM simulates the vegetation respiration and photosynthesis using the following 
inputs:

* temperature
* radiation
* satellite data
* vegetation parameters


There are 4 implementation of the VPRM model:

- standard (Mahadevan et al., 2008)
- urban  (Hardiman et al., 2017)
- urban winbourne (Winbourne et al., 2021)
- modified (Gourdij et al., 2021)

see :py:mod:`emiproc.profiles.vprm`

Standard VPRM
^^^^^^^^^^^^^

PAR (Photosynthetically Active Radiation) is calculated from the shortwave radiation:

.. math::
    \\mathrm{PAR} = \\frac{\\mathrm{RAD}}{0.505}

Respiration is calculated from the temperature:

.. math::
    \\mathrm{Resp} = \\alpha * T + \\beta

The Gross Ecosystem Exchange (GEE) is calculated from the temperature, PAR and the vegetation indices:

.. math::
    \\mathrm{GEE} = \\lambda * T_{scale} * P_{scale} * W_{scale} * \\mathrm{EVI} * \\frac{ \\mathrm{PAR} }{1 + \\frac{\\mathrm{PAR}}{PAR0}}

where the different scales are:

- :math:`T_{scale}`: Temperature scale

.. math::
    T_{\\text{scale}} = \\frac{(T - T_{\\text{min}}) \\cdot (T - T_{\\text{max}})}{(T - T_{\\text{min}}) \\cdot (T - T_{\\text{max}}) + (T - T_{\\text{opt}})^2} \\text{if } T \\geq T_{\\text{min}} \\text{ else } 0

- :math:`P_{scale}`: Photosynthesis scale

.. math::
    P_{scale} = \\frac{1 + \\mathrm{LSWI}}{2}

- :math:`W_{scale}`: Water scale

.. math::
    W_{scale} = \\frac{1 + \\mathrm{LSWI}}{1 + \\mathrm{LSWI}_{max}}

The Net Ecosystem Exchange (NEE) is calculated from the respiration and GEE.

.. math::
    \\mathrm{NEE} = \\mathrm{Resp} + \\mathrm{GEE}


Units for all fluxes (NEE, GEE, Resp, ...) are

.. math::
    \\frac{\\mu mol_{\\mathrm{CO2}}}{m^2 * s}

Urban VPRM
^^^^^^^^^^

The VPRM model can be extended to urban areas according to [Urban_VPRM_Hardiman_2017].

- A "urban temperature" is used instead of the global temperature to represent
    the urban heat island phenomenon.
- The formula for :math:`P_{scale}` is modified to

.. math::
    P_{scale} = \\frac{\\mathrm{EVI} - \\mathrm{EVI}_{min}}{\\mathrm{EVI}_{max} - \\mathrm{EVI}_{min}}

- The respiration is calculated differently

.. math::
    \\mathrm{Resp} = \\frac{\\mathrm{Resp_{e-init}}}{2} * (1 - \\mathrm{ISA}) + \\frac{\\mathrm{EVI} + \\mathrm{EVI}_{min} * \\mathrm{ISA}}{\\mathrm{EVI}_{ref}} * \\frac{\\mathrm{Resp_{e-init}}}{2}

where :math:`\\mathrm{Resp_{e-init}}` is the basic vprm respiration and :math:`\\mathrm{ISA}` is the impervious surface area at the vegetation location.


Urban VPRM with winbourne
^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a variant of the urban VPRM model that uses a different formulation for :math:`T_{scale}`:

.. math::
    T_{\\text{scale}} = \\begin{cases}
    \\frac{(T - 0) \\cdot (T - 40)}{(T - 0) \\cdot (T - 40) + (T - 20)^2} & \\text{if } T \\leq 20 \\\\
    1 & \\text{if } 20 < T < 30 \\\\
    \\frac{(T - 0) \\cdot (T - 40)}{(T - 0) \\cdot (T - 40) + (T - 30)^2} & \\text{if } T \\geq 30
    \\end{cases}



modified-VPRM
^^^^^^^^^^^^^

The modified-VPRM model follows the standard VPRM for GEE and has a different model
for the estimate of respiration: for more details see [VPRM_modified_groudji_2022]_.


Heating Degree Days (HDD)
-------------------------


Calculates the demand of heating based on the temperature.

see :py:mod:`emiproc.profiles.hdd`


Human Respiration
-----------------

Human respirations can be generated using maps of population density and 
emission factors per person.

see :py:mod:`emiproc.human_respiration`
