---
title: 'Emiproc: A Python package for emission inventory processing'
tags:
  - Python
  - Emission inventory
  - Air quality
  - Data processing
  - Remapping
  - Greenhouse gases
  - Air pollutants
authors:
  - name: Constantin Lionel
    orcid: 0009-0009-0347-4897
    affiliation: 1 
  - name: Brunner Dominik
    orcid: 0000-0002-4007-6902
     # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Thanwerdas Joel
    orcid: 0000-0003-1040-831X
    equal-contrib: true
    affiliation: 1
  - name: Keller Corina
    equal-contrib: true
    affiliation: 1
affiliations:
 - name: Empa, Laboratory for Air Pollution / Environmental Technology, Switzerland
   index: 1
date: 22 April 2024
bibliography: paper.bib

---

# Summary

Emission inventories are created by countries and regions in the effort to improve air quality and to reduce the impacts of climate change. Knowing the magnitude of emissions and the share of different source sectors is a critical first step to design efficient emission reduction policies.
Modellers use spatially gridded inventories to simulate the atmospheric transport of emitted species to compute their distribution and potential impact on the environment.
The simulations are often compared with measurements to control if the declared emissions and their trends are consistent with the observed changes in the atmosphere, enhancing the confidence in the inventories.

Inventories are created in multiple different formats and resolutions, which makes it difficult to compare and use them in atmospheric transport models.

Figure \ref{fig:tno} presents an example of a gridded inventory.


![CO2 emissions of the year 2015 from the TNOGHGco_v6 inventory produced by TNO \label{fig:tno}.](raster_total_CO2.png){ width=100% }

`emiproc` is a Python package that provides tools for processing and harmonizing emission inventories and for generating  input files for atmospheric models. It includes functions for reading, writing, and exporting emission inventory data to various formats used in atmospheric transport models. `emiproc` also provides functions for performing various 
operations on inventory data, such as remapping to different model grids, aggregating emissions by sector or pollutant, or scaling emissions based on projection scenarios. 

Emission input files can be generated in regular (e.g. hourly) intervals by applying sector- and country-specific temporal and vertical emission profiles. Alternatively, a small set of input files can be generated, which describe the sectorial gridded emissions and their temporal and vertical profiles.
This set of files can then be read by OEM (online emissions module) described in [@gmd-13-2379-2020], which applies the temporal and vertical scaling online during the model simulation. OEM is integrated into the atmospheric chemistry and transport models COSMO-ART and ICON-ART.

`emiproc` is designed to be flexible and extendable, allowing users to easily add custom functionality, to read new inventories or export data to other formats.

# Statement of need

Emission inventory data can be represented in various formats and resolutions. For example, TNO (Dutch Organization for Applied Scientific Research) provides an inventory which contains both, area emissions on a regular grid and point
sources at their exact locations.
Other inventories, such as the one from the city of Zurich, are provided as 
GIS (Geographic information system) data with various shapes depending on the category of the source. As an 
example: 

* Traffic emissions are represented as lines
* Building/industry emissions are represented as point sources
* Private boats on the lake are represented as polygons

Atmospheric models require emission inventories to be in a specific file format. The atmospheric chemistry transport model ICON-ART [@icon_art_2.1], for example, requires emissions on its complex, semi-structured triangular grid. As input for the OEM module it also requires cyclic time profiles to scale the emissions with daily, weekly and monthly variability.
As an other example, the Graz Lagrangian dispersion model (GRAL) can make direct use of emissions represented in a GIS format as line, point or rectangular sources. GRAL needs additional detailed information such as the height, the temperature or the gas exit velocity of a point source. 

When modellers design their simulations, they are often interested in modifying the inventories. For example, they could do the following: scale the emissions based on different future scenarios, aggregate emissions by sector or pollutant to simplify their simulations or combine multiple inventories to represent different sources such as anthropogenic and natural.

`emiproc` provides this functionality and has already been successfully applied for different use cases:

* [@acp-24-2759-2024] produced emission files for ICON-ART-OEM based on the EDGARv6
inventory [@edgar_v6] (Emissions Database for Global Atmospheric Research).

* [@donmez2024urban] conducted urban climate simulation using emissions produced 
with `emiproc` for cities of Zurich and Basel.

* [@ponomarev2024estimation] used `emiproc` to nest the Zurich city inventory
inside the Swiss national inventory and to further nest the Swiss inventory
inside the European TNOGHGco inventory.

`emiproc` shares some of its functionality with another python tool, `HERMESv3` [@hermesv3_part1], which is also designed to process emission data and generate input files for atmospheric transport models.
Compared to `HERMESv3`, which relies on specific configuration files, `emiproc` is more flexible, extensible and practical as it can by integrated in existing python-based workflows.

# History

An older version of the `emiproc` package was already published [@gmd-13-2379-2020], but it could only handle specific models and inventories. `emiproc` was then refactored starting in 2022 to satisfy the requirements of high flexibility and modularity. This included major changes to code structure, the addition of new capabilities, a major performance increase for the task of spatial regridding, a comprehensive documentation and the addition of test examples.

Since then the package is regularly updated with new features and bug fixes.


# Design 

To be able to use these different kind of inventories in atmospheric models, it is
necessary to harmonize them. This is what the `emiproc` package is designed for.

![Design of the idea behind emiproc \label{fig:design}](pipeline.drawio.png){ width=60% }

Thanks to the harmonization of the data, functions for additional data processing
can easily be applied to the different inventories once loaded into `emiproc`. 
The processed inventories can finally be exported to any of the formats supported by the package.

The API of `emiproc` leverages the advantages of object-oriented programming. 
`emiproc` is built on top of the `geopandas` package [@kelsey_jordahl_2020_3946761], 
which allows storing the geometries of the emission maps and offers many functionalities
related to geometric operations. In an inventory, the emission data is stored as 
[`geopandas.GeoDataFrame`](https://geopandas.org/en/stable/docs/reference/geodataframe.html).


## Inventory
The main object is the `Inventory` object. Any inventory is a subclass of the base `Inventory`.
`emiproc` provides many utility functions to process them. 
Typically, a processing function takes an inventory and some parameters as input 
and returns a new inventory with the processed data.

Example: the `group_categories` function regroups an inventory with a given set of source
categories to a new inventory with another set of categories, based on a mapping provided by the user. 
This is useful to reduce the number of categories simulated or to use a standardized set of 
categories such as the GNFR (Groupped Nomenclature For Reporting) sectors from [@emep_guidelines].


## Grid
As inventories and models always come on different grids, `emiproc` defines a 
grid and projection through the `Grid` object.
Many use cases are covered by the `RegularGrid` child class, which represents a standard latitude-longitude grid. 
On the other hand, users of models with more complex geometries can define their own grid. 

Example: The ICON Model [@IconRelease01] is simulated on an icosahedral grid composed of
triangular grid cells instead of rectangular. For this model, `emiproc` provides an `ICON_grid` object.

Functions in `emiproc` can then perform various operations on these `Grid` objects.

Example: the `remap_inventory` function remaps the emissions to a different grid.
It takes an `Inventory` and a `Grid` as input, and returns a new
`Inventory` containing emissions remapped to the new grid.

## Temporal and vertical profiles

To handle the temporal distribution of the emissions, `emiproc` uses the
`TemporalProfile` object, which is assigned to the `Inventory` and 
stores the temporal distribution of the emissions. 
Profiles can be either defined at specific datetimes or can be cyclic
profiles at different temporal resolutions (e.g. daily, weekly, monthly).

The vertical distribution of the emissions is handled in a similar manner by the 
`VerticalProfile` object. 

The temporal and vertical profile objects can be assigned very specifically to 
certain types of emissions. For example, it is possible to assign them to a specific
category / pollutant / country / gridcell.


## Export functions

`emiproc` contains many functions to export inventories for various
atmospheric models. The export produces all emission input files required by the
model. These export functions are designed to make life of the modellers
as simple as possible. 

Some transport models might require additional data not included in the 
inventories. In this case, `emiproc` provides error messages, which
guide the user into adding the missing data.


## Emissions generation 

Emission inventories do not necessarily contain all data relevant for a
simulation. For example, human respiration contributes to emissions of CO2 but
is rarely included in an inventory.  However, an emission map for this sector can 
be estimated based on population density. For this purpose, `emiproc` provides 
a module that helps produce emission maps that rely on a spatial proxy
(e.g. population density) and an emission factor.

Another example is CO2 emissions from vegetation. Several different models
are available that estimate the exchange of CO2 with vegetation. 
`emiproc` implements VPRM (Vegetation Photosynthesis and Respiration Model) [@vprm], which (by default) makes use of 
vegetation and soil moisture indices, extracted from the satellite observations.

This topic is already well-developed in the bottom-up component of the HERMES model [@hermesv3_part2], 
as it includes emission modules for a wide range of sectors and pollutants.

## Visualizing the data
At the end of the whole processing chain, a modeller may want to check the output files 
to see if the processing was successful. 

For some of the outputs, `emiproc` provides example plot scripts based on `matplotlib` [@matplotlib].

For regular grids, the `plot_inventory` function from the `emiproc.plots` module
can be used to plot the emissions on a map.
Figure \ref{fig:tno} was created using this function.



# Availability

The package is availabe on [GitHub](https://github.com/C2SM-RCM/emiproc)
and the documentation is available on [readthedocs](https://emiproc.readthedocs.io/).

[Tutorials](https://emiproc.readthedocs.io/en/master/tutos/tutorials.html)
are available to guide new users. 
A good first start is the
[Edgar processing tutorial](https://emiproc.readthedocs.io/en/master/tutos/edgar_processing.html)
which shows how `emiproc` can be used to load, process and export a freely available inventory.

# Acknowledgements

We acknowledge all the previous and current contributers of emiproc:
Michael Jähn, Gerrit Kuhlmann, Qing Mu, Jean-Matthieu Haussaire, David Ochsner, Katherine Osterried, Valentin Clément, Erik Koene,Alessandro Bigi

The developers of the models and inventories included in `emiproc`. In particular, 
Hugo Denier van der Gon and Jeroen Kuenen from TNO and the city of zurich for providing their inventories.


We also acknoledge C2SM (Center for Climate Systems Modeling) for the development of `emiproc`.

Finally we would like to thank the developers of the python packages used by `emiproc` and the whole python community for providing such a great ecosystem.

# References
