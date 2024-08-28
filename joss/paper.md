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
  - name: Lionel Constantin
    orcid: 0009-0009-0347-4897
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Brunner Dominik
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
affiliations:
 - name: Empa, Laboratory for Air Pollution / Environmental Technology, Switzerland
   index: 1
date: 22 April 2024
bibliography: paper.bib

---

# Summary

In the effort to reduce air pollution and greenhouse gases, emission inventories
are created by countries and regions to estimate which sectors emit how much of which 
polluant.
Modellers use these inventories to simulate the transport of these substances in the
athmosphere and then compare the results with measurements to control if the 
declared emissions match the measurements, establishing the correctness of the inventories.

The inventories are
often created in different formats and resolutions, which can make it difficult to compare
and use them in air transport models.
Figure \ref{fig:tno} an example of a gridded inventory.


![CO2 emissions of the for year 2015 from the TNO inventory. \label{fig:tno}](raster_total_CO2.png){ width=100% }. 

`emiproc` is a Python package that
provides tools for processing and harmonizing emission inventories. It includes functions
for reading, writing, and exporting emission inventory data to various formats used 
in air transport models. `emiproc` also provides functions for performing various 
operations on emission inventory data, such as remapping emissions to different spatial
resolutions, aggregating emissions by sector or pollutant, or scaling emissions based on
projection scenarios. The package is designed to be flexible and extensible, allowing
users to easily add new functionality, to read new inventories or export data to new formats.



# Statement of need

Emission inventory data can be represented in various formats and resolutions. 
For example, TNO provides an inventory which contains both, a gridded map and point
sources. 
Other inventories, such as Mapluft (the inventory from the city of Zurich) are provided as 
GIS data with various shapes depending on the category of the emission source. As an 
example: 

* Traffic emissions are represented as lines
* Building/industry emissions are represented as point sources
* Private boats on the lake are represented as polygons


Air quality models usually require emission inventories to be in a specific format.
This is often due to the different functionalities these models have. For example,
ICON-ART-OEM requires emissions on the ICON triangular grid. It also requires cyclic time 
profiles for the emissions at daily, weekly and monthly resolutions.
On the other hand, the GRAL model can simulate, line, point or rectangular sources.
GRAL also needs very detailed information about the emissions, such as the height,
the temperature or the gas exit velocity of the source.

Moreover, when modellers design transport simulations, they often are interested in
modifying the inventories. For example, they may want to scale the emissions based on
different scenarios, or they may want to aggregate emissions by sector or pollutant.

`emiproc` has already been successfully applied for different use cases.

* [@acp-24-2759-2024] produced emission files for ICON-ART-OEM based on the EDGARv6
inventory [@edgar_v6].

* [@donmez2024urban] conducted urban climate simulation using emissions produced 
with `emiproc` for cities of zurich and basel.

* [@ponomarev2024estimation] used `emiproc` to nest the zurich city inventory
inside the swiss national inventory around the city boundary

Another python software, `HERMESv3` [@hermesv3_part1], can already process emission 
data and generate input files for athmospheric transport models. However, `HERMESv3` is
relying on specific confiuration files. `emiproc` is more flexible and extensible
as it is designed to
be used directly in python scripts. 


# History

An older version of the `emiproc` package was already published [@gmd-13-2379-2020],
but it was written only for specific models and was not modular enough to easily account
for new inventories and models. `emiproc` was then refactored in 2022 to reach 
the new requirements. This included a major structure change, the addition of new
capabilities, a performance increase, a new documentation and a test coverage.

Since then the package is regularly updated with new features and bug fixes.


# Design 


To be able to use these different kind of inventories in air quality models, it is
necessary to harmonize them. This is what the `emiproc` package is designed for.

![Design of the idea behind emiproc \label{fig:design}.](pipeline.drawio.png)

Thanks to the harmonization of the data, processing functions can be applied to the
different inventories once loaded into emiproc. They can later be exported to any
of the supported formats provided by the package.

The API of `emiproc` makes a great use of object-oriented programming. 

`emiproc` is built on top of ,the `geopandas` package [@kelsey_jordahl_2020_3946761], 
which allows storing the geometries of the emission maps and offers a lot of functionalities
related to geometric operations.
In the inventory, the emission data is stored as 
[`geopandas.GeoDataFrame`](https://geopandas.org/en/stable/docs/reference/geodataframe.html).


## Inventory
The main object
is the `Inventory` object. Any inventory is a subclass of the `Inventory` object.

`emiproc` provides many utility functions to process this object. 
Typically processing function take and inventory and some parameters as input and return a new Inventory object
with the processed data.
Example: the `group_categories` function will take a mapping that converts the categories to a new set of categories. 
This is useful if you want to reduce the number of categories for your simulation and use a more general set of categories such as the GNFR categories from [@emep_guidelines].


## Grid
As inventories and models always come on different grids, `emiproc` uses a `Grid` object.
For many use cases the `RegularGrid` child class can be used,
but for specific models the user can define their own grid. 
Example: The ICON Model [@IconRelease01] is simulated over
an icosahedral (triangular) grid, for which `emiproc` provides an `ICON_grid` object.

Functions in emiproc can then handle this `Grid` objects.
Example: the `remap_inventory` function can be used to remap the emissions to a different grid.
The `remap_inventory` function takes an `Inventory` and a `Grid` as input, and returns a new
`Inventory` containing emissions remapped to the new grid.

## Temporal and vertical profiles

To handle the temporal distribution of the emissions, the `emiproc` package uses the
`TemporalProfile` object which are assigned to the `Inventory`. 
This object stores the temporal distribution of the emissions
and can be used to scale the emissions based on different scenarios. 
Profiles can be either defined at specific datetimes 
or cyclic defined at different temporal resolutions (e.g. daily, weekly, monthly).

Vertical distribution of the emissions is handled in a similar manner by the 
`VerticalProfile` object. 

These profile objects can be assigned very specifically to certain types of emissions.
For example, it is possible to assign them to specific category/polluant/country/gridcell.

## Export functions

Finally, the `emiproc` package contain functions to export the inventory to various
formats used in air quality models. 
These export functions are designed in a way to make life as simple as possible for the
modeller. However some transport model might require some specific data which is not 
given in the inventories. In this case, `emiproc` provides error messages which
guide the modeller into adding the missing data.


## Emissions generation 

In some cases, some emissions are not provided in the inventory. For example, the
human respiration is rarely provided.  However, 
an emission map for this sector can be estimated based on the population density. 
For this purpose, `emiproc` provides a module that helps to calculate these emissions.


Another example is the emissions from vegetation. 
Different models can estimate the emissions from vegetation based on satellite
observations. `emiproc` implements the VPRM model [@vprm].

This topic is one for which the HERMES model is already well developed in 
its bottom-up part [@hermesv3_part2] as they implemented many sectors and polluants.

In the future, we hope users will contribute by adding new emission models.

## Visualizing the data
At the end of the processing, a modeller usually wants to check the output files 
to see if the processing was successful. 

For some of the outputs, emiproc provides example plot scripts based on matplotlib.


For regular grids, the `plot_inventory` function from the `emiproc.plots` module
can be used to plot the emissions on a map.
Figure \ref{fig:tno} was created using this function.



# Availability

The package is availabe on [GitHub](https://github.com/C2SM-RCM/emiproc)
and the documentation is available on [readthedocs](https://emiproc.readthedocs.io/en/latest/).

# Acknowledgements



We acknowledge all the previous and current contributers,
the developpers of all models and inventories included in
`emiproc` and the developpers of the packages used by `emiproc`.

# References