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
    equal-contrib: true
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Brunner Dominik
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
  - given-names: Ludwig
    dropping-particle: van
    surname: Beethoven
    affiliation: 3
affiliations:
 - name: Swiss Federal Laboratories for Materials Science and Technology (Empa), Switzerland
   index: 1
date: 19 April 2024
bibliography: paper.bib

---

# Summary

In the effort to reduce air pollution and greenhouse gases, emission inventories
are created by countries and regions to estimate which sectors emit how much of which 
substances.
These inventories are
often created in different formats and resolutions, which can make it difficult to compare
them or use them in air quality models. The `emiproc` package is a Python package that
provides tools for processing and harmonizing emission inventories. It includes functions
for reading, writing, and exporting emission inventory data to various formats used 
in air transport models. `emiproc` also includes functions for performing various 
operations on emission inventory data, such as remapping emissions to different spatial
resolutions, aggregating emissions by sector or pollutant, or scaling emissions based on
projection scenarios. The package is designed to be flexible and extensible, allowing
users to easily add new functionality, to read new inventories or export data to new formats.


INCLUDE AN IMAGE OF AN INVENOTRY

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }


# Statement of need

Emission inventory data can be represented in various formats and resolutions. 
For example, TNO provides an invenotry which is splitted with one part which is 
regulary gridded and another part containing point sources. 
Other inventories, such as Mapluft (the city invenotry of Zurich) are provided as 
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

Moreover, when a modeller designs a transport simulation, they often are interested in
modifying the inventories. For example, they may want to scale the emissions based on
different scenarios, or they may want to aggregate emissions by sector or pollutant.
(SHOULD WE CITE SOME PAPERS FOR EXAMPLES).
Nikolai used `emiproc` to fill the space around the city of zurich. 

# History

An older version of the `emiproc` package was already published [@gmd-13-2379-2020],
but it was written only for specific models and was not modular enough to easily account
for other inventories and models. `emiproc` was then refactored in 2022 to reach 
the new requirements. This included a major structure change, the addition of new
capabilities, a performance increase, a new documentation and a test coverage.


# Design 


To be able to use these different kind of inventories in air quality models, it is
necessary to harmonize them. This is what the `emiproc` package is designed for.

Thanks to the harmonization of the data, processing functions can be applied to the
different inventories once loaded into emiproc. 

The API of `emiproc` makes a great use of object-oriented programming. The main object
is the `Inventory` object. Any invenotory is a subclass of the `Inventory` object.

The utility functions can process this Inventory object. And return a new Inventory object
with the processed data.

For example, the `remap` function can be used to remap the emissions to a different grid.
The `remap` function takes an Inventory object and a grid as input, and returns a new
Inventory object with the emissions remapped to the new grid.

`empiproc` is built on top of ,the `geopandas` package [@kelsey_jordahl_2020_3946761], 
which allows storing the geometries of the emission maps and offers a lot of functionalities
to geometrically process the data.
In the inventory, the emission data is stored as 
[`GeoDataFrame`](https://geopandas.org/en/stable/docs/reference/geodataframe.html)
from .


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

Finally, the `emiproc` package contain functions to export the invenotry to various
formats used in air quality models. 
These export functions are designed in a way to make life as simple as possible for the
modeller. However some transport model might require some specific data which is not 
given in the inventories. In this case, `emiproc` provides error messages which
guide the modeller into adding the missing data.


# Acknowledgements

We acknowledge all the previous contributers to the `emiproc` package,
as well as the developpers of all packages, models and inventories
used in the `emiproc` package.

# References