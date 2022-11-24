.. emiproc documentation master file, created by
   sphinx-quickstart on Mon Oct  3 10:41:41 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

emiproc documentation
=====================

emiproc is a python package to generate emission input files for atmospheric 
transport models.

It can process inventories available in different formats and on different
grids and generate input files for models operating on any type of grid
including complex unstructured grids. It has been designed specifically
for the models COSMO-ART and ICON-ART but can readily be adapted for other
models. The output of emiproc is a set of netcdf files with emissions
mapped onto the model grid. There are two options:

* Single emission file with sector-specific annual mean emissions and additional files describing temporal and vertical profiles and, in case of reactive gases like VOCs, speciation profiles.
* Separate emission files for each hour of a given time period.


.. image:: diagrams/pipeline.drawio.svg

Features 
--------

* Support of multiple inventories like EDGAR, TNO-CAMS
* Conservative spatial regridding
* Exporting to different formats (icon-art, cosmo-art, netcdf rasters)
* Spatially merging inventories around a region
* Separate handling of point sources if desired
* Categories/Substance selection
* Re-grouping of emission categories
* Visualization of the output

Contents 
--------

.. toctree::
   :maxdepth: 2

   installation
   inventories
   grids
   api
   tutos/tutorials
   support



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
