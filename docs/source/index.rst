.. emiproc documentation master file, created by
   sphinx-quickstart on Mon Oct  3 10:41:41 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

emiproc documentation
=====================

emiproc is a python package for processing emission inventories, generating
input files for athmospheric transport models.

It can handle gridded annual emissions inventories, as well as hourly defined
data.


.. image:: diagrams/pipeline.drawio.svg

Features 
--------

* Support of different inventories
* Exporting to different formats (icon-art, cosmo-art, netcdf rasters)
* Spatially merging inventories around a region
* Separate point source handling
* Categories/Substance selection
* Groupping categories
* Plotting the inventory
* Conservative spatial regridding

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
