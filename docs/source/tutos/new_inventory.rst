Implementing a new Inventory
============================

If you want to implement a new inventory, you can take as an example
the tno inventory which has an easy to understand implementation.

Inherit from the base class 
---------------------------

If you want to include a new inventory in emiproc. 
You should first create a class that inherits from 
:py:class:`~emiproc.inventories.Inventory` .


Create __init__ of the class 
----------------------------

Make sure your class is as easy as possible to use, such that 
for example you only need to specify the paths of the files.

If you have some parameter to choose (ex. Year of Inventory)
define them in the :py:meth:`__init__` method.


Fill data into the geodataframe 
-------------------------------

There are 2 possible ways of specifing emissions:
1. give emissions values for categories and substances on a grid 
2. give custom geometries (e.g. point sources) that emit some substance/category

emiproc can handle both simulatenously but it has to follow this pattern:

1. gridded emissions will be stored in a :py:mod:`geopandas.GeoDataFrame`  called `gdf`
2. gridded emissions will be stored in a dictionary 
   mapping categories to :py:mod:`geopandas.GeoDataFrame`


gdf 
^^^

Contains information on the grid.
The geometry column contains all the grid cells of the grid.

Other columns contain values for the emissions for each category/substance in that grid.
Columns are mutliindex column, with 
the first line containing the categories and second line 
the substnces.


.. image::
    ../../images/gdf_head.png

The len of gdf is the number of cells in the grid.

gdfs
^^^^
This is a python dict mapping the name of the category
to geodataframes.

For each dataframe, 
the geometry column contains all the geometries from that category.

Other columns contain values for the emissions for each substance in that categories sources.

.. image::
    ../../images/gdfs_head.png

The len of a gdfs is the number of source of the gdfs's category.


Add crs information
-------------------

When you create the gdf and gdfs, make sure you add 
the information about the crs directly in the gdf and gdfs.

Think about the Units
---------------------

By convention emiproc uses units :math:`\frac{kg}{y}` .
In particular, every emission value in the gdf and gdfs means
kg/y per geometry (== per grid cell in gdf) .

Some export function will then convert automatically to the 
unit required when saving to file.