Grids 
=====

In the file `grids.py` you can find definitions for classes handling common gridtypes
(regular lat/lon grids, rotated lat/lon grids as used by COSMO,
unstructured grids as used by ICON, cartesian grids, etc.). 

Grids are implemented for inventories as well as for models. Grid classes
are named accordingly.

Use them in your configuration file to specify your grid.

If your grid can not be represented by an existing one, implement your own grid class
by inheriting from the `Grid` baseclass and implementing the required methods,
e.g. projections.


Available Grids 
---------------

.. autoclass:: emiproc.grids.RegularGrid

.. autoclass:: emiproc.grids.TNOGrid

.. autoclass:: emiproc.grids.EDGARGrid

.. autoclass:: emiproc.grids.GeoPandasGrid

.. autoclass:: emiproc.grids.VPRMGrid

.. autoclass:: emiproc.grids.SwissGrid

.. autoclass:: emiproc.grids.COSMOGrid

.. autoclass:: emiproc.grids.ICONGrid



