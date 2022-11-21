Grids 
=====

In the file `grids.py` you can find definitions for classes handling common gridtypes
(COSMO, TNO, swiss). Use them in your configuration file to specify your grid.

Grids are implemented for inventories as well as for models.


If your grid can not be represented by an existing one, implement your own grid class
by inheriting from the `Grid` baseclass and implementing the required methods.


Available Grids 
---------------

.. autoclass:: emiproc.grids.LatLonNcGrid

.. autoclass:: emiproc.grids.TNOGrid

.. autoclass:: emiproc.grids.EDGARGrid

.. autoclass:: emiproc.grids.GeoPandasGrid

.. autoclass:: emiproc.grids.VPRMGrid

.. autoclass:: emiproc.grids.SwissGrid

.. autoclass:: emiproc.grids.COSMOGrid

.. autoclass:: emiproc.grids.ICONGrid



