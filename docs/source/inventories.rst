Inventories
===========

There are already several popular inventories compatible with emiproc.

Some of them are open source and freely available,
others are proprietary and have to be obtained from the respective data provider.

Make sure you follow the rules defined by the data provider 
regarding usage and distribution of the data.


Available Inventory Objects 
---------------------------

TNO
^^^ 

:py:class:`emiproc.inventories.tno.TNO_Inventory`


EDGAR
^^^^^

:py:class:`emiproc.inventories.edgar.EDGARv8`

Swiss Rasterized Inventory 
^^^^^^^^^^^^^^^^^^^^^^^^^^

:py:class:`emiproc.inventories.swiss.SwissRasters`

MapLuft Zurich
^^^^^^^^^^^^^^

:py:class:`emiproc.inventories.zurich.MapLuftZurich`


GFAS 
^^^^

:py:class:`emiproc.inventories.gfas.GFAS_Inventory`


GFED
^^^^

:py:class:`emiproc.inventories.gfed.GFED4_Inventory`


LPJ-GUESS
^^^^^^^^^

:py:class:`emiproc.inventories.lpjguess.LPJ_GUESS_Inventory`


Saunois
^^^^^^^

:py:class:`emiproc.inventories.saunois.SaunoisInventory`

CAMS Regional Air Quality
^^^^^^^^^^^^^^^^^^^^^^^^^

:py:class:`emiproc.inventories.cams_reg_aq.CAMS_REG_AQ`


WetCHARTs
^^^^^^^^^

:py:class:`emiproc.inventories.wetcharts.WetCHARTs`


Grids 
-----

All inventories are defined on a grid.

In the file `grids.py` you can find definitions for classes handling common gridtypes
(regular lat/lon grids, rotated lat/lon grids as used in TNO,
unstructured grids as used by ICON, cartesian grids, etc.). 

Grids are implemented for inventories as well as for models. Grid classes
are named accordingly.


If your grid can not be represented by an existing one, implement your own grid class
by inheriting from the :py:class:`emiproc.grids.Grid` baseclass and implementing the required methods,
e.g. projections.