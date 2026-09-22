Inventories
===========


Inventory Class
---------------

.. autoclass:: emiproc.inventories.Inventory
    :members:


Available Inventories 
---------------------

.. autoclass:: emiproc.inventories.tno.TNO_Inventory
    :special-members: __init__

.. autofunction:: emiproc.inventories.tno.read_tno_gridded_profiles

.. autoclass:: emiproc.inventories.edgar.EDGARv8
    :special-members: __init__

.. autofunction:: emiproc.inventories.edgar.download_edgar_files

.. autoclass:: emiproc.inventories.swiss.SwissRasters
    :special-members: __init__

.. autofunction:: emiproc.inventories.swiss.read_prtr

.. autoclass:: emiproc.inventories.zurich.MapLuftZurich
    :special-members: __init__

.. autoclass:: emiproc.inventories.zurich.duck.DuckDBInventory
    :special-members: __init__

.. autoclass:: emiproc.inventories.netcdf_raster.NetcdfRaster
    :special-members: __init__

.. autoclass:: emiproc.inventories.gfas.GFAS_Inventory
    :special-members: __init__

.. autoclass:: emiproc.inventories.gfed.GFED4_Inventory
    :special-members: __init__

.. autoclass:: emiproc.inventories.gfed.GFEDv51
    :special-members: __init__

.. autofunction:: emiproc.inventories.gfed.download_gfedv51

.. autoclass:: emiproc.inventories.lpjguess.LPJ_GUESS_Inventory
    :special-members: __init__

.. autoclass:: emiproc.inventories.saunois.Saunois
    :special-members: __init__

.. autoclass:: emiproc.inventories.gral.GralInventory
    :special-members: __init__

.. autoclass:: emiproc.inventories.cams_reg_aq.CAMS_REG_AQ
    :special-members: __init__

.. autoclass:: emiproc.inventories.cams_global.CAMSGlobal
    :special-members: __init__

.. autoclass:: emiproc.inventories.corso.CorsoInventory
    :special-members: __init__

.. autoclass:: emiproc.inventories.corso.CorsoEmissions
    :members:

:py:class:`~emiproc.inventories.corso.CorsoInventory` reads the CORSO
catalogue as a regular emiproc inventory: the sources are stored as points in
:py:attr:`~emiproc.inventories.Inventory.gdfs` in ``kg/y``, one category per
CORSO sector, together with their temporal and vertical profiles. It can
therefore be used with every emiproc operator and exporter.
:py:class:`~emiproc.inventories.corso.CorsoEmissions` remains available as the
lower level reader for the individual CSV files and per-source time series.

Because a global inventory such as
:py:class:`~emiproc.inventories.cams_global.CAMSGlobal` already contains the
emissions of those point sources, combining the two naively would count them
twice. Use
:py:func:`~emiproc.inventories.utils.merge_inventories_without_double_counting`
to remove the point sources from the grid cells that contain them and add them
back at their own resolution::

    from emiproc.inventories.cams_global import (
        CAMSGlobal,
        DEFAULT_CORSO_SECTOR_MAPPING,
        DEFAULT_CORSO_SUBSTANCE_MAPPING,
    )
    from emiproc.inventories.corso import CorsoInventory
    from emiproc.inventories.utils import (
        merge_inventories_without_double_counting,
    )

    cams = CAMSGlobal("CAMS-GLOB-ANT_v6.2_nitrogen-oxides_2021.nc")
    corso = CorsoInventory(
        "/capstor/store/cscs/empa/em05/vhess/input_data/corso"
    )
    inventory = merge_inventories_without_double_counting(
        cams,
        corso,
        category_mapping=DEFAULT_CORSO_SECTOR_MAPPING,
        substance_mapping=DEFAULT_CORSO_SUBSTANCE_MAPPING,
    )

By default CORSO sectors map to ``power`` -> ``ene``, ``cement`` and
``iron&steel`` -> ``ind`` and ``syntheticfuel`` -> ``ref``. Pass your own
``category_mapping`` to override them. If the point sources exceed the coarse
emissions of a cell, the residual is clipped to zero and a warning is issued;
use ``negative_policy="raise"`` for strict validation or ``"allow"`` to keep
negative values. The returned inventory carries a ``subtraction_summary``
reporting what was requested, what was actually removed and how much was
clipped.

Use :py:func:`~emiproc.inventories.utils.subtract_inventories` directly when
you only want the coarse inventory with the detailed sources removed.

.. autoclass:: emiproc.inventories.wetcharts.WetCHARTs
    :special-members: __init__
