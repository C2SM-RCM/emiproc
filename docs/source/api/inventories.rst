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

.. autoclass:: emiproc.inventories.lpjguess.LPJ_GUESS_Inventory
    :special-members: __init__

.. autoclass:: emiproc.inventories.saunois.SaunoisInventory
    :special-members: __init__

.. autoclass:: emiproc.inventories.gral.GralInventory
    :special-members: __init__

.. autoclass:: emiproc.inventories.cams_reg_aq.CAMS_REG_AQ
    :special-members: __init__

.. autoclass:: emiproc.inventories.cams_global.CAMSGlobal
    :special-members: __init__

.. autoclass:: emiproc.inventories.corso.CorsoEmissions
    :members:

CORSO point sources can optionally be removed from their corresponding CAMS
grid cells. Pass the raw dataset returned by the processing-chain importer::

    from emiproc.inventories.corso import CorsoEmissions
    from emiproc.inventories.cams_global import CAMSGlobal

    corso = CorsoEmissions(
        "/capstor/store/cscs/empa/em05/vhess/input_data/corso"
    ).get_dataset()
    inventory = CAMSGlobal(
        "CAMS-GLOB-ANT_v6.2_nitrogen-oxides_2021.nc",
        corso_dataset=corso,
    )

The ``*_kty`` values are interpreted as annual totals in kilotonnes per year,
converted to annual-mean ``kg s-1``, divided by the CAMS cell area, and removed
from every monthly mean. By default, CORSO sectors are mapped as ``power`` to
``ene``, ``cement`` and ``iron&steel`` to ``ind``, and ``syntheticfuel`` to
``ref``. Pass ``corso_sector_mapping`` to override these assignments. If a
CORSO source is larger than CAMS in a month, the residual is clipped to zero;
use ``corso_negative_policy="raise"`` for strict validation or ``"allow"`` to
retain negative values.

.. autoclass:: emiproc.inventories.wetcharts.WetCHARTs
    :special-members: __init__
