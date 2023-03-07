Preparing emission data for icon-oem
====================================


Load TNO inventory 
------------------

For this examples we are using the TNO inventory.

.. code::
    
    from emiproc.inventories.tno import TNO_Inventory
    inv = TNO_Inventory("TNO_GHGco_v4_0_year2018.nc")

Load the ICON Grid 
------------------

The grid on which we want to remap the tno 

.. code::

    from emiproc.grids import ICONGrid
    icon_grid_file = "icon_Zurich_R19B9_DOM01.nc"
    icon_grid = ICONGrid(icon_grid_file)


Remap the inventory to the grid 
-------------------------------

.. code::

    # Convert to a planar crs before 
    # you will get a warning from pygeos if you dont do that
    from emiproc.grids import WGS84_PROJECTED
    inv.to_crs(WGS84_PROJECTED)
    
    # Does the remapping, the weights are saved in case you need to remap 
    # another time
    remaped_tno = remap_inventory(inv, icon_grid, ".remap_tno2icon")

Export to OEM inputs 
--------------------

Now we need to add the outputs to the netcdf file.

.. code::

    from emiproc.inventories.exports import export_icon_oem

    export_icon_oem(remaped_tno, icon_grid_file, "icon_with_tno_emissions.nc")



.. note:: 
    OEM also requires vertical and temporal profiles.
    This is currently being implemented.