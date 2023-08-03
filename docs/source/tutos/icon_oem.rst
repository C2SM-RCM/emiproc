Preparing emission data for ICON-OEM
====================================

This tutorial explains how to create emissions files that can be used in 
ICON-OEM.

Documentation for ICON-ART can be found here:
https://www.icon-art.kit.edu/userguide/index.php


Load the inventory 
------------------

For this examples we are using the TNO inventory.

.. code::
    
    from emiproc.inventories.tno import TNO_Inventory
    inv = TNO_Inventory("TNO_GHGco_v4_0_year2018.nc")


Make sure your inventory has some temporal and vertical profiles.
These are needed for OEM.

If you don't, you can set some profiles using
:py:meth:`~emiproc.inventories.Inventory.set_profile`.

You can also use :py:func:`~emiproc.exports.icon.get_constant_time_profile` to 
assign a constant profile in time .

Load the ICON Grid 
------------------

You will need to specify the grid on which to remap the emissions.

.. code::

    from emiproc.grids import ICONGrid
    icon_grid_file = "icon_Zurich_R19B9_DOM01.nc"
    icon_grid = ICONGrid(icon_grid_file)


Remap the inventory to the grid 
-------------------------------

.. code::

    # Convert to a planar crs before 
    from emiproc.grids import WGS84_PROJECTED
    inv.to_crs(WGS84_PROJECTED)
    
    # Does the remapping, returning an inventory on the ICONGrid
    remaped_tno = remap_inventory(inv, icon_grid)

Export to OEM inputs 
--------------------

The function :py:func:`emiproc.exports.icon.export_icon_oem` will generate 
the files needed for OEM in the output_dir.
You can then add this files in your namelist and add the tracers in the xml files.

Note that you have to options for the Temporal Profiles in OEM:

* Three cycles (hour of day, day of week, month of year)
* Hour of year cycle. In this case you will need to set for which year you want
  the profiles.

.. code::

    from emiproc.inventories.exports import export_icon_oem, TemporalProfilesTypes

    export_icon_oem(
        inv=remaped_tno,
        icon_grid_file=icon_grid_file,
        output_dir=output_dir,
        temporal_profiles_type=TemporalProfilesTypes.THREE_CYCLES,
        # Following parameters are for HOUR_OF_YEAR profiles
        #temporal_profiles_type=TemporalProfilesTypes.HOUR_OF_YEAR,
        #year=2022,
    )


Plot your emissions 
-------------------

At the end you can check how your emissions look on the ICON grid.
For that you can use the python script
`plot_icon.py <https://github.com/C2SM-RCM/emiproc/blob/master/scripts/plot_icon.py>`_.