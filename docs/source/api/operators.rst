Operators
=========


Inventory Operators
-------------------

.. autofunction:: emiproc.inventories.utils.add_inventories

.. autofunction:: emiproc.inventories.utils.combine_inventories

.. autofunction:: emiproc.inventories.utils.subtract_inventories

.. autofunction:: emiproc.inventories.utils.merge_inventories_without_double_counting

.. autofunction:: emiproc.inventories.utils.scale_inventory

.. autofunction:: emiproc.inventories.utils.get_total_emissions

.. autofunction:: emiproc.inventories.utils.drop



Avoiding double counting
------------------------

Two inventories often describe the same emissions, for example a global
gridded inventory and a database of the large point sources it contains.
Adding them with :py:func:`~emiproc.inventories.utils.add_inventories` would
count those sources twice.

:py:func:`~emiproc.inventories.utils.subtract_inventories` removes the
emissions of one inventory from another. The inventory to remove is remapped
onto the grid of the target first, so point, shaped and gridded sources are all
supported::

    from emiproc.inventories.utils import subtract_inventories

    only_the_rest = subtract_inventories(
        gridded_inv,
        point_sources_inv,
        category_mapping={"power": "ene"},
    )

:py:func:`~emiproc.inventories.utils.merge_inventories_without_double_counting`
does the whole job: it subtracts the detailed inventory and then adds it back
at its own resolution, so the total emissions are unchanged while the detailed
sources keep their geometry and their profiles::

    from emiproc.inventories.utils import (
        merge_inventories_without_double_counting,
    )

    combined = merge_inventories_without_double_counting(
        gridded_inv,
        point_sources_inv,
        category_mapping={"power": "ene"},
    )

Both are not aggregative but they do read the emission values, so apply them
**before** :py:func:`~emiproc.inventories.utils.group_categories` or
:py:func:`~emiproc.regrid.remap_inventory`, while the categories of the two
inventories can still be matched.

The subtraction works on the annual emissions of
:py:attr:`~emiproc.inventories.Inventory.gdf`, so the annual budget is exact,
but the emissions removed from each time step follow the temporal profile of
the target inventory. When the detailed sources exceed what the target holds in
a cell, ``negative_policy`` decides whether the residual is clipped to zero
(the default, with a warning), rejected, or kept negative. The returned
inventory carries a ``subtraction_summary`` comparing what was requested with
what was actually removed, which is the way to check that clipping did not
silently lose emissions.

Profiles Operators 
------------------

A set of operators to manipulate profiles in inventories.


.. autofunction:: emiproc.inventories.utils.country_to_cells

.. autofunction:: emiproc.inventories.utils.interpolate_temporal_profiles

Geometric Transformations 
-------------------------

.. autofunction:: emiproc.inventories.utils.crop_with_shape

.. autofunction:: emiproc.inventories.utils.clip_box

.. autofunction:: emiproc.regrid.remap_inventory