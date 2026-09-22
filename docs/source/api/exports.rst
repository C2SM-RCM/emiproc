Exporting 
=========

.. autofunction:: emiproc.exports.netcdf.nc_cf_attributes

.. autofunction:: emiproc.exports.icon.export_icon_oem

.. autofunction:: emiproc.exports.icon.make_icon_time_profiles

.. autofunction:: emiproc.exports.icon.make_icon_vertical_profiles

.. autoenum:: emiproc.exports.icon.TemporalProfilesTypes
    
.. autofunction:: emiproc.exports.rasters.export_raster_netcdf

.. autofunction:: emiproc.exports.profiles.export_inventory_profiles

.. autofunction:: emiproc.exports.utils.get_temporally_scaled_array

.. autofunction:: emiproc.exports.wrf.export_wrf_hourly_emissions

.. autofunction:: emiproc.exports.fluxie.export_fluxie

.. autofunction:: emiproc.exports.geopackage.export_to_geopackage

MicroHH
-------

.. autofunction:: emiproc.exports.microhh.export_to_microhh

.. autofunction:: emiproc.exports.microhh.import_and_export_cams_global

:py:func:`~emiproc.exports.microhh.export_to_microhh` accepts **any**
inventory. It remaps it conservatively on the MicroHH grid and uses its
temporal profiles to resolve the emissions in time, writing one
``emi_<substance>`` variable with dimensions ``(time, category, y, x)`` in
``kg m-2 s-1``. Inventories without temporal profiles are written as constant::

    from emiproc.exports.microhh import export_to_microhh
    from emiproc.grids import MicroHHGrid

    export_to_microhh(inventory, grid, "emissions_microhh.nc")

``emiproc.exports.microhh.export_cams_global_to_microhh`` is kept as a
deprecated alias of it.

The convenience interface crops the global file before building cell polygons::

    from emiproc.exports.microhh import import_and_export_cams_global
    from emiproc.grids import MicroHHGrid

    grid = MicroHHGrid(
        xsize=20_000,
        ysize=20_000,
        itot=400,
        jtot=400,
        lon=8.54,
        lat=47.38,
        anchor="center",
        crs="EPSG:32632",
    )
    import_and_export_cams_global(
        "CAMS-GLOB-ANT_v6.2_methane_2025.nc",
        grid,
        "cams_ch4_microhh.nc",
        variables=["sum"],
        weights_file="cams_to_microhh_weights.npz",
    )

Multiple compatible files can be summed before remapping by passing a list.
For every file, ``sum`` is used when present. If it is absent, the dataset must
contain exactly one emission-flux field with ``time``, ``lat``, and ``lon``
dimensions and supported units. Ancillary data fields are ignored; multiple
emission candidates cause an error::

    import_and_export_cams_global(
        [
            "cams_nox_part_1_2021.nc",
            "cams_nox_part_2_2021.nc",
        ],
        grid,
        "summed_cams_nox_microhh.nc",
    )

All files must have matching years and selected time intervals. Inputs on
different grids are conservatively remapped to the finest input CAMS grid
before summation. The ``variables`` argument is not accepted with list input
because selection is performed independently using the rule above.

Use a :class:`datetime.datetime` to combine monthly inventories with daily
shipping data, or to load a daily oceanic inventory. The applicable monthly
value and the exact daily value are selected before summation::

    from datetime import datetime

    import_and_export_cams_global(
        [
            "CAMS-GLOB-ANT_v6.2_nitrogen-oxides_2021.nc",
            "CAMS-GLOB-SHIP_v3.2_nitrogen-oxides_2021.nc",
        ],
        grid,
        "cams_nox_2021-06-15.nc",
        date=datetime(2021, 6, 15, 12, 30),
    )

Daily inventories require ``date``. The selected output contains one time
step. Oceanic inventories with non-equidistant latitude spacing are handled as
rectilinear grids.

CAMS substance aliases are normalized before compatibility checks. In
particular, ``CO2_s_c``, ``CO2_ex_s_c``, and ``Carbon dioxide`` are combined as
``CO2``; ``CO`` and ``Carbon monoxide`` are combined as ``CO``. Their exported
variables are therefore named ``emi_CO2`` and ``emi_CO``.

VOC labels including ``NMVOCs``, ``Other-VOCs``, and ``Volatile organic
compounds (all)`` are resolved to ``VOCs``. When an explicit list also contains
isoprene or monoterpene component files, those are treated as members of the
requested VOC aggregate. Emiproc sums exactly the files supplied and cannot
detect chemical double counting, so the caller remains responsible for the
composition of that list. The output records the selected field and original
substance label for every input in ``input_details``.

To remove point sources that the CAMS files already contain, pass an inventory
of them. With list input, the CAMS files are summed first and the point sources
are subtracted once from that sum::

    from emiproc.inventories.cams_global import DEFAULT_CORSO_SECTOR_MAPPING
    from emiproc.inventories.corso import CorsoInventory

    corso = CorsoInventory(
        "/capstor/store/cscs/empa/em05/vhess/input_data/corso"
    )
    import_and_export_cams_global(
        "CAMS-GLOB-ANT_v6.2_nitrogen-oxides_2021.nc",
        grid,
        "cams_nox_without_corso.nc",
        point_sources=corso,
        category_mapping=DEFAULT_CORSO_SECTOR_MAPPING,
    )

Pass ``keep_point_sources=True`` to add the sources back at their own
resolution instead of only exporting the remainder, which conserves the total
emissions. The written file records what was removed in its
``double_counting_*`` attributes.

The subtraction is done on the annual emissions, so the annual budget is
exact while the emissions removed from each time step follow the temporal
profile of the CAMS inventory. See
:py:func:`~emiproc.inventories.utils.subtract_inventories`.

The current CORSO catalogue has no VOC field, so a VOC aggregate has no point
sources to subtract.


Gral
----
.. automodule:: emiproc.exports.gral
    :members:
