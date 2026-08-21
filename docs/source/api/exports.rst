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

.. autofunction:: emiproc.exports.microhh.export_cams_glob_ant_to_microhh

.. autofunction:: emiproc.exports.microhh.import_and_export_cams_glob_ant

The convenience interface crops the global file before building cell polygons::

    from emiproc.exports.microhh import import_and_export_cams_glob_ant
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
    import_and_export_cams_glob_ant(
        "CAMS-GLOB-ANT_v6.2_methane_2025.nc",
        grid,
        "cams_ch4_microhh.nc",
        variables=["sum"],
        weights_file="cams_to_microhh_weights.npz",
    )

To subtract CORSO point sources before remapping, pass the raw CORSO dataset to
the same convenience function::

    from emiproc.inventories.corso import CorsoEmissions

    corso = CorsoEmissions(
        "/capstor/store/cscs/empa/em05/vhess/input_data/corso"
    ).get_dataset()
    import_and_export_cams_glob_ant(
        "CAMS-GLOB-ANT_v6.2_nitrogen-oxides_2021.nc",
        grid,
        "cams_nox_without_corso.nc",
        corso_dataset=corso,
    )


Gral
----
.. automodule:: emiproc.exports.gral
    :members:
