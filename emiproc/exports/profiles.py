from __future__ import annotations
import logging
from os import PathLike
from pathlib import Path
import xarray as xr
import numpy as np
from emiproc.inventories import Inventory
from emiproc.grids import RegularGrid
from emiproc.profiles.utils import get_desired_profile_index
from emiproc.exports.netcdf import NetcdfAttributes
from emiproc.utilities import (
    get_timezone_mask,
)

from emiproc.exports.icon import (
    TemporalProfilesTypes,
    make_icon_time_profiles,
    make_icon_vertical_profiles,
)


def export_inventory_profiles(
    inv: Inventory,
    output_dir: PathLike,
    grid: RegularGrid,
    netcdf_attributes: NetcdfAttributes,
    var_name_format: str = "{substance}_{category}",
    temporal_profiles_type: TemporalProfilesTypes = TemporalProfilesTypes.THREE_CYCLES,
    year: int | None = None,
) -> Path:
    """Export the vertical and temporal profiles of the inventory to netcdfs.

    This will export the vertical and temporal profiles of the inventory to netcdf files.
    The vertical profiles will be exported to a single netcdf file.
    The temporal profiles will be exported according to
    :py:func:`~emiproc.exports.icon.make_icon_time_profiles`.

    :param inv: the inventory to export
    :param output_dir: the path to the output directory
    :param grid: the raster grid to export to.
        This is used to calculate the country shifts for the temporal profiles.
    :param netcdf_attributes: Same as in :py:func:`emiproc.exports.netcdf.export_raster_netcdf`.
    :param var_name_format: Same as in :py:func:`emiproc.exports.netcdf.export_raster_netcdf`.
    :param temporal_profiles_type: The type of temporal profiles to export.
        See :py:class:`emiproc.exports.icon.TemporalProfilesTypes` for the available options.
    :param year: The year of the temporal profiles.
    """
    logger = logging.getLogger("emiproc.export_inventory_profiles")
    output_dir = Path(output_dir)

    if "type" in inv.v_profiles_indexes.coords:
        raise NotImplementedError(
            "Can only export only profiles varying on 'category' and 'substance' . \n"
            "If you want to export inventory profiles, "
            "please use another function or remap your inventory."
        )

    # Add the vertical profiles
    if inv.v_profiles:
        profiles = {
            var_name_format.format(substance=sub, category=cat): inv.v_profiles[
                get_desired_profile_index(inv.v_profiles_indexes, cat=cat, sub=sub)
            ]
            for cat in inv.categories
            for sub in inv.substances
        }
        make_icon_vertical_profiles(
            profiles,
            output_dir,
            nc_attrs=netcdf_attributes,
        )
    else:
        logger.warning(
            f"No vertical profiles found in {inv}, no vertical profiles will be"
            " exported."
        )

    if inv.t_profiles_groups:
        tz_mask = get_timezone_mask(grid)
        profiles = {
            var_name_format.format(substance=sub, category=cat): inv.t_profiles_groups[
                get_desired_profile_index(inv.t_profiles_indexes, cat=cat, sub=sub)
            ]
            for cat in inv.categories
            for sub in inv.substances
        }

        unique_tz, tz_indexes = np.unique(tz_mask, return_inverse=True)

        make_icon_time_profiles(
            time_profiles=profiles,
            time_zones=unique_tz,
            profiles_type=temporal_profiles_type,
            year=year,
            out_dir=output_dir,
            nc_attrs=netcdf_attributes,
        )
        ds_tz_mask = xr.Dataset(
            {
                "tz_mask": (
                    ["lat", "lon"],
                    tz_indexes.reshape(grid.shape).T,
                ),
                "timezones": (
                    ["country_id"],
                    unique_tz,
                ),
            },
            coords={
                "lat": grid.lat_range,
                "lon": grid.lon_range,
            },
            attrs=netcdf_attributes,
        )
        ds_tz_mask.to_netcdf(output_dir / "tz_mask.nc")

    else:
        logger.warning(
            f"No temporal profiles found in {inv}, no temporal profiles will be"
            " exported."
        )
