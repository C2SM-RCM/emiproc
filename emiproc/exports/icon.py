from __future__ import annotations
from datetime import datetime, timedelta
from enum import Enum, auto
import logging
from os import PathLike
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import xarray as xr
import numpy as np
from emiproc.exports.netcdf import DEFAULT_NC_ATTRIBUTES
from emiproc.grids import ICONGrid
from emiproc.inventories import Inventory
from emiproc.profiles.temporal.profiles import (
    DailyProfile,
    MounthsProfile,
    TemporalProfile,
    WeeklyProfile,
    HourOfYearProfile,
    HourOfLeapYearProfile,
)
from emiproc.profiles.temporal.operators import create_scaling_factors_time_serie
from emiproc.profiles.vertical_profiles import (
    VerticalProfile,
    VerticalProfiles,
    resample_vertical_profiles,
)
from emiproc.utilities import (
    SEC_PER_HOUR,
    SEC_PER_YR,
    get_timezone_mask,
)
from emiproc.profiles.utils import get_desired_profile_index


class TemporalProfilesTypes(Enum):
    """Possible temporal profiles for OEM.

    :param HOUR_OF_YEAR:  Every hour gets a scaling factor
    :param THREE_CYCLES:  Three cycles (hour of day, day of week, month of year)
    """

    HOUR_OF_YEAR = 0
    THREE_CYCLES = 3


def get_constant_time_profile(
    type: TemporalProfilesTypes = TemporalProfilesTypes.THREE_CYCLES,
    year: int | None = None,
) -> list[TemporalProfile]:
    """Get a constant time profile compatible with ICON-OEM.

    Emits the same at every time.

    Contains three profiles: hour of day, day of week, month of year.

    """
    if type == TemporalProfilesTypes.THREE_CYCLES:
        return [
            DailyProfile(),  # Hour of day
            WeeklyProfile(),  # Day of week
            MounthsProfile(),  # Month of year
        ]
    elif type == TemporalProfilesTypes.HOUR_OF_YEAR:
        if year is None:
            raise ValueError("You must provide a year for the HOUR_OF_YEAR option.")
        # Check if it is a leap year
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            return [HourOfLeapYearProfile()]
        else:
            return [HourOfYearProfile()]
    else:
        raise NotImplementedError(f"{type} is not implemented.")


def export_icon_oem(
    inv: Inventory,
    icon_grid_file: PathLike,
    output_dir: PathLike,
    group_dict: dict[str, list[str]] = {},
    temporal_profiles_type: TemporalProfilesTypes = TemporalProfilesTypes.THREE_CYCLES,
    year: int | None = None,
    nc_attributes: dict[str, str] = DEFAULT_NC_ATTRIBUTES,
    substances: list[str] | None = None,
):
    """Export to a netcdf file for ICON OEM.

    The inventory should have already been remapped to the
    :py:class:`emiproc.grids.ICONGrid` .

    For ICON-OEM you will need to add in the ICON namelist the path the
    files produced by this module::


        ! oem_nml: online emission module ---------------------------------------------
        &oemctrl_nml
        gridded_emissions_nc        =   '${OEMDIR}/oem_gridded_emissions.nc'
        vertical_profile_nc         =   '${OEMDIR}/vertical_profiles.nc'
        hour_of_day_nc              =   '${OEMDIR}/hourofday.nc'
        day_of_week_nc              =   '${OEMDIR}/dayofweek.nc'
        month_of_year_nc            =   '${OEMDIR}/monthofyear.nc'
        ! If you use the hour of year profile, use this instead of the three above
        ! hour_of_year_nc             =   '${OEMDIR}/hourofyear.nc'
        /


    Values will be converted from emiproc units: `kg/y` to
    OEM units  `kg/m2/s` .
    The grid cell area given in the icon grid file is used for this conversion,
    and 365.25 days per year.

    Temporal profiles are adapted to the different countries present in the data.
    Shifts for local time are applied to the countries individually.
    Grid cells are assigned to a country using the timezone mask from
    :py:func:`emiproc.utilities.get_timezone_mask` and country ids are
    set as the `country_id` attribute of the output NetCDF file.

    :arg inv: The inventory to export.
    :arg icon_grid_file: The icon grid file.
    :arg output_dir: The output directory.
    :arg group_dict: If you groupped some categories, you can optionally
        add the groupping in the metadata.
    :arg country_resolution: The resolution
        can be either '10m', '50m' or '110m'
    :arg temporal_profiles_type: The type of temporal profiles to use.
        Can be either :py:class:`~emiproc.exports.icon.TemporalProfilesTypes.HOUR_OF_YEAR`
        or :py:class:`~emiproc.exports.icon.TemporalProfilesTypes.THREE_CYCLES`
    :arg year: The year to use for the temporal profiles. This is mandatory
        only with `temporal_profiles_type` set to
        :py:class:`~emiproc.exports.icon.TemporalProfilesTypes.HOUR_OF_YEAR`
    :arg nc_attributes: The attributes to add to the netcdf file.
    :arg substances: The substances to export. If None, all substances of the inv.

    """
    logger = logging.getLogger("emiproc.export_icon_oem")

    icon_grid_file = Path(icon_grid_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    # Load the output xarray

    ds_out: xr.Dataset = xr.load_dataset(icon_grid_file)
    time_profiles: dict[str, list[TemporalProfile]] = {}
    vertical_profiles: dict[str, VerticalProfile] = {}

    # Check that the inventory has the same amount of cells
    # as the icon grid
    if len(inv.gdf) != ds_out["cell"].size:
        raise ValueError(
            f"The inventory has {len(inv.gdf)} cells, but the icon grid has"
            f" {ds_out['cell'].size} cells."
        )

    for categorie, sub in inv._gdf_columns:
        if substances is not None and sub not in substances:
            continue
        name = f"{categorie}-{sub}"

        # Convert from kg/year to kg/m2/s
        emissions = (
            inv.gdf[(categorie, sub)].to_numpy() / ds_out["cell_area"] / SEC_PER_YR
        )

        attributes = {
            "units": "kg/m2/s",
            "standard_name": name,
            "long_name": f"Emission of {sub} from {categorie}",
            "created_by_emiproc": f"{datetime.now()}",
        }
        if group_dict:
            attributes["group_made_from"] = f"{group_dict[categorie]}"

        emission_with_metadata = emissions.assign_attrs(attributes)

        ds_out = ds_out.assign({name: emission_with_metadata})

        if inv.v_profiles is not None:
            profile_index = get_desired_profile_index(
                inv.v_profiles_indexes, cat=categorie, sub=sub, type="gridded"
            )
            vertical_profiles[name] = inv.v_profiles[profile_index]

        if inv.t_profiles_groups is not None:
            profile_index = get_desired_profile_index(
                inv.t_profiles_indexes, cat=categorie, sub=sub, type="gridded"
            )
            time_profiles[name] = inv.t_profiles_groups[profile_index]

    # Find the proper country codes
    mask_file = (output_dir / f".emiproc_tz_mask_{icon_grid_file.stem}").with_suffix(
        ".npy"
    )

    tz_mask = None

    if mask_file.is_file():
        tz_mask = np.load(mask_file)
    else:
        icon_grid = ICONGrid(icon_grid_file)
        tz_mask = get_timezone_mask(icon_grid)
        np.save(mask_file, tz_mask)

    # Get the countries as integers
    unique_timezones, tz_mask_int = np.unique(tz_mask.reshape(-1), return_inverse=True)

    # Save the profiles
    if time_profiles:
        make_icon_time_profiles(
            time_profiles=time_profiles,
            time_zones=unique_timezones,
            profiles_type=temporal_profiles_type,
            year=year,
            out_dir=output_dir,
            nc_attrs=nc_attributes,
        )
    if vertical_profiles:
        make_icon_vertical_profiles(
            vertical_profiles, out_dir=output_dir, nc_attrs=nc_attributes
        )

    # Add the country ids variable for oem
    ds_out = ds_out.assign(
        {
            "country_ids": (
                "cell",
                tz_mask_int,
                {
                    "standard_name": "country_ids",
                    "long_name": "Timezone of the cell",
                    "history": f"Added by emiproc",
                    "country_resolution": f"country_resolution",
                },
            ),
            "timezone_of_country": (
                "timezones",
                unique_timezones,
                {
                    "standard_name": "timezone_of_country",
                    "long_name": "Timezone corresponding the country id",
                    "history": f"Added by emiproc",
                },
            ),
        }
    )
    # Save the emissions
    ds_out.to_netcdf(output_dir / "oem_gridded_emissions.nc")

    logger.info(f"Exported inventory to {output_dir}.")


def make_icon_time_profiles(
    time_profiles: dict[str, list[TemporalProfile]],
    time_zones: list[str],
    profiles_type: TemporalProfilesTypes = TemporalProfilesTypes.THREE_CYCLES,
    year: int | None = None,
    out_dir: PathLike | None = None,
    nc_attrs: dict[str, str] = DEFAULT_NC_ATTRIBUTES,
) -> dict[str, xr.Dataset]:
    """Make the profiles in the format for icon oem.

    :arg time_profiles: A dictionary with
        the names of variables in the file as keys and
        the profiles as values.
    :arg time_zones: A list with valid timezones names that have to be
        included.
    :arg profiles_type: The type of profiles to use.
    :arg year: Used for the HOUR_OF_YEAR option.
    :arg out_dir: The directory where to export the files.
        If None, the files are not saved and dataset returned.


    .. note::
        OEM can differentiate profiles based on the grid cell.
        It tries to group grid cells in what it calls "countries".
        This should not be mixed with the real countries.
        The countries identifiers should match between the emission files.

    .. warning::
        Currently the same profiles are used for all the countries.
        Only the shifts are different.
    """

    date_of_shift = datetime(
        year=year if year is not None else datetime.now().year, month=1, day=1
    )
    shifts = np.array(
        [
            # Calculate the country shifts for the grid cells
            int(ZoneInfo(code).utcoffset(date_of_shift).seconds / SEC_PER_HOUR)
            for code in time_zones
        ]
    )
    max_shift = int(max(np.abs(shifts)))

    profiles_type = TemporalProfilesTypes(profiles_type)

    if profiles_type == TemporalProfilesTypes.THREE_CYCLES:
        nc_attrs["title"] = "Hour of day profiles"
        hourofday = xr.Dataset(attrs=nc_attrs.copy())
        nc_attrs["title"] = "Day of week profiles"
        dayofweek = xr.Dataset(attrs=nc_attrs.copy())
        nc_attrs["title"] = "Month of year profiles"
        monthofyear = xr.Dataset(attrs=nc_attrs.copy())
    else:
        nc_attrs["title"] = "Hour of year profiles"
        hourofyear = xr.Dataset(attrs=nc_attrs)

    var_metadata = lambda var_name, profile_name: {
        "units": "1",
        "long_name": f"{profile_name} scaling factors for {var_name}",
    }

    for key in time_profiles:
        if profiles_type == TemporalProfilesTypes.THREE_CYCLES:
            for profile in time_profiles[key]:
                if not issubclass(type(profile), TemporalProfile):
                    raise TypeError(f"{profile} from {key} is not a TemporalProfile")
                if profile.n_profiles != 1:
                    raise ValueError(
                        f"{profile} from {key} has {profile.n_profiles} profiles."
                        " Only one profile is allowed for the THREE_CYCLES option."
                    )
                scaling_factors = profile.ratios.reshape(-1) * profile.size
                if isinstance(profile, DailyProfile):
                    ds = hourofday
                    # Use the shifts in the intervals
                    data = np.asarray(
                        [np.roll(scaling_factors, -shift) for shift in shifts]
                    )
                    dim = "hourofday"

                else:
                    if isinstance(profile, WeeklyProfile):
                        ds = dayofweek
                        dim = "dayofweek"
                    elif isinstance(profile, MounthsProfile):
                        ds = monthofyear
                        dim = "monthofyear"
                    else:
                        raise TypeError(
                            f"{profile} from {key} is not on of the three profiles:"
                            " DailyProfile, WeeklyProfile, MounthsProfile. You can use"
                            " the HOUR of YEAR option to have scaling with this type"
                            " of profile."
                        )
                    data = np.asarray([scaling_factors for _ in time_zones])

                ds[key] = xr.DataArray(
                    data.T,
                    dims=[dim, "country"],
                    attrs=var_metadata(key, dim),
                )
        elif profiles_type == TemporalProfilesTypes.HOUR_OF_YEAR:
            if year is None:
                raise ValueError("You must provide a year for the HOUR_OF_YEAR option.")

            # Use the shifts in the intervals
            dt_start = datetime(year, 1, 1, hour=0)  # - timedelta(hours=max_shift)
            dt_end = datetime(year, 12, 31, hour=23)  # + timedelta(hours=max_shift + 1)

            concatenated_profiles = np.asarray(
                [
                    # Start around the shift and end
                    create_scaling_factors_time_serie(
                        dt_start,
                        dt_end,
                        time_profiles[key],
                        local_tz=tz,
                        freq="h",
                    ).to_numpy()  # [max_shift + shift : -max_shift + shift - 1]
                    for shift, tz in zip(shifts, time_zones)
                ]
            )

            # Apply the shift for each contry
            hourofyear[key] = xr.DataArray(
                data=concatenated_profiles.T,
                coords={
                    "datetime": (
                        ("hourofyear",),
                        pd.date_range(dt_start, dt_end, freq="h"),
                    ),
                    "timezone_of_country": (("country",), time_zones),
                },
                dims=["hourofyear", "country"],
                attrs=var_metadata(key, "hourofyear"),
            )
        else:
            raise NotImplementedError(f"{profiles_type} is not implemented.")

    if profiles_type == TemporalProfilesTypes.HOUR_OF_YEAR:
        dict_ds = {"hourofyear": hourofyear}
    elif profiles_type == TemporalProfilesTypes.THREE_CYCLES:
        dict_ds = {
            "hourofday": hourofday,
            "dayofweek": dayofweek,
            "monthofyear": monthofyear,
        }
    else:
        raise NotImplementedError(f"{profiles_type} is not implemented.")

    for ds in dict_ds.values():
        ds["country"] = [i for i in range(len(time_zones))]
        ds.assign_coords({"timezone_of_country": (("country",), time_zones)})

    if out_dir is not None:
        # Save the files
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, ds in dict_ds.items():
            ds.to_netcdf(out_dir / f"{name}.nc")

    return dict_ds


def make_icon_vertical_profiles(
    vertical_profiles: dict[str, VerticalProfile],
    out_dir: PathLike | None = None,
    nc_attrs: dict[str, str] = DEFAULT_NC_ATTRIBUTES,
) -> xr.Dataset:
    """Create the vertical profiles in the icon format."""

    # Make sure all teh vertical profiles have the same heights
    # oem does not process different heigths
    resampled_profiles = resample_vertical_profiles(*vertical_profiles.values())

    data_vars = {
        key: (
            "level",
            resampled_profiles[i].ratios,
            {
                "long_name": f"vertical scaling factor for sources of {key} category ",
                "units": "1",
            },
        )
        for i, key in enumerate(vertical_profiles.keys())
    }

    # Add the layers
    data_vars["layer_top"] = ("level", resampled_profiles.height)
    layer_bot = np.roll(resampled_profiles.height, 1)
    layer_bot[0] = 0
    data_vars["layer_bot"] = ("level", layer_bot)
    data_vars["layer_mid"] = ("level", (resampled_profiles.height + layer_bot) / 2.0)

    # Create the dataset
    nc_attrs["title"] = "Vertical profiles"
    ds = xr.Dataset(data_vars=data_vars, attrs=nc_attrs)

    if out_dir is not None:
        # Save the files
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(out_dir / "vertical_profiles.nc")

    return ds
