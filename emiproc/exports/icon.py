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
from emiproc.inventories import Category, Inventory, Substance
from emiproc.profiles.temporal.composite import (
    CompositeTemporalProfiles,
    make_composite_profiles,
)
from emiproc.profiles.temporal.profiles import (
    DailyProfile,
    MounthsProfile,
    TemporalProfile,
    WeeklyProfile,
    HourOfYearProfile,
    HourOfLeapYearProfile,
    get_leap_year_or_normal,
)
from emiproc.profiles.temporal.operators import (
    create_scaling_factors_time_serie,
    TemporalProfilesInterpolated,
    interpolate_profiles,
)
from emiproc.profiles.vertical_profiles import (
    VerticalProfile,
    VerticalProfiles,
    resample_vertical_profiles,
)
from emiproc.tests_utils.temporal_profiles import (
    oem_const_profile,
    get_oem_const_hour_of_year_profile,
)
from emiproc.utilities import (
    SEC_PER_HOUR,
    SEC_PER_YR,
    get_timezone_mask,
)
from emiproc.profiles.utils import (
    get_desired_profile_index,
    group_profile_cells_by_regions,
)


class TemporalProfilesTypes(Enum):
    """Possible temporal profiles for OEM.

    :param HOUR_OF_YEAR:  Every hour gets a scaling factor
    :param THREE_CYCLES:  Three cycles (hour of day, day of week, month of year)
    """

    HOUR_OF_YEAR = TemporalProfilesInterpolated.HOUR_OF_YEAR
    THREE_CYCLES = TemporalProfilesInterpolated.THREE_CYCLES


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
    nc_attributes: dict[str, str] = DEFAULT_NC_ATTRIBUTES,
    substances: list[str] | None = None,
    correct_tz_shift: bool = True,
    # Deprectated, use inv.year instead
    year: int | None = None,
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

    if year is not None:
        logger.warning(
            "The year parameter is deprecated and will be removed in the future. "
            "You need now to specify the year in the inventory object."
        )

    test_year_and_profiles_types_combination(
        year=inv.year,
        profiles_type=temporal_profiles_type,
    )

    icon_grid_file = Path(icon_grid_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    # Load the output xarray

    ds_out: xr.Dataset = xr.load_dataset(icon_grid_file)
    catsubs: dict[tuple[Category, Substance], str] = {}
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
        catsubs[(categorie, sub)] = name

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

    date_of_shift = datetime(
        year=inv.year if inv.year is not None else datetime.now().year, month=1, day=1
    )

    t_profiles_indexes = inv.t_profiles_indexes
    time_profiles = inv.t_profiles_groups

    if t_profiles_indexes is None:
        assert time_profiles is None
        t_profiles_indexes = xr.DataArray(
            np.zeros(
                shape=(len(inv.categories), len(inv.substances)), dtype=int
            ),  # All profiles are constant
            coords={
                "category": inv.categories,
                "substance": inv.substances,
            },
        )
        time_profiles = CompositeTemporalProfiles(
            [
                (
                    oem_const_profile
                    if temporal_profiles_type == TemporalProfilesTypes.THREE_CYCLES
                    else get_oem_const_hour_of_year_profile(inv.year)
                )
            ]
        )

    if "cell" in t_profiles_indexes.dims:
        # Get the regions with same temporal profiles
        regions_index, region_of_cell = group_profile_cells_by_regions(
            t_profiles_indexes
        )
    else:
        # Create a single region for all cells
        regions_index = xr.zeros_like(t_profiles_indexes, dtype=int).expand_dims(
            {"region": np.array([0], dtype=int)}
        )
        region_of_cell = xr.DataArray(
            np.zeros_like(tz_mask.reshape(-1), dtype=int),
            dims=["cell"],
            coords={"cell": ds_out.coords["cell"]},
        )

    tz_mask_str = xr.DataArray(
        tz_mask.reshape(-1),
        dims=["cell"],
        coords={"cell": region_of_cell.coords["cell"]},
    )

    str_identifier = (tz_mask_str.str + "_").str + region_of_cell.astype(str)
    u2, ind2, inv2 = np.unique(str_identifier, return_index=True, return_inverse=True)

    tz_region = tz_mask_str.sel(cell=ind2).values
    tz_shift = [
        # Calculate the country shifts for the grid cells
        ZoneInfo(code).utcoffset(date_of_shift).seconds
        for code in tz_region
    ]
    regions = xr.DataArray(
        u2,
        dims=["region"],
        coords={
            "temporal_profile_id": ("region", region_of_cell.sel(cell=ind2).values),
            "tz_region": ("region", tz_region),
            "tz_shift": (
                "region",
                # Convert the seconds to hours
                (np.array(tz_shift) / SEC_PER_HOUR).astype(int),
            ),
        },
    )
    # Save the profiles
    make_icon_time_profiles(
        catsubs=catsubs,
        time_profiles=time_profiles,
        inv=inv,
        profiles_indexes=regions_index,
        regions=regions,
        profiles_type=temporal_profiles_type,
        out_dir=output_dir,
        nc_attrs=nc_attributes,
        correct_tz_shift=correct_tz_shift,
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
                inv2,
                {
                    "standard_name": "country_ids",
                    "long_name": "Timezone of the cell",
                    "history": f"Added by emiproc",
                    "country_resolution": f"country_resolution",
                },
            ),
            "timezone_of_country": (
                "region",
                regions["tz_region"].values,
                {
                    "standard_name": "timezone_of_country",
                    "long_name": "Timezone corresponding the country id",
                    "history": f"Added by emiproc",
                },
            ),
            "temporal_profile_id": (
                "region",
                regions["temporal_profile_id"].values,
                {
                    "standard_name": "temporal_profile_id",
                    "long_name": "Temporal profile id",
                    "history": f"Added by emiproc",
                },
            ),
            "region_identifier": (
                "region",
                regions.values.astype(str),
                {
                    "standard_name": "region_identifier",
                    "long_name": "Identifier of the region",
                    "history": f"Added by emiproc",
                },
            ),
            "tz_shift": (
                "region",
                regions["tz_shift"].values,
                {
                    "standard_name": "tz_shift",
                    "long_name": "Timezone shift (in hours)",
                    "history": f"Added by emiproc",
                },
            ),
        }
    )
    # Save the emissions
    ds_out.to_netcdf(output_dir / "oem_gridded_emissions.nc")

    logger.info(f"Exported inventory to {output_dir}.")


def test_year_and_profiles_types_combination(
    year: int | None,
    profiles_type: TemporalProfilesTypes,
):
    if year is None:
        if profiles_type == TemporalProfilesTypes.HOUR_OF_YEAR:
            raise ValueError(
                "You must provide a year for the temporal profiles option."
            )


def make_icon_time_profiles(
    catsubs: dict[tuple[Category, Substance], str],
    time_profiles: CompositeTemporalProfiles,
    inv: Inventory,
    profiles_indexes: xr.DataArray | None,
    regions: xr.DataArray,
    profiles_type: TemporalProfilesTypes = TemporalProfilesTypes.THREE_CYCLES,
    out_dir: PathLike | None = None,
    nc_attrs: dict[str, str] = DEFAULT_NC_ATTRIBUTES,
    correct_tz_shift: bool = True,
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
    year = inv.year
    if year is None and profiles_type == TemporalProfilesTypes.THREE_CYCLES:
        # set a random year for profiles that don't depend on the year
        year = datetime.now().year

    profiles_type = TemporalProfilesTypes(profiles_type)

    HoyProfile = get_leap_year_or_normal(HourOfYearProfile, year=year)

    if time_profiles is None:
        time_profiles = oem_const_profile
        assert (
            profiles_indexes is None
        ), "When no profiles are given, indexes should be None as well"
        profiles_indexes = xr.DataArray(
            np.zeros(
                shape=(len(inv.categories), len(inv.substances)), dtype=int
            ),  # All profiles are constant
            coords={
                "category": inv.categories,
                "substance": inv.substances,
            },
        )

    time_profiles, profiles_indexes = make_composite_profiles(
        time_profiles, indexes=profiles_indexes
    )
    dict_ds = {}

    # Put the profiles on common regions times

    if profiles_type == TemporalProfilesTypes.THREE_CYCLES:
        if not sorted(time_profiles.types, key=lambda t: t.__name__) == sorted(
            [
                DailyProfile,
                WeeklyProfile,
                MounthsProfile,
            ],
            key=lambda t: t.__name__,
        ):
            raise ValueError(
                f"Expected {DailyProfile}, {WeeklyProfile}, {MounthsProfile} in the"
                f" time profiles, but got {time_profiles.types}."
                f" You can interpolate the profiles to get the expected ones with "
                f"emiproc.inventories.utils.interpolate_temporal_profiles."
            )
        nc_attrs["title"] = "Hour of day profiles"
        dict_ds["hourofday"] = xr.Dataset(attrs=nc_attrs.copy())
        nc_attrs["title"] = "Day of week profiles"
        dict_ds["dayofweek"] = xr.Dataset(attrs=nc_attrs.copy())
        nc_attrs["title"] = "Month of year profiles"
        dict_ds["monthofyear"] = xr.Dataset(attrs=nc_attrs.copy())
    elif profiles_type == TemporalProfilesTypes.HOUR_OF_YEAR:
        if not time_profiles.types == [HoyProfile]:
            raise ValueError(
                f"Expected {HourOfYearProfile} or {HourOfLeapYearProfile} "
                "based on the year it is in the"
                f" time profiles, but got {time_profiles.types}."
                f" You can interpolate the profiles to get the expected ones with "
                f"emiproc.inventories.utils.interpolate_temporal_profiles."
            )
        nc_attrs["title"] = "Hour of year profiles"
        dict_ds["hourofyear"] = xr.Dataset(attrs=nc_attrs.copy())
    else:
        raise NotImplementedError(f"{profiles_type} is not implemented.")

    var_metadata = lambda var_name, profile_name: {
        "units": "1",
        "long_name": f"{profile_name} scaling factors for {var_name}",
    }
    profile_name = {
        DailyProfile: "hourofday",
        WeeklyProfile: "dayofweek",
        MounthsProfile: "monthofyear",
        HoyProfile: "hourofyear",
    }

    for (cat, sub), name in catsubs.items():
        scaling_factors = time_profiles.scaling_factors
        index_sel_dict = {}
        if "category" in profiles_indexes.dims:
            index_sel_dict["category"] = cat
        if "substance" in profiles_indexes.dims:
            index_sel_dict["substance"] = sub

        index_sel_dict["region"] = regions.coords["temporal_profile_id"].values
        sf_indexes = profiles_indexes.sel(**index_sel_dict).values
        scaling_factors = scaling_factors[sf_indexes, :]
        # Skip the scaling factors in the three cycles
        sf_counter = 0
        for profile_type in time_profiles.types:
            this_scaling_factors = scaling_factors[
                :, sf_counter : sf_counter + profile_type.size
            ]
            sf_counter += profile_type.size

            if profile_type in [DailyProfile, HoyProfile] and correct_tz_shift:
                # Shift the scaling factors for the hour of day
                this_scaling_factors = np.roll(
                    this_scaling_factors, -regions.coords["tz_shift"].values, axis=1
                )

            dim = profile_name[profile_type]

            dict_ds[dim][name] = xr.DataArray(
                this_scaling_factors.T,
                dims=[dim, "country"],
                attrs=var_metadata(name, dim),
            )

    coords = {
        "timezone_of_country": (("country",), regions["tz_region"].values),
        "temporal_profile_id": (
            ("country",),
            regions["temporal_profile_id"].values,
        ),
        "region_key": (("country",), regions.values.astype(str)),
    }
    for ds in dict_ds.values():
        ds["country"] = np.arange(regions.sizes["region"], dtype=int)

    dict_ds = {name: ds.assign_coords(coords) for name, ds in dict_ds.items()}
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
