from __future__ import annotations

from enum import Enum
import logging
from datetime import datetime, timedelta
from typing import Type
import numpy as np
import pandas as pd
import xarray as xr


from emiproc.profiles.temporal.composite import CompositeTemporalProfiles, _get_type
from emiproc.profiles.temporal.profiles import (
    AnyTimeProfile,
    DailyProfile,
    DayOfLeapYearProfile,
    DayOfYearProfile,
    Hour3OfDayPerMonth,
    HourOfLeapYearProfile,
    HourOfWeekPerMonthProfile,
    HourOfYearProfile,
    MounthsProfile,
    SpecificDayProfile,
    TemporalProfile,
    WeeklyProfile,
    get_leap_year_or_normal,
)
from emiproc.profiles.temporal.specific_days import SpecificDay, get_days_as_ints
from emiproc.profiles.temporal.utils import ensure_specific_days_consistency
from emiproc.profiles.utils import (
    ratios_dataarray_to_profiles,
    ratios_to_factors,
)


logger = logging.getLogger(__name__)


class TemporalProfilesInterpolated(Enum):
    """Possible temporal profiles for interpolation.

    :param HOUR_OF_YEAR:  Every hour gets a scaling factor
    :param THREE_CYCLES:  Three cycles (hour of day, day of week, month of year)
    """

    HOUR_OF_YEAR = "hour_of_year"
    THREE_CYCLES = "three_cycles"


def get_index_in_profile(
    profile: Type[TemporalProfile] | tuple[Type[TemporalProfile], SpecificDay],
    time_range: pd.DatetimeIndex,
) -> pd.Series:
    """Get the index in the profile for each time in the time range.

    Careful, if using some profiles which don't apply at all times
    (eg. `SpecificDayProfile`), the index will be -1 for these times.

    :param profile: the profile to use
    :param time_range: the time range to use
    :return: the index in the profile for each time in the time range
    """

    if isinstance(profile, tuple):
        profile, specific_day = profile

    if profile == MounthsProfile:
        indexes = time_range.month - 1
    elif profile in [DayOfYearProfile, DayOfLeapYearProfile]:
        indexes = time_range.day_of_year - 1
    elif profile in [DailyProfile, SpecificDayProfile]:
        indexes = time_range.hour
        if profile == SpecificDayProfile:
            # Filter the correct days
            days = get_days_as_ints(specific_day)
            indexes = indexes.where(time_range.day_of_week.isin(days), -1)
    elif profile == WeeklyProfile:
        indexes = time_range.day_of_week
    elif profile in [HourOfYearProfile, HourOfLeapYearProfile]:
        indexes = time_range.hour + (time_range.day_of_year - 1) * 24
    elif profile == Hour3OfDayPerMonth:
        indexes = (time_range.hour // 3) + (time_range.month - 1) * 8
    elif profile == HourOfWeekPerMonthProfile:
        indexes = (
            time_range.hour
            + time_range.day_of_week * 24
            + (time_range.month - 1) * 24 * 7
        )
    else:
        raise NotImplementedError(f"Profile type {profile} not implemented.")

    assert indexes.min() >= -1, f"{profile=}, {time_range=}"
    assert indexes.max() < profile.size, f"{profile=}, {time_range=}"

    return indexes


def get_scaling_factors_at_time(
    profile: CompositeTemporalProfiles,
    time_range: pd.DatetimeIndex,
) -> xr.DataArray:
    """Evaluate the temporal profile at the given time range.

    :arg profile: The temporal profile to evaluate.
    :arg time_range: The time range to evaluate the profile at.

    :return: A DataArray with the scaling factors for each time in the time range.
    """

    # Scaling factor data array
    da_sf = xr.DataArray(
        profile.scaling_factors,
        coords={"profile": np.arange(len(profile)), "ratio": np.arange(profile.size)},
    )

    # Get teh index of of the scaling factor for each type of temporal profile
    size_offset = 0
    indices = []

    for profile_type in profile.types:

        this_index = get_index_in_profile(profile_type, time_range)

        out_index = this_index + size_offset
        out_index = out_index.where(this_index != -1, -1)

        indices.append(out_index)

        size_offset += _get_type(profile_type).size

    indices_to_use = xr.DataArray(
        np.array(indices).T,
        coords=dict(
            time=time_range,
            sub_profile=np.arange(len(profile.types)),
        ),
        dims=["time", "sub_profile"],
    )

    # Here for the indexing to work, we need to temporary set the missing values to 0 and then put them back to 1
    da_sf_of_profile = (
        da_sf.sel(
            ratio=indices_to_use.where(indices_to_use != -1, 0)
            # Set the no profile to 1. scaling factor value
        ).where(indices_to_use != -1, 1.0)
        # Multiply the scaling factors for each sub-profile
        .prod(dim="sub_profile")
    )

    return da_sf_of_profile


def get_profile_da(
    profile: TemporalProfile,
    year: int,
    use_offset: bool = True,
) -> xr.DataArray:
    """Return the profile as a data array.

    The index of the data array is exact timestamp at the middle of the period.
    """
    daterange_kwargs = {
        "start": f"{year}-01-01",
        "end": f"{year+1}-01-01",
        "inclusive": "both",
    }

    # The following will create correct timestamps at which the profile is true
    # An offset is also given, which is half the period

    if isinstance(
        profile,
        (
            DailyProfile,
            HourOfYearProfile,
            HourOfLeapYearProfile,
            SpecificDayProfile,
            HourOfWeekPerMonthProfile,
        ),
    ):
        ts = pd.date_range(**daterange_kwargs, freq="h")
        offset = pd.Timedelta("30m")
    elif isinstance(profile, (WeeklyProfile, DayOfLeapYearProfile, DayOfYearProfile)):
        ts = pd.date_range(**daterange_kwargs, freq="d")
        offset = pd.Timedelta("12h")
    elif isinstance(profile, MounthsProfile):
        ts = pd.date_range(**daterange_kwargs, freq="MS")
        offset = pd.Timedelta("15d")
    elif isinstance(profile, Hour3OfDayPerMonth):
        ts = pd.date_range(**daterange_kwargs, freq="3h")
        offset = pd.Timedelta("1h30m")
    else:
        raise NotImplementedError(
            f"{type(profile)=} not implemented in `get_profile_da`."
        )

    # Add a first value at the begginning, such that we cover the whole year
    ts = pd.DatetimeIndex([ts[0] - 2 * offset] + list(ts))

    # Correct for non cyclic profiles (day of specific year)
    if isinstance(
        profile,
        (
            HourOfYearProfile,
            HourOfLeapYearProfile,
            DayOfLeapYearProfile,
            DayOfYearProfile,
        ),
    ):
        ts = ts[1:-1]

    indices = get_index_in_profile(
        (
            (type(profile), profile.specific_day)
            if isinstance(profile, SpecificDayProfile)
            else type(profile)
        ),
        ts,
    )

    # Drop the -1 values
    mask_valid = indices != -1

    indices = indices[mask_valid]
    ts = ts[mask_valid]

    da = xr.DataArray(
        profile.ratios[:, indices],
        dims=["profile", "datetime"],
        coords={
            "profile": range(profile.n_profiles),
            "datetime": ts + offset if use_offset else ts,
        },
    )

    return da


def interpolate_profiles(
    profiles: CompositeTemporalProfiles,
    year: int,
    interpolation_method: str = "linear",
    return_profiles: bool = False,
    output_type: TemporalProfilesInterpolated = TemporalProfilesInterpolated.HOUR_OF_YEAR,
) -> (
    CompositeTemporalProfiles
    | xr.DataArray
    | tuple[CompositeTemporalProfiles | xr.DataArray, xr.DataArray]
):
    """Interpolate the profiles to create another specific profile.

    :arg profiles: The profiles to use.
    :arg year: The year to use.
    :arg interpolation_method: The interpolation method to use.
        See `xarray <https://docs.xarray.dev/en/stable/user-guide/interpolation.html>`_
        for more details.
    :arg return_profiles: If True, return the profiles instead of the ratios.
    :arg output_type: The type of the output profile.

    :return: The interpolated profiles or the ratios based on the
        `return_profiles` argument.
    """
    serie = pd.date_range(
        f"{year}-01-01", f"{year+1}-01-01", freq="h", inclusive="left"
    )

    if (
        return_profiles is False
        and output_type != TemporalProfilesInterpolated.HOUR_OF_YEAR
    ):
        raise ValueError(
            "If `return_profiles` is False, `output_type` must be "
            f"{TemporalProfilesInterpolated.HOUR_OF_YEAR}."
        )

    das_scaling_factors = []

    ratios = profiles.ratios

    offset = 0
    for t in profiles.types:
        if _get_type(t) == SpecificDayProfile:
            raise ValueError(f"Cannot interpolate {t=}, it is a specific day profile.")
        # create an array with the ratios
        t_len = t.size
        this_ratios = ratios[:, offset : offset + t_len]
        this_ratios = np.nan_to_num(this_ratios, nan=1.0 / t_len)
        offset += t_len
        # Create a data array for these ratios and convert to scaling factors
        da_sf = get_profile_da(profile=t(this_ratios), year=year) * t_len
        # Interpolate the data array
        da_interp = da_sf.interp(
            datetime=serie,
            method=interpolation_method,
            assume_sorted=True,
        )

        das_scaling_factors.append(da_interp.expand_dims("profile_type"))

    # Multiply the data arrays
    da = xr.concat(das_scaling_factors, dim="profile_type").prod(dim="profile_type")
    da_ratios = da / da.sum(dim="datetime")

    if output_type == TemporalProfilesInterpolated.THREE_CYCLES:
        # Create the threee cyles
        ratios = {}
        ratios[DailyProfile] = (
            da_ratios.groupby("datetime.hour").mean().rename({"hour": "ratio"})
        )
        ratios[WeeklyProfile] = (
            da_ratios.groupby("datetime.dayofweek")
            .mean()
            .rename({"dayofweek": "ratio"})
        )
        # Sum for the mounths to account for longer or shorter months
        ratios[MounthsProfile] = (
            da_ratios.groupby("datetime.month").sum().rename({"month": "ratio"})
        )
        types = list(ratios.keys())
        da_ratios = xr.concat([da for da in ratios.values()], dim="ratio")
    elif output_type == TemporalProfilesInterpolated.HOUR_OF_YEAR:
        types = [get_leap_year_or_normal(HourOfYearProfile, year)]
    else:
        raise NotImplementedError(
            f"Output type {output_type} not implemented. "
            "Use any of the "
            f"{TemporalProfilesInterpolated}."
        )

    # Create the profile
    if return_profiles:
        return CompositeTemporalProfiles.from_ratios(
            da_ratios.values,
            types=types,
            rescale=output_type != TemporalProfilesInterpolated.HOUR_OF_YEAR,
        )

    return da_ratios


# Old legacy function
interpolate_profiles_hour_of_year = interpolate_profiles


def resolve_daytype(
    profiles: CompositeTemporalProfiles, profiles_indexes: xr.Dataset, year: int
) -> tuple[CompositeTemporalProfiles, xr.Dataset]:

    time_range = pd.date_range(
        f"{year}-01-01", f"{year+1}-01-01", freq="h", inclusive="left"
    )

    # Few checks on the day types given
    assert (
        "day_type" in profiles_indexes.dims
    ), "The profiles indexes must have a 'day_type' dimension"
    day_types = profiles_indexes.day_type.values
    specific_days = [SpecificDay(day_type) for day_type in day_types]
    days_mapping = {day: get_days_as_ints(day) for day in specific_days}
    all_values = sum([days for days in days_mapping.values()], [])
    if sorted(all_values) != sorted(list(range(7))):
        raise ValueError(
            f"Invalid {day_types=}, must cover all days of the week but they cover {all_values}."
        )

    # Check that the profile given is correct
    expected_profile = get_leap_year_or_normal(HourOfYearProfile, year)
    if not isinstance(profiles, CompositeTemporalProfiles | expected_profile):
        raise TypeError(
            f"{profiles=} must be a {CompositeTemporalProfiles} or {expected_profile}."
        )
    if profiles.types != [expected_profile]:
        raise ValueError(
            f"{profiles=} must contain only {expected_profile} for the given {year=}."
        )

    # Get only the profiles which are not the same on the day types
    dims = list(profiles_indexes.dims)
    dims.remove("day_type")
    stacked_indexes = profiles_indexes.stack(ind=dims)
    mask_differ_over_daytype = ~(
        stacked_indexes == stacked_indexes.isel(day_type=0)
    ).all("day_type")
    require_merge_indexes = stacked_indexes.loc[{"ind": mask_differ_over_daytype}]

    if np.any(require_merge_indexes == -1):
        raise ValueError(
            f"Cannot resolve {profiles_indexes=} as some profiles are missing. \n"
            "Please fill them with constant profiles."
        )

    # Get the ratios of each datetime and time
    da_ratios = xr.DataArray(
        profiles.ratios[require_merge_indexes],
        dims=["day_type", "ind", "time"],
        coords={
            "day_type": require_merge_indexes["day_type"],
            "ind": require_merge_indexes["ind"],
            "time": time_range,
        },
    )

    # Create the output index array
    da_indexes_out = xr.zeros_like(
        mask_differ_over_daytype.loc[mask_differ_over_daytype], dtype=int
    )

    da_profiles_out = da_ratios.isel(day_type=0).drop_vars("day_type")

    for day_type, days in days_mapping.items():
        # Assign the values of each datetime to the correct day
        mask = da_ratios.time.dt.dayofweek.isin(days)
        da_profiles_out.loc[{"time": mask}] = da_ratios.sel(
            day_type=day_type.value
        ).loc[{"time": mask}]

    new_profiles, new_indexes = ratios_dataarray_to_profiles(
        da_profiles_out.rename({"time": "ratio"}).unstack(fill_value=-1)
    )

    new_indexes = new_indexes.stack(ind=dims)
    # Remove the missing
    new_indexes = new_indexes.loc[new_indexes != -1]
    # Set them to the correct value  (as we later append at the end the new profiles)
    new_indexes += profiles.n_profiles

    out_indexes = stacked_indexes.isel(day_type=0).drop_vars("day_type").copy(deep=True)
    out_indexes.loc[{"ind": new_indexes.ind}] = new_indexes

    # Need to rescale as now the ratio might not sum up to 1 exactly
    new_profiles /= new_profiles.sum(axis=1).reshape(-1, 1)

    out_profiles = CompositeTemporalProfiles.from_ratios(
        np.concatenate([profiles.ratios, new_profiles]), types=[expected_profile]
    )

    return out_profiles, out_indexes.unstack("ind", fill_value=-1)


# below are legacy functions, but still used in some places


def create_scaling_factors_time_serie(
    start_time: datetime,
    end_time: datetime,
    profiles: list[AnyTimeProfile],
    apply_month_interpolation: bool = True,
    freq: str = "h",
    inclusive: str = "both",
    local_tz: str | None = None,
) -> pd.Series:
    """Create a time serie of ratios for the requested time range.

    :arg start_time: The start time of the time serie.
    :arg end_time: The end time of the time serie.
    :arg profiles: The profiles to use to create .
    :arg apply_month_interpolation: If True, apply the month interpolation.
    :arg inclusive: {“both”, “neither”, “left”, “right”}, default “both”
        same as `pd.date_range <https://pandas.pydata.org/docs/reference/api/pandas.date_range.html#pandas-date-range>`_
        Include boundaries; Whether to set each bound as closed or open.
    """

    kwargs = {}
    if local_tz is not None:
        kwargs["tz"] = "UTC"
    # Create the time serie
    time_serie = pd.date_range(
        start_time, end_time, freq=freq, inclusive=inclusive, **kwargs
    )
    if local_tz is not None:
        time_serie = time_serie.tz_convert(local_tz)

    # Create the scaling factors
    scaling_factors = np.ones(len(time_serie))

    # Correct profiles list with specific day profiles
    profiles = ensure_specific_days_consistency(profiles)

    # Apply the profiles
    for profile in profiles:
        scaling_factors *= _profile_to_scaling_factors(
            time_serie, profile, apply_month_interpolation=apply_month_interpolation
        )

    return pd.Series(scaling_factors, index=time_serie)


def _profile_to_scaling_factors(
    time_serie: pd.DatetimeIndex,
    profile: AnyTimeProfile,
    apply_month_interpolation: bool = True,
) -> np.ndarray:
    """Convert a temporal profile to a time serie.

    :arg apply_month_interpolation: If True, apply the month interpolation.
        Only applies when the profile is a :py:class: MounthsProfile
        Each mounthly values is assinged to the 15th of the month and then
        interpolation is used to get the values for the other days.

    :return: Scaling factors
        An array by which you can multiply the emission factor using the time serie.

    """

    # Get scaling factors, convert the ratios to scaling factors
    if len(profile) != 1:
        raise ValueError(
            f"Cannot apply {profile=} to a time serie, it must have only one profile."
        )

    factors = ratios_to_factors(profile.ratios.reshape(-1))

    # This will be the output
    scaling_factors = np.ones(len(time_serie))

    # Get the profile
    if isinstance(profile, SpecificDayProfile):
        # Find the days corresponding to this factor
        days_allowed = get_days_as_ints(profile.specific_day)
        if len(days_allowed) != 1:
            raise ValueError(
                f"Cannot apply {profile=} to a time serie, it must have only one day."
                "convert the time profiles with `ensure_specific_days_consistency`."
            )
        mask_matching_day = np.isin(time_serie.day_of_week, days_allowed)
        for hour, factor in enumerate(factors):
            # Other days will not have a scaling factor
            scaling_factors[(time_serie.hour == hour) & mask_matching_day] *= factor
    elif isinstance(profile, DailyProfile):
        # Get the mask for each hour of day and apply the scaling factor
        for hour, factor in enumerate(factors):
            scaling_factors[time_serie.hour == hour] *= factor
    elif isinstance(profile, WeeklyProfile):
        # Get the mask for each day of week and apply the scaling factor
        for day, factor in enumerate(factors):
            scaling_factors[time_serie.dayofweek == day] *= factor
    elif isinstance(profile, MounthsProfile):
        if apply_month_interpolation:
            # Apply the factor to the 15 th of each month by getting the exact datetime
            # of the 15th of each month
            mid_months = pd.date_range(
                start=time_serie[0],
                end=time_serie[-1],
                freq="MS",
            ) + timedelta(days=14)
            mid_months_factors = np.ones(len(mid_months))
            # Set the value to each month
            for month, factor in enumerate(factors):
                # Months start with 1
                month += 1
                mid_months_factors[mid_months.month == month] *= factor
            # Interpolate the values to the other days
            scaling_factors = np.interp(
                time_serie,
                mid_months,
                mid_months_factors,
            )
        else:
            # Simply apply to each month the scaling factor
            for month, factor in enumerate(factors):
                # Months start with 1
                month += 1
                scaling_factors[time_serie.month == month] *= factor
    else:
        raise NotImplementedError(
            f"Cannot apply {profile=}, {type(profile)=} is not implemented."
        )

    # Return the scaling factors
    return scaling_factors
