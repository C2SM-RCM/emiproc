import numpy as np
import pandas as pd
import xarray as xr

from emiproc.inventories import Inventory
from emiproc.profiles import temporal_profiles
from emiproc.utils.translators import inv_to_xarray


def get_temporally_scaled_array(
    inv: Inventory,
    time_range: pd.DatetimeIndex,
) -> xr.DataArray:
    """Transform the inventory to a temporally resolved emissions array.

    Missing profiles are assumed to be constant profiles.

    You can easily plot the profiles of you different category and substances
    of the received inventory with the following code:
    ```
    da.sum("cell").stack(catsub=["category", "substance"]).plot.line(x="time")
    ```

    :param inv: the inventory to transform
    :param time_range: the time range to use for the temporal resolution.

    :return: the temporally resolved emissions array.
        The units are the same as in the inventory. (kg/y/cell)
        But now even scaled on the time axis given units are still kg/y/cell.
        If you want to get the emissions at your time resolution you need divide
        by the number of your time resolution that fits in a year.
    """

    profiles, profiles_indexes = inv.t_profiles_groups, inv.t_profiles_indexes

    da_totals = inv_to_xarray(inv)

    # Acess the scaling factors
    scaling_factors_array = profiles.scaling_factors[profiles_indexes]

    if "cell" in profiles_indexes.dims:
        # The profiles are usually only given on cells with emissions
        missing_cells = da_totals.cell.loc[~da_totals.cell.isin(profiles_indexes.cell)]
        # Check that the profiles are given for all cells
        assert (
            da_totals.sel(cell=missing_cells).sum().values == 0
        ), "Some cell or emissions with none zero values have missing profiles"

    da_scaling_factors = xr.DataArray(
        scaling_factors_array,
        coords=dict(
            **profiles_indexes.coords,
            profile_index=np.arange(profiles.size),
        ),
    )

    # Get teh index of of the scaling factor for each type of temporal profile
    size_offset = 0
    indices = []

    for profile_type in profiles.types:

        if profile_type == temporal_profiles.MounthsProfile:
            indices.append(time_range.month - 1 + size_offset)
        elif profile_type == temporal_profiles.DayOfYearProfile:
            indices.append(time_range.day_of_year - 1 + size_offset)
        elif profile_type == temporal_profiles.DailyProfile:
            indices.append(time_range.hour + size_offset)
        elif profile_type == temporal_profiles.WeeklyProfile:
            indices.append(time_range.day_of_week + size_offset)
        elif profile_type in [
            temporal_profiles.HourOfYearProfile,
            temporal_profiles.HourOfLeapYearProfile,
        ]:
            indices.append(
                time_range.hour + (time_range.day_of_year - 1) * 24 + size_offset
            )
        else:
            raise ValueError(f"Profile type {profile_type} not recognized")

        size_offset += profile_type.size

    indices_to_use = xr.DataArray(
        np.array(indices).T,
        coords=dict(
            time=time_range,
            profile=np.arange(len(profiles.types)),
        ),
        dims=["time", "profile"],
    )

    # Get the proper scaling factors for each index
    scaling_factors_at_times = da_scaling_factors.loc[
        dict(profile_index=indices_to_use)
    ]
    # Merge all the time factors together
    scaling_factor_at_times = scaling_factors_at_times.prod("profile")
    # Set the scaling factors on the missing cells
    scaling_factor_at_times_all_cells = scaling_factor_at_times.reindex(
        da_totals.coords
    ).fillna(1.0)

    # Finally scale the emissions at each time
    temporally_scaled_emissions = da_totals * scaling_factor_at_times_all_cells

    return temporally_scaled_emissions
