from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr

from emiproc.inventories import Inventory
from emiproc.profiles.temporal_profiles import get_index_in_profile
from emiproc.utils.translators import inv_to_xarray


def get_temporally_scaled_array(
    inv: Inventory,
    time_range: pd.DatetimeIndex | int,
    sum_over_cells: bool = True,
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
        If an integer is given the time range will a daily range of the given year.
    :param sum_over_cells: if True the emissions are summed over the cells.
        This can be useful to improve the performance of the plotting.

    :return: the temporally resolved emissions array.
        The units are the same as in the inventory. (kg/y/cell)
        But now even scaled on the time axis given units are still kg/y/cell.
        If you want to get the emissions at your time resolution you need divide
        by the number of your time resolution that fits in a year.
    """

    profiles, profiles_indexes = inv.t_profiles_groups, inv.t_profiles_indexes

    if profiles is None or profiles_indexes is None:
        raise ValueError(
            "The inventory does not have temporal profiles."
            "You need to set the profiles to get a temporally resolved emissions array."
        )

    if isinstance(time_range, int):
        time_range = pd.date_range(
            start=f"{time_range}-01-01",
            end=f"{time_range}-12-31",
            freq="D",
            inclusive="both",
        )

    da_totals = inv_to_xarray(inv)
    if sum_over_cells:
        da_totals = da_totals.sum("cell")

    # Acess the scaling factors
    scaling_factors_array = profiles.scaling_factors[profiles_indexes]

    if "cell" in profiles_indexes.dims:
        if sum_over_cells:
            raise ValueError(
                "The scaling factors are defined for individual cells."
                "You need to set sum_over_cells to False"
            )
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

        indices.append(get_index_in_profile(profile_type, time_range) + size_offset)

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
