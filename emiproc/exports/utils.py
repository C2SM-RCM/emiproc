from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr

from emiproc.inventories import Inventory
from emiproc.profiles.temporal.composite import CompositeTemporalProfiles, _get_type
from emiproc.profiles.temporal.operators import get_scaling_factors_at_time
from emiproc.profiles.temporal.specific_days import get_days_as_ints
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

    .. warning::
        If you use this function and specify a time frequency larger than the
        temporal resolution of the profiles, you might perform undersampling.
        This means for example that if you sample with a daily frequency and
        you have hour of day profiles, you will use the same value in the the
        profile, which will lead to a wrong result.

    :param inv: the inventory to transform
    :param time_range: the time range to use for the temporal resolution.
        If an integer is given the time range will a daily range of the given year.
    :param sum_over_cells: if True the emissions are summed over the cells.
        This can be useful to improve the performance of the plotting.

    :return: the temporally resolved emissions array.
        The units are the same as in the inventory. (kg/y/cell or kg/y if sum_over_cells is True)
        But now even scaled on the time axis given units are still that unit.
        If you want to get the emissions at your time resolution you need divide
        by the number of your time resolution that fits in a year.
    """

    profiles, profiles_indexes = inv.t_profiles_groups, inv.t_profiles_indexes

    if profiles is None or profiles_indexes is None:
        raise ValueError(
            "The inventory does not have temporal profiles."
            "You need to set the profiles to get a temporally resolved emissions array."
        )

    if "country" in profiles_indexes.dims:
        raise ValueError(
            "Inventory profiles have a country dimension "
            "To calculate the temporally resolved emissions array, "
            "you need to convert country profiles to cell profiles first, with "
            "`emiproc.inventories.utils.country_to_cells()` ."
        )

    if isinstance(time_range, int):
        time_range = pd.date_range(
            start=f"{time_range}-01-01",
            end=f"{time_range}-12-31",
            freq="D",
            inclusive="both",
        )
    elif not isinstance(time_range, pd.DatetimeIndex):
        raise TypeError(
            f"Expected a pd.DatetimeIndex or int for `time_range`, got {type(time_range)}."
        )

    da_totals = inv_to_xarray(inv)

    if not isinstance(profiles, CompositeTemporalProfiles):
        profiles = CompositeTemporalProfiles(profiles)

    if "cell" in profiles_indexes.dims:
        # The profiles are usually only given on cells with emissions
        missing_cells = da_totals.cell.loc[~da_totals.cell.isin(profiles_indexes.cell)]
        # Check that the profiles are given for all cells
        zero_cells_missing = da_totals.sel(cell=missing_cells).where(
            da_totals.sel(cell=missing_cells) == 0, drop=True
        )
        if zero_cells_missing.size > 0:
            raise ValueError(
                "Some cells have emissions but no profiles are given for them."
                f" Missing cells: {zero_cells_missing}"
            )
    else:
        # If cell not given, we can speed up the calculation
        if sum_over_cells:
            da_totals = da_totals.sum("cell")

    da_sf = get_scaling_factors_at_time(profiles, time_range)

    # Get the scaling factors for each profile
    da_scaling_factors = da_sf.sel(
        # Apply similar strategy for missing profiles
        profile=profiles_indexes.where(profiles_indexes != -1, 0)
    ).drop_vars("profile")
    # Apply similar strategy for missing profiles which is more performant (in place)
    da_scaling_factors.loc[
        profiles_indexes.where(profiles_indexes == -1, drop=True).coords
    ] = 1.0

    da_scaling_factors = da_scaling_factors.broadcast_like(da_totals)

    if sum_over_cells and "cell" in profiles_indexes.dims:
        # instad of multilplying in a first step and summing in a second
        # we can use the dot product to get the same result
        temporally_scaled_emissions = da_totals.dot(da_scaling_factors, dim="cell")

    else:
        # Finally scale the emissions at each time
        temporally_scaled_emissions = da_totals * da_scaling_factors

    return temporally_scaled_emissions
