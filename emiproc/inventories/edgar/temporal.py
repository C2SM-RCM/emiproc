"""Temporal profiles for EDGAR inventory."""

from __future__ import annotations

import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from emiproc.inventories import Inventory
from emiproc.profiles.temporal.profiles import WeeklyProfile, HourOfWeekPerMonthProfile
from emiproc.profiles.utils import read_profile_file
from emiproc.profiles.temporal.composite import (
    make_composite_profiles,
    CompositeTemporalProfiles,
    concatenate_time_profiles,
)
from emiproc.profiles.utils import (
    profiles_to_scalingfactors_dataarray,
    ratios_dataarray_to_profiles,
)


def read_weekly_profile_file(
    auxiliary_filesdir: os.PathLike,
) -> tuple[WeeklyProfile, xr.DataArray]:
    """Read weekly weekly_profiles.csv file."""

    weekly_file = Path(auxiliary_filesdir) / "weekly_profiles.csv"

    if not weekly_file.exists():
        raise FileNotFoundError(f"Weekly profile file {weekly_file} does not exist.")

    df = read_profile_file(weekly_file)

    df_pivot = df.pivot(
        index=["Country_code_A3", "activity_code"],
        columns="Weekday_id",
        values="daily_factor",
    )
    ratios = df_pivot[[i for i in range(1, 8)]].to_numpy()

    profiles = WeeklyProfile(ratios=ratios)

    indexes = xr.DataArray(
        coords=xr.Coordinates.from_pandas_multiindex(
            df_pivot.index.rename(
                {"Country_code_A3": "country", "activity_code": "category"}
            ),
            dim="index",
        ),
        data=np.arange(len(df_pivot.index), dtype=int).reshape(-1),
        dims="index",
    ).unstack("index", fill_value=-1)

    return profiles, indexes


def read_hourly_profiles_file(
    auxiliary_filesdir: os.PathLike,
) -> tuple[HourOfWeekPerMonthProfile, xr.DataArray]:
    """Read hourly_profiles.csv file.

    This file contains hourly profiles dependant on the day of the week and each country
    is having a different week type definition. (eg weekends are not the same day of the week
    in each country).

    For this reason, this function decomposes the hour of day profiles into hour of week
    profiles. Also this file contains for each mounth a different daily profile,
    which means that we need to decompose the hour of week profiles into
    hour of week per mounth profiles.
    """
    auxiliary_filesdir = Path(auxiliary_filesdir)
    hourly_file = auxiliary_filesdir / "hourly_profiles.csv"
    weekendtypes_file = auxiliary_filesdir / "weekenddays.csv"
    weekenddefinitions_file = auxiliary_filesdir / "weekdays.csv"

    df_weekend_of_country = pd.read_csv(
        weekendtypes_file, index_col="Country_code_A3", sep=";"
    )
    df_weekend_definitions = pd.read_csv(weekenddefinitions_file, index_col=0, sep=";")

    if not hourly_file.exists():
        raise FileNotFoundError(f"Hourly profile file {hourly_file} does not exist.")

    df = read_profile_file(hourly_file)

    ds_hourly = xr.DataArray(
        df[[f"h{i}" for i in range(1, 25)]].values,
        dims=["index", "hour"],
        coords={
            "hour": np.arange(1, 25),
            "index": df.index,
            "country": ("index", df["Country_code_A3"].values),
            "category": ("index", df["activity_code"].values),
            "month": ("index", df["month_id"].values),
            "daytype_id": ("index", df["Daytype_id"].values),
        },
    )

    # Concatenate the profiles over the weeks based on the category-country indices
    concat_list = []
    for mount in range(1, 13):
        ds_this_month = ds_hourly.where(ds_hourly["month"] == mount, drop=True)
        ds_this_month["country_daytype"] = (
            "index",
            ds_this_month["country"].values
            + "_"
            + ds_this_month["daytype_id"].astype(str).values,
        )
        for day_of_week in range(1, 8):
            df_this_day = df_weekend_definitions[
                df_weekend_definitions["Weekday_id"] == day_of_week
            ]
            daytype_to_use_per_country = pd.Series(
                index=df_weekend_of_country.index,
                data=df_this_day.loc[df_weekend_of_country["Weekend_type_id"]][
                    "Daytype_id"
                ].values,
            )
            # mask the valid country datetype combinations
            ds_this_day = ds_this_month.where(
                ds_this_month["country_daytype"].isin(
                    daytype_to_use_per_country.index
                    + "_"
                    + daytype_to_use_per_country.values.astype(str)
                ),
                drop=True,
            )
            # Instead of modifying the 'hour' coordinate inplace, create a new DataArray with shifted hours
            ds_this_day_shifted = ds_this_day.assign_coords(
                hour=ds_this_day["hour"]
                + 24 * (day_of_week - 1)
                + (24 * 7 * (mount - 1))
            )
            # Restet the index to have only country and category
            country_cat = (
                ds_this_day_shifted["country"].values
                + "_"
                + ds_this_day_shifted["category"].values
            )
            ds_this_day_shifted = ds_this_day_shifted.assign_coords(index=country_cat)
            concat_list.append(ds_this_day_shifted)

    # Concatenate over the hour, as this is the dimension of the profiles now
    ds_weeklymounth = xr.concat(concat_list, dim="hour")

    indexes = xr.DataArray(
        coords=xr.Coordinates.from_pandas_multiindex(
            pd.MultiIndex.from_arrays(
                [ds_weeklymounth["country"].values, ds_weeklymounth["category"].values],
                names=["country", "category"],
            ),
            dim="index",
        ),
        data=np.arange(len(ds_weeklymounth.index), dtype=int).reshape(-1),
        dims="index",
    ).unstack("index", fill_value=-1)

    profiles = HourOfWeekPerMonthProfile(
        ds_weeklymounth.values / ds_weeklymounth.sum(dim="hour").values[:, None],
    )

    return profiles, indexes


def read_edgar_auxilary_profiles(
    auxiliary_filesdir: os.PathLike,
    inventory: Inventory,
) -> tuple[CompositeTemporalProfiles, xr.DataArray]:
    """Read the auxiliary profiles for the EDGAR inventory.

    Outputs them in a format that they can directly be set as profiles
    to the inventory.

    The auxiliary profiles are available at, as "auxiliary tables":
    https://edgar.jrc.ec.europa.eu/dataset_temp_profile

    some categories might missmatch.
    """

    # Note: this is easily extensible if in the future we have more auxiliary profiles
    p1, i1 = read_weekly_profile_file(auxiliary_filesdir)
    p2, i2 = read_hourly_profiles_file(auxiliary_filesdir)

    da_all = xr.concat(
        [
            profiles_to_scalingfactors_dataarray(p1, i1, use_ratios=True),
            profiles_to_scalingfactors_dataarray(p2, i2, use_ratios=True),
        ],
        dim="ratio",
    )

    full_ratios, full_indexes = ratios_dataarray_to_profiles(da_all)
    profiles_full = CompositeTemporalProfiles.from_ratios(
        full_ratios, types=[type(p1), type(p2)]
    )

    # Add the categories which are in the inventory but not in the auxiliary profiles
    inv_cats = inventory.categories
    indices_cats = full_indexes["category"].values
    categories_present = [c for c in inv_cats if c in indices_cats]
    categories_missing = [c for c in inv_cats if c not in indices_cats]
    # Correct the missing categories to take only the cat of the first part
    category_to_use = {c: c[:3] for c in categories_missing}
    not_in_aux = [c for c in category_to_use.values() if c not in indices_cats]
    if not_in_aux:
        raise ValueError(
            f"Some categories are not available in the auxiliary profiles: {not_in_aux}"
        )
    indexes_corrected = xr.concat(
        [
            full_indexes.sel(category=categories_present),
            full_indexes.sel(category=list(category_to_use.values())).assign_coords(
                category=list(category_to_use.keys())
            ),
        ],
        dim="category",
    )

    # Rename the SEA country to -99 as is convention in emiproc for not specific country
    indexes_corrected = indexes_corrected.assign_coords(
        country=indexes_corrected["country"].where(
            indexes_corrected["country"] != "SEA", "-99"
        )
    )

    # These countries are missing in the profiles
    # SSD: south sudan
    # SRB: serbia
    # MNE: montenegro
    # PSE: palestine
    # ATA: antarctica
    # ATF: french antarctic
    countries_to_assign = {
        "SSD": "SDN",  # sudan
        "SRB": "SCG",  # serbia and montenegro
        "MNE": "SCG",  # serbia and montenegro
        "PSE": "LAO",  # Lebanon
        "ATA": "ARG",  # Argentina (closest)
        "ATF": "ARG",  # Argentina (closest)
    }
    indexes_corrected = xr.concat(
        [
            indexes_corrected,
            indexes_corrected.sel(
                country=list(countries_to_assign.values())
            ).assign_coords(country=list(countries_to_assign.keys())),
        ],
        dim="country",
    )

    return profiles_full, indexes_corrected
