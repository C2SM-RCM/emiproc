from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import yaml

import emiproc
from emiproc.profiles.temporal.composite import make_composite_profiles
from emiproc.profiles.temporal.profiles import (
    AnyProfiles,
    AnyTimeProfile,
    DailyProfile,
    MounthsProfile,
    SpecificDayProfile,
    WeeklyProfile,
)
from emiproc.profiles.temporal.specific_days import SpecificDay
from emiproc.profiles.utils import (
    get_profiles_indexes,
    merge_indexes,
    read_profile_csv,
    read_profile_file,
)

_weekdays_long = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
_months_short = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
_months_long = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

timprofile_colnames = {
    WeeklyProfile: [
        _weekdays_short := [i[:3] for i in _weekdays_long],
        [i.lower() for i in _weekdays_short],
        _weekdays_long,
        [i.lower() for i in _weekdays_long],
        # TNO AVENGERS Format
        [i[:2] for i in _weekdays_short],
    ],
    MounthsProfile: [
        _months_short,
        [i.lower() for i in _months_short],
        _months_long,
        [i.lower() for i in _months_long],
    ],
    DailyProfile: [
        [str(i) for i in range(1, 25)],
        hours := [str(i) for i in range(24)],
        # TNO AVENGERS Format
        [f"H{i}" for i in hours],
    ],
}


def read_temporal_profiles(
    profiles_dir: PathLike,
    time_profiles_files_format: str = "timeprofiles*.csv",
    profile_csv_kwargs: dict[str, Any] = {},
    rtol: float = 1e-5,
) -> tuple[list[list[AnyTimeProfile]] | None, xr.DataArray | None]:
    """Read the temporal profiles csv files to the emiproc inventory format.

    The files for the time profiles are csv and must be all in the same directory
    named according to the argument `time_profiles_files_format`.

    If no files are found, this returns a warning.

    The format of the file will influence the name of the columns.
    Use the day of the weeks or the month names or the hour of the day to define the profile.

    :arg profiles_dir: The directory where the time profiles are stored.
    :arg time_profiles_files_format: The format of the filenames to read.
    :arg profile_csv_kwargs: Extra arguments to pass to the function
        :py:func:`emiproc.profiles.utils.read_profile_file` that reads the csv files.
    :arg rtol: The relative tolerance to use when checking if the ratios sum to 1.

    :return: A tuple of the profiles and the indexes, following the
        emiproc inventory profiles format.

    """

    # Note: The logic of this is a bit tricky because it has to handle
    #       the case where the profiles are speciated or not.

    profiles_dir = Path(profiles_dir)
    logger = logging.getLogger("emiproc.profiles.temporal.io.read_temporal_profiles")

    # List files with the expected format
    if profiles_dir.is_file():
        files = [profiles_dir]
        logger.info(f"File {profiles_dir=} found, will be used for timeprofiles.")
    else:
        if not profiles_dir.is_dir():
            raise ValueError(f"{profiles_dir=} is not a file or a directory.")
        files = list(profiles_dir.glob(time_profiles_files_format))
        if not files:
            logger.warning(
                "Cannot find any temporal profiles matching"
                f" {time_profiles_files_format=} in {profiles_dir=}.\n"
            )
            return None, None
        logger.info(
            f"Found {len(files)} files matching {time_profiles_files_format=} in"
            f" {profiles_dir=}"
        )

    out_profiles = AnyProfiles()
    indexes_list: list[xr.DataArray] = []
    for file in files:
        df = read_profile_file(file, **profile_csv_kwargs)
        possible_matching = {
            profile_type: colnames
            for profile_type, colnames_list in timprofile_colnames.items()
            for colnames in colnames_list
            if all(col in df.columns for col in colnames)
        }
        if not possible_matching:
            raise ValueError(
                f"Cannot find any matching time profile for {file=} with Columns"
                f" {df.columns}."
                "Please check the file format."
                "See more about time profiles file at "
                "https://emiproc.rtfd.io/en/latest/api.html#emiproc.profiles.temporal.io.read_temporal_profiles"
            )
        logger.info(f"{possible_matching=}")
        # Generate the profiles objects
        indexes = get_profiles_indexes(df)
        for profile_type, colnames in possible_matching.items():
            try:
                ratios = np.array([df[col] for col in colnames])
                if np.all(np.isclose(ratios.sum(axis=0), 1.0, rtol=rtol)):
                    # Ratios found
                    ratios = ratios
                elif np.all(np.isclose(np.mean(ratios, axis=0), 1.0, rtol=rtol)):
                    # Scaling factors found
                    ratios = ratios / ratios.sum(axis=0)
                else:
                    raise ValueError(
                        "Could not determine if scaling factors or ratios were given"
                        f" in {file=}.\n data:{ratios=} and \n"
                        f" mean:{np.mean(ratios, axis=0)} \n"
                        f" sum:{np.sum(ratios, axis=0)} \n Try to set {rtol=} to a"
                        " higher value if this is due to rounding erros."
                    )
                if isinstance(profile_type, tuple):
                    profiles = profile_type[0](
                        ratios=ratios.T, specific_day=profile_type[1]
                    )
                else:
                    profiles = profile_type(ratios.T)
            except Exception as e:
                raise ValueError(
                    f"Cannot create profile {profile_type=} from {file=} with {ratios=}"
                ) from e
            indexes += len(out_profiles)
            out_profiles.append(profiles)
            # Add a new dim which is the profile type
            indexes_list.append(
                indexes.expand_dims({"profile": [profile_type.__name__]})
            )

    combined_indexes = merge_indexes(indexes_list)

    composite_profiles, out_indexes = make_composite_profiles(
        out_profiles, combined_indexes
    )
    # Drop the profile dim
    if "profile" in out_indexes.dims:
        out_indexes = out_indexes.drop_vars("profile")

    return composite_profiles, out_indexes


@emiproc.deprecated
def from_csv(
    file: PathLike,
    profile_csv_kwargs: dict[str, Any] = {},
) -> dict[str | tuple[str, str] : AnyTimeProfile]:
    """Create the profile from a csv file.

    Based on the file, guess the correct profile
    to create.

    This is now deprectaed. Use :py:func:`read_temporal_profiles` instead.
    """

    df, cat_header, sub_header = read_profile_csv(file, **profile_csv_kwargs)

    if "Mon" in df.columns:
        # Weekly profile with 3 letters identification
        data_columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        profile = WeeklyProfile
    elif "mon" in df.columns:
        # Weekly profile with 3 letters identification
        data_columns = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        profile = WeeklyProfile
    elif " mon " in df.columns:
        data_columns = [" mon ", " tue ", " wed ", " thu ", " fri ", " sat ", " sun "]
        profile = WeeklyProfile
    elif "Monday" in df.columns:
        # Weekly profile with full name identification
        data_columns = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        profile = WeeklyProfile
    elif "Jan" in df.columns:
        # Yearly profile with 3 letters identification
        data_columns = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        profile = MounthsProfile
    elif " jan " in df.columns:
        data_columns = [
            " jan ",
            " feb ",
            " mar ",
            " apr ",
            " may ",
            " jun ",
            " jul ",
            " aug ",
            " sep ",
            " oct ",
            " nov ",
            " dec ",
        ]
        profile = MounthsProfile
    elif "January" in df.columns:
        # Yearly profile with full name identification
        data_columns = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        profile = MounthsProfile
    elif (
        "0" not in df.columns
        and "1" in df.columns
        and "24" in df.columns
        and "25" not in df.columns
    ):
        # Daily profile
        data_columns = [str(i) for i in range(1, 25)]
        profile = DailyProfile
    else:
        raise NotImplementedError(f"Cannot guess the profile from {file=}")

    # Check that all the data columns are in the file
    for col in data_columns:
        if not col in df.columns:
            raise ValueError(
                f"Cannot find {col=} in {file=}, required to build {profile=}"
            )

    # Read all rows to generate the profiles
    profiles = {}
    for _, row in df.iterrows():
        cat = row[cat_header]
        if sub_header is not None:
            sub = row[sub_header]
            key = (cat, sub)
        else:
            key = cat
        ratios = np.array([row[col] for col in data_columns])

        # Check if scaling factors are given instead of ratios
        if np.isclose(ratios.sum(), len(ratios)):
            ratios = ratios / ratios.sum()
        elif not np.isclose(ratios.sum(), 1.0):
            pass  # Already ratios
        else:
            raise ValueError(
                f"Cannot guess if {ratios=} are ratios or scaling factors for {row=} in"
                f" {file=}."
            )
        profiles[key] = profile(ratios=ratios)

    return profiles


def from_yaml(yaml_file: PathLike) -> list[AnyTimeProfile]:
    """Read a yml file containing a temporal profile.

    Only one temporal profile is currently accepted in the yaml definition.
    """
    logger = logging.getLogger("emiproc.profiles.from_yaml")
    yaml_file = Path(yaml_file)

    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        logger.warning(f"Empty yaml file {yaml_file=}")
        return []
    elif not isinstance(data, dict):
        raise ValueError(f"Invalid yaml file {yaml_file=}, expected to load a dict.")

    profiles = []

    # Create possible aliases for the names
    profiles_mapping: dict[AnyTimeProfile, list[str]] = {
        DailyProfile: ["diurn", "daily", "day"],
        SpecificDayProfile: [f"diurn_{day.value}" for day in SpecificDay],
        WeeklyProfile: ["weekly", "week"],
        MounthsProfile: ["season", "year", "monthly", "month"],
    }
    profile_of_key = {
        key: profile for profile, keys in profiles_mapping.items() for key in keys
    }

    _types_added = []
    # Check that the yaml does not contain any unkown key
    for key in data.keys():
        if not any(key in profile_names for profile_names in profiles_mapping.values()):
            logger.warning(f"Unknown key {key=} in {yaml_file=}")
            continue
        profile_class = profile_of_key[key]
        # Check that a profile of that type was not already added
        if profile_class in _types_added:
            raise ValueError(
                f"Cannot add {key=} to {yaml_file=} as a {profile_class=} was already"
                " added."
            )
        # add the profile
        ratios = data[key]
        # Check the ratio
        if not np.isclose(np.sum(ratios), 1.0):
            raise ValueError(
                f"{ratios=} in {yaml_file=} do not sum to 1 but {np.sum(ratios)}."
            )

        # Add additional information on the profiles if requried
        kwargs = {}
        if profile_class is SpecificDayProfile:
            # get the type of the profile (remove the diurn_ prefix)
            profile_type = '_'.join(key.split("_")[1:])
            # Add the selected day
            kwargs["specific_day"] = SpecificDay(profile_type)

        try:
            profile = profile_class(ratios=ratios, **kwargs)
        except Exception as e:
            raise ValueError(
                f"Cannot create profile {key=} from {yaml_file=} with {ratios=}"
            ) from e

        profiles.append(profile)

    if len(profiles) == 0:
        logger.warning(f"No profile found in {yaml_file=}")
    return profiles


def to_yaml(profiles: list[AnyTimeProfile], yaml_file: PathLike):
    """Write a list of profiles to a yaml file."""
    yaml_file = Path(yaml_file)
    yaml_file.parent.mkdir(parents=True, exist_ok=True)

    data = {}
    for profile in profiles:
        if isinstance(profile, DailyProfile):
            if isinstance(profile, SpecificDayProfile):
                key = f"diurn_{profile.specific_day.value}"
            else:
                key = "daily"
        elif isinstance(profile, WeeklyProfile):
            key = "weekly"
        elif isinstance(profile, MounthsProfile):
            key = "monthly"
        else:
            raise NotImplementedError(f"Cannot write {profile=}")

        data[key] = profile.ratios.tolist()

    with open(yaml_file, "w") as f:
        yaml.dump(data, f)
