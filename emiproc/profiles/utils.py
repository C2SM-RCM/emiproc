"""Utitlity functions for profiles."""
from __future__ import annotations
import logging
from os import PathLike
from pathlib import Path
from typing import Any, Type

import pandas as pd
import xarray as xr
import numpy as np

import emiproc
from emiproc.profiles import naming

logger = logging.getLogger(__name__)


def ratios_to_factors(ratios: np.ndarray) -> np.ndarray:
    """Convert ratios to factors."""

    return ratios * len(ratios)


def factors_to_ratios(factors: np.ndarray) -> np.ndarray:
    """Convert factors to ratios."""

    return factors / len(factors)


def type_in_list(object: Any, objects_list: list[Any]) -> bool:
    """Check if an object of the same type is in the list."""
    return any([isinstance(object, type(o)) for o in objects_list])


def remove_objects_of_type_from_list(object: Any, objects_list: list[Any]) -> list[Any]:
    """Remove objects of the same type from the list."""
    return [o for o in objects_list if not isinstance(object, type(o))]


def get_objects_of_same_type_from_list(
    object: Any, objects_list: list[Any]
) -> list[Any]:
    """Return the object of the same type from the list."""
    return [o for o in objects_list if isinstance(object, type(o))]


def get_desired_profile_index(
    profiles_indexes: xr.DataArray,
    cell: int | None = None,
    cat: str | None = None,
    sub: str | None = None,
) -> int:
    """Return the index of the desired profile.

    Smart function allowing to select based on desired attributes.
    It will check that the profile can be extracted.
    """

    # First check that the user did not omit a required dimension
    dims = profiles_indexes.dims
    if cell is None and "cell" in dims:
        raise ValueError("cell must be specified, as each cell has a specific profile.")
    if cat is None and "category" in dims:
        raise ValueError(
            "category must be specified, as each category has a specific profile."
        )
    if sub is None and "substance" in dims:
        raise ValueError(
            "substance must be specified, as each substance has a specific profile."
        )

    access_dict = {}

    # Add to the access the dimension specified,
    # If a dimension is specified but not in the dims, it means
    # we don't care becausse it is the same for all the dimension cooridnates
    if cell is not None and "cell" in dims:
        if cell not in profiles_indexes.coords["cell"]:
            raise ValueError(
                f"cell {cell} is not in the profiles indexes, "
                f"got {profiles_indexes.coords['cell']}"
            )
        access_dict["cell"] = cell
    if cat is not None and "category" in dims:
        if cat not in profiles_indexes.coords["category"]:
            raise ValueError(
                f"category {cat} is not in the profiles indexes, "
                f"got {profiles_indexes.coords['category']}"
            )
        access_dict["category"] = cat
    if sub is not None and "substance" in dims:
        if sub not in profiles_indexes.coords["substance"]:
            raise ValueError(
                f"substance {sub} is not in the profiles indexes, "
                f"got {profiles_indexes.coords['substance']}"
            )
        access_dict["substance"] = sub

    # Access the xarray
    desired_index = profiles_indexes.sel(**access_dict)

    # Check the the seleciton is just a single value
    if desired_index.size != 1:
        raise ValueError(
            f"More than one profile matches the selection: {desired_index}, got"
            f" {desired_index.size =}"
        )

    # Return the index as int
    return int(desired_index.values)


@emiproc.deprecated
def read_profile_csv(
    file: PathLike,
    cat_colname: str = "Category",
    sub_colname: str = "Substance",
    read_csv_kwargs: dict[str, Any] = {},
) -> tuple[pd.DataFrame, str, str | None]:
    """Read a profile csv file and return the dataframe, the category column name and the substance column name.

    Checks the name of the category and substances columns.
    """
    file = Path(file)

    df = pd.read_csv(file, **read_csv_kwargs)
    if cat_colname not in df.columns:
        raise ValueError(f"Cannot find '{cat_colname}' header in {file=}")

    if sub_colname in df.columns:
        sub_header = "Substance"
    else:
        sub_header = None
        logger = logging.getLogger("emiproc.profiles.read_cat_sub_from_csv")
        logger.warning(
            f"Cannot find 'Substance' header in {file=}.\n"
            "All substances will be treated the same way."
        )

    return df, cat_colname, sub_header


def get_profiles_indexes(
    df: pd.DataFrame,
    colnames: dict[str, list[str]] = naming.attributes_accepted_colnames,
) -> xr.DataArray:
    """Return the profiles indexes from the dataframe.

    The dataframe can contain any of the column matching to
    one of the dimensions allowed by the indexes.
    """

    # First get the dimensions present in the columns of the dataframe
    col_of_dim = {}
    for dim, colnames in colnames.items():
        columns = [col for col in colnames if col in df.columns]
        if len(columns) > 1:
            raise ValueError(
                f"Cannot find which column to use for {dim=} in {columns=}.\n"
                f"All columns refer to {dim=}."
            )
        elif len(columns) == 1:
            col_of_dim[dim] = columns[0]
        else:
            logger.debug(f"Cannot find column for {dim=}")
            pass
    logger.info(f"Found {col_of_dim=}")
    # Now get the values of the coords
    coords = {}
    for dim, col in col_of_dim.items():
        coords[dim] = df[col].unique()

    # Create the empty xarray
    indexes = xr.DataArray(
        # -1 means no index specified
        np.full([len(c) for c in coords.values()], -1),
        coords=coords,
        dims=list(coords),
    )

    # Fill the xarray with the indexes

    indexing_dict = dict(zip(coords, df[list(col_of_dim.values())].values.T))
    indexing_arrays = {}
    for coord, values in indexing_dict.items():
        indexing_arrays[coord] = xr.DataArray(values, dims=["index"])
    logger.debug(f"Indexing dataarray: {indexing_arrays=}")
    indexes.loc[indexing_arrays] = df.index

    return indexes


def load_country_tz(file: Path | None = None) -> pd.DataFrame:
    """Load the dataframe with the country timezones."""

    if file is None:
        file = Path(*emiproc.__path__).parent / "files" / "country_tz.csv"

    if not file.is_file():
        raise FileNotFoundError(f"Cannot find country_tz {file=}")

    return pd.read_csv(
        file,
        # File start with comment lines starting with '#'
        comment="#",
        # Index column must be 'iso3', the first one
        index_col=0,
        sep=";",
    )


if __name__ == "__main__":
    print(load_country_tz())
