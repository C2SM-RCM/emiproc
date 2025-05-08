"""Utitlity functions for profiles."""

from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Type

import numpy as np
import pandas as pd
import xarray as xr

import emiproc
from emiproc.profiles import naming

if TYPE_CHECKING:
    from emiproc.profiles.temporal.composite import CompositeTemporalProfiles
    from emiproc.profiles.vertical_profiles import VerticalProfiles

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
    object: Any, objects_list: list[Any], exact_type: bool = False
) -> list[Any]:
    """Return the object of the same type from the list."""
    func = isinstance if exact_type else lambda x, y: type(x) == y
    return [o for o in objects_list if isinstance(object, type(o))]


def check_valid_indexes(
    indexes: xr.DataArray, profiles: VerticalProfiles | CompositeTemporalProfiles = None
) -> None:
    """Check that the given indexes are valid.

    :raises ValueError: if the indexes are not valid
    """

    if not isinstance(indexes, xr.DataArray):
        raise TypeError(f"Indexes should be an xarray.DataArray, got {type(indexes)=}")
    # Check the dtype of the indexes, should be int
    if indexes.dtype not in [int, np.int64]:
        raise TypeError(f"Indexes should be of type int, got {indexes.dtype=}")

    # check all the dims names are valid
    dims_not_allowed = set(indexes.dims) - set(naming.type_of_dim.keys())
    if len(dims_not_allowed) > 0:
        raise ValueError(
            f"Indexes are not allowed to contain {dims_not_allowed=}, "
            f"allowed dims are {naming.type_of_dim.keys()}"
        )
    # Make sure no coords has duplicated values
    for dim in indexes.dims:
        if indexes.coords[dim].size == 0:
            raise ValueError(f"Indexes are empty for {dim=}")
        if len(indexes.coords[dim]) != len(np.unique(indexes.coords[dim])):
            raise ValueError(
                f"Indexes are not valid, they contain duplicated values for {dim=}:"
                f" {indexes.coords[dim]}"
            )

    if profiles is not None:
        # Check that the max value of the index is given in the profiles
        if indexes.max().values >= len(profiles):
            raise ValueError(
                "Indexes are not valid, they contain values that are not in the"
                f" profiles.Got {indexes.max().values=} but profiles has"
                f" {len(profiles)=}"
            )


def get_desired_profile_index(
    profiles_indexes: xr.DataArray,
    cell: int | None = None,
    cat: str | None = None,
    sub: str | None = None,
    type: str | None = None,
) -> int:
    """Return the index of the desired profile.

    Smart function allowing to select based on desired attributes.
    It will check that the profile can be extracted.
    """

    dimensions_values = {
        "cell": cell,
        "category": cat,
        "substance": sub,
        "type": type,
    }

    # First check that the user did not omit a required dimension
    dims = profiles_indexes.dims
    for dim_name, dim_value in dimensions_values.items():
        if dim_value is None and dim_name in dims:
            raise ValueError(
                f"dimension {dim_name=} is required because the indexes differentiate"
                " it. Please specify it."
            )

    access_dict = {}

    # Add to the access the dimension specified,
    # If a dimension is specified but not in the dims, it means
    # we don't care becausse it is the same for all the dimension cooridnates
    for dim_name, dim_value in dimensions_values.items():
        if dim_value is not None and dim_name in dims:
            if dim_value not in profiles_indexes.coords[dim_name]:
                raise ValueError(
                    f"{dim_name} {dim_value} is not in the profiles indexes, "
                    f"got {profiles_indexes.coords[dim_name]}"
                )
            access_dict[dim_name] = dim_value

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

    If a column of the index has some nan values, the profiles
    with nan values will fill other other values when no profile is
    given.
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
        coord_values = df[col].unique()
        expected_type = naming.type_of_dim.get(dim, str)
        coord_values = coord_values.astype(expected_type)
        if expected_type == str:
            # Clean the strings from tabs and spaces at the sides
            coord_values = [v.strip() for v in coord_values]
            df[col] = df[col].str.strip()
        coords[dim] = coord_values
    # Create the empty xarray
    indexes = xr.DataArray(
        # -1 means no index specified
        np.full([len(c) for c in coords.values()], -1),
        coords=coords,
        dims=list(coords),
    )
    logger.debug(f"Created {indexes=}")

    # Fill the xarray with the indexes

    indexing_dict = dict(zip(coords, df[list(col_of_dim.values())].values.T))
    indexing_arrays = {}
    for coord, values in indexing_dict.items():
        expected_type = naming.type_of_dim.get(dim, str)
        indexing_arrays[coord] = xr.DataArray(
            values.astype(expected_type), dims=["index"]
        )
    logger.debug(f"Indexing dataarray: {indexing_arrays=}")
    indexes.loc[indexing_arrays] = df.index

    # Finally fill missing profiles with "common profiles" (where there was a nan value)

    for dim in indexes.dims:
        if "nan" in indexes.coords[dim].values:
            common_values = indexes.sel(**{dim: "nan"})
            # Drop the nan indices
            specific_indexes = indexes.sel(
                **{dim: [v for v in indexes.coords[dim].values if v != "nan"]}
            )

            # Get the missing indices
            indexes = xr.where(specific_indexes == -1, common_values, specific_indexes)

    return indexes


def read_profile_file(file: PathLike, **kwargs: Any) -> pd.DataFrame:
    """Read any kind of profile file and return the dataframe."""
    default_kwargs = {
        "comment": "#",
        "sep": r";|\t|,",
        "engine": "python",  # This is needed to use regex in sep
    }
    for key, value in default_kwargs.items():
        if key not in kwargs:
            kwargs[key] = value
    try:
        logger.log(emiproc.PROCESS, f"Reading {file}")
        df = pd.read_csv(
            file,
            **kwargs,
        )
    except Exception as e:
        raise ValueError(
            f"Could not read profiles from {file}. Please check the format of the file."
        ) from e

    df.rename(columns={col: col.strip() for col in df.columns}, inplace=True)
    # Look a the columns and see if we find columns with names we should clean
    for col in df.columns:
        if col in naming.all_reserved_colnames:
            df[col] = df[col].str.strip()
    return df


def merge_indexes(indexes: list[xr.DataArray]) -> xr.DataArray:
    """Merge together arrays of indexes.

    When merging the indexes, we assume that some arrays have more dimensions
    and thus are more specific than others.
    This implies that the a more specific index will overwrite a less specific one.
    However sometimes conflict can arise. If you have two indexes array and each
    of them are more specific on a dimension.
    Currently function will raise an error in this case.
    """

    # First get the coords of the indexes
    coords = {}
    for index in indexes:
        for coord, values in index.coords.items():
            if coord not in coords:
                coords[coord] = values
            else:
                coords[coord] = np.unique(np.concatenate([coords[coord], values]))
    logger.debug(f"Found {coords=}")

    # Create the empty xarray
    merged_indexes = xr.DataArray(
        # -1 means no index specified
        np.full([len(c) for c in coords.values()], -1),
        coords=coords,
        dims=list(coords),
    )
    logger.debug(f"Created {merged_indexes=}")

    # sort the indexes by the number of dimensions
    indexes.sort(key=lambda x: len(x.dims))

    # Fill the xarray with the indexes
    # Here we assume that the more dimensions we have the more specifc we will be
    # However if the dimensions are others
    specifed_dims = []
    for index_da in indexes:
        # Get the dims that are not specified yet
        dims_to_specify = [dim for dim in index_da.dims if dim not in specifed_dims]
        # Get the dims that have already been specifed but not in this index
        dims_to_overwrite = [dim for dim in specifed_dims if dim not in index_da.dims]
        # Check that there is no conflict
        if len(dims_to_overwrite) > 0:
            raise ValueError(
                f"Cannot merge indexes {indexes=} because of conflict on"
                f" {dims_to_overwrite=}.Please only specify profiles which are being"
                " more specific on the dimensions of others."
            )

        # Get coords where the indexes are given on a single dimension
        indexes_to_set = index_da.stack(z=index_da.dims)
        logger.debug(f"Stacked: {indexes_to_set=}")
        indexes_to_set = indexes_to_set[indexes_to_set != -1]
        indexing_arrays = {}
        for dim in index_da.dims:
            indexing_arrays[dim] = xr.DataArray(indexes_to_set[dim].values, dims="z")
        logger.debug(f"Indexing dataarray: {indexing_arrays=}")
        profile_index = xr.DataArray(indexes_to_set.values, dims="z")
        logger.debug(f"Profiles to set : {profile_index=}")

        # Now fill the indexes xarray
        merged_indexes.loc[indexing_arrays] = profile_index

        logger.debug(f"After Filled: {merged_indexes=}")
        # Add the dims to the specified dims
        specifed_dims.extend(dims_to_specify)

    return merged_indexes


def profiles_to_scalingfactors_dataarray(
    profiles: CompositeTemporalProfiles | VerticalProfiles, indexes: xr.DataArray
) -> xr.DataArray:
    """Convert a profiles object to a ratios DataArray.

    When there is no profile for a given index, the scaling factors are set to 1.0.

    :arg profiles: The profiles object.
    :arg indexes: The indexes DataArray.

    :returns: A DataArray with the scaling factors.
    """

    sf = profiles.scaling_factors
    return xr.DataArray(
        sf[indexes],
        dims=[*indexes.dims, "scaling_factors"],
        coords={
            **indexes.coords,
            "scaling_factors": range(sf.shape[-1]),
        },
        # Remove the profiles with no ratios (will be set to nan)
        # This assumes that no profile = no contribution, so only the other ratios in the cell will have an impact
    ).where(indexes != -1, 1.0)


def ratios_dataarray_to_profiles(
    da: xr.DataArray, rounding_decimals: int | None = None
) -> tuple[np.ndarray, xr.DataArray]:
    """Convert a dataarray of ratios to a profiles array and the indexes compatible for emiproc.

    :arg da: DataArray with the ratios.
        Must contain a 'ratio' dimension. Other dimensions must be the ones
        allowed by the emiproc profiles.
    :arg rounding_decimals: The number of decimals to round the profiles to.
        This can be useful to reduce the number of unique profiles.

    :returns: A tuple with the profiles array and the indexes DataArray.
        The profiles array is a 2D array which can be set at ratios in a Profile object.
        The indexes DataArray is an array that can be set to the indexes of an inventory.

    """
    assert "ratio" in da.dims
    other_coords = {dim: da.coords[dim] for dim in da.dims if dim != "ratio"}

    if len(other_coords) == 0:
        # Add a dummy dimension to allow stacking
        da = da.expand_dims("dummy")
        other_coords = {"dummy": da["dummy"]}

    # Stack the other dimensions
    da_stacked = da.stack(profiles=other_coords.keys())

    # Set the output profiles indexes
    da_profiles_indexes = da_stacked.sum(dim="ratio")
    mask_valid = da_profiles_indexes != 0
    da_profiles_indexes.values = -np.ones(da_profiles_indexes.shape, dtype=int)

    values = da_stacked.sel(profiles=mask_valid).fillna(0.0).values
    if rounding_decimals is not None:
        values = values.round(decimals=rounding_decimals)

    # Get the unique profiles to avoid duplicates
    unique_profiles, unique_indices = np.unique(
        # Fill the nans as the unique functions does not like them
        values,
        axis=-1,
        return_inverse=True,
    )

    # Set the profiles indexes
    da_profiles_indexes.loc[mask_valid] = unique_indices
    profiles_indexes = da_profiles_indexes.unstack(fill_value=-1)

    if "dummy" in profiles_indexes.dims:
        profiles_indexes = profiles_indexes.squeeze("dummy").drop_vars("dummy")

    return unique_profiles.T, profiles_indexes.astype(int)
