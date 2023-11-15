from __future__ import annotations
import logging
from typing import TYPE_CHECKING
import numpy as np
import xarray as xr
import geopandas as gpd

from emiproc.profiles.vertical_profiles import (
    VerticalProfiles,
    VerticalProfile,
    resample_vertical_profiles,
)
from emiproc.profiles.temporal_profiles import (
    CompositeTemporalProfiles,
    SpecificDayProfile,
    TemporalProfile,
)
from emiproc.profiles.utils import get_objects_of_same_type_from_list
from emiproc.utilities import get_country_mask

if TYPE_CHECKING:
    from emiproc.grids import Grid


def concatenate_profiles(
    profiles_list: list[
        VerticalProfiles | list[list[TemporalProfile]] | CompositeTemporalProfiles
    ],
) -> VerticalProfiles | CompositeTemporalProfiles:
    # Check that all profiles are of the same type
    profile_type = type(profiles_list[0])
    if not all([type(p) == profile_type for p in profiles_list]):
        raise TypeError(
            "All profiles must be of the same type, got"
            f" {[type(p) for p in profiles_list]}"
        )
    if profile_type == VerticalProfiles:
        return resample_vertical_profiles(*profiles_list)
    elif profile_type == list:
        reduced_list = sum(profiles_list, [])
        return CompositeTemporalProfiles(reduced_list)
    elif profile_type == CompositeTemporalProfiles:
        return CompositeTemporalProfiles.join(*profiles_list)
    else:
        raise TypeError(f"Unknown profile type {profile_type}")


def weighted_combination(
    profiles: VerticalProfiles | list[TemporalProfile], weights: np.ndarray
) -> VerticalProfile | TemporalProfile:
    """Combine the different profiles according to the specified weights.

    Each ratio of the new profile follow the formula:

    .. math::
        r_{new} = \\frac{\\sum_{i}^n w_i r_i}{\\sum_{i}^n w_i}

    where :math:`r_i` is the ratio of the profile i and :math:`w_i` is the weight of the profile i.
    and :math`n` is the number of profiles.

    :arg profiles: The profiles to combine.
        If temporal, they must be all of the same type.
        If vertical, it must be a :py:class:`VerticalProfiles` object.
    :arg weights: The weights to use for the combination.
        See numpy.average() for more details on the weights.

    :return: The combined profile.
    """

    if len(profiles) != len(weights):
        raise ValueError(
            f"The number of profiles and weights must be the same: {len(profiles)=} !="
            f" {len(weights)=}"
        )

    if len(profiles) == 0:
        raise ValueError("No profiles given.")

    if isinstance(profiles, VerticalProfiles):
        return VerticalProfile(
            np.average(profiles.ratios, axis=0, weights=weights), profiles.height.copy()
        )
    elif isinstance(profiles, list):
        # Time profile, make sure they are all of the same type
        type_profile = type(profiles[0])
        if not all([isinstance(p, type_profile) for p in profiles]):
            raise TypeError(
                "All profiles must be of the same type, got"
                f" {[type(p) for p in profiles]}"
            )
        kwargs = {}
        if isinstance(profiles[0], SpecificDayProfile):
            kwargs["specific_day"] = profiles[0].specific_day
        return type_profile(
            ratios=np.average(
                np.array([p.ratios for p in profiles]), axis=0, weights=weights
            ),
            **kwargs,
        )


def weighted_combination_time(
    profiles: list[list[TemporalProfile]], weights: np.ndarray
) -> list[TemporalProfile]:
    """Combine the different profiles according to the specified weights.

    Same as :py:func:`weighted_combination` but for combinated time profiles.
    """
    out_profiles = []
    weights = weights.copy()
    # First read the types from the first combined profile
    for profile in profiles[0]:
        # Find in each list the profile of that type
        p_to_merge = []
        for i, p_list in enumerate(profiles):
            same_types = get_objects_of_same_type_from_list(profile, p_list)
            if len(same_types) > 1:
                raise ValueError(f"Duplicated profiles of type {type(profile)}")
            elif len(same_types) == 0:
                # No profile of this type don't use it in the combination
                weights[i] = 0
                p_to_merge.append(profile)
            else:
                p_to_merge.append(same_types[0])

        out_profiles.append(weighted_combination(p_to_merge, weights))

    return out_profiles


def combine_profiles(
    profiles: VerticalProfiles
    | list[list[TemporalProfile]]
    | CompositeTemporalProfiles,
    profiles_indexes: xr.DataArray,
    dimension: str,
    weights: xr.DataArray,
) -> tuple[VerticalProfiles | CompositeTemporalProfiles, xr.DataArray,]:
    """Combine profiles from multidimensional array by reducing over a specified dimension.

    The indexes and the weights but be of the same dimensions.

    :arg profiles: The profiles to use for merging.
    :arg profiles_indexes: The profiles indexes of the data.
    :arg dimension: The dimension along which the combination should be done.
    :arg weights: The weights of the data. In terms of emissions, it means
        the total emission of that data.
        Weights can be obtained through :py:func:`get_weights_of_gdf_profiles`

    :return: the new profiles and the indexes of the combined data
        in these new profiles.
        The indexes array has the dimension of combination removed.
    """

    logger = logging.getLogger(__name__)
    logger.debug(f"Combining {profiles_indexes=} on {dimension=} with {weights=}")
    # assert len(indexes.coords[dimension]) > 1, "Cannot combine over 1 dimension"
    if dimension not in profiles_indexes.dims:
        raise ValueError(
            f"Dimension {dimension} not in {profiles_indexes.dims}, cannot combine"
            " over it."
        )

    # Make sure that where the index is -1, the weights are 0
    mask_missing_profile = profiles_indexes == -1
    assert np.all(
        np.logical_or(
            weights.where(mask_missing_profile) == 0,
            np.isnan(weights.where(mask_missing_profile)),
        )
    ), "Some invalid profiles (=-1) don't have 0 weights."

    # Check that weights never sum to 0
    mask_sum_0 = weights.sum(dim=dimension) == 0
    if np.any(mask_sum_0):
        # We need to modify the weight for ensuring the success of the following operation
        weights = weights.copy()
        mask_invalid_weights = ~np.isnan(weights.where(mask_sum_0)).to_numpy()
        # Assign a value to the weights for teh average to work
        # at the end, we will assign invalid profiles to them
        # weights.to_numpy()[mask_invalid_weights] = 1
        weights = xr.where(mask_invalid_weights, 1, weights)
        # I assum this is okay as the weight value is only used on the axis where
        # the sum exists

    # Weights can be either given in the averaging dimension or in all the dimensions
    weights_missing_dims = set(profiles_indexes.dims) - set(weights.dims)
    if weights_missing_dims:
        # We need to broadcast the weights to the indexes
        weights = weights.broadcast_like(profiles_indexes)

    new_coords = [c for dim, c in profiles_indexes.coords.items() if dim != dimension]

    if isinstance(profiles, list):
        # Make it composite temporal profiles
        profiles = CompositeTemporalProfiles(profiles)
    # Get the ratios to use depending on the profile type
    if isinstance(profiles, (VerticalProfiles, CompositeTemporalProfiles)):
        # Access the profile data (this adds a new dim to the array)
        ratios_to_average = profiles.ratios[profiles_indexes, :]
        # Find the specified dimension of the profiles
        axis = profiles_indexes.dims.index(dimension)
        # Weights must be extended on the last dimension such that a weight can take care of the whole time index
        averaging_weights = np.repeat(
            weights.to_numpy().reshape(*weights.shape, 1),
            ratios_to_average.shape[-1],
            -1,
        )
        logger.debug(
            f"Creating averaged profiles with {ratios_to_average=} and"
            f" {averaging_weights=} on {axis=}"
        )
        # Perform the average on the profiles
        new_profiles = np.average(
            ratios_to_average,
            axis=axis,
            weights=averaging_weights,
        )
        # Reduce the size of the profiles by removing duplicates
        unique_profiles, inverse = np.unique(
            new_profiles.reshape(-1, new_profiles.shape[-1]),
            axis=0,
            return_inverse=True,
        )

        # Reshape the indexes of the new profiles
        shape = list(profiles_indexes.shape)
        shape.pop(profiles_indexes.dims.index(dimension))
        # These are now the indexes of this category
        logger.debug(f"Reshaping {inverse=} to {shape=} with {new_coords=}")
        new_indexes = xr.DataArray(
            inverse.reshape(shape),
            # Remove the coord of the given dimension
            coords=new_coords,
        )
        if isinstance(profiles, VerticalProfiles):
            new_profiles = VerticalProfiles(unique_profiles, profiles.height)
        elif isinstance(profiles, CompositeTemporalProfiles):
            new_profiles = CompositeTemporalProfiles.from_ratios(
                ratios=unique_profiles,
                types=profiles.types,
            )

    elif isinstance(profiles, list):
        if len(profiles_indexes.dims) > 2:
            raise NotImplementedError(
                "Currently no implementation for time profiles varying on more than 2"
                f" dimensions. Got {profiles_indexes.dims=}"
            )
        # get the name of the other dimension
        other_dim = [d for d in profiles_indexes.dims if d != dimension][0]

        # Iterate over the other dimension
        new_profiles = []
        new_indexes = []

        coords_of_merging = profiles_indexes.coords[dimension]
        coords_other_dim = profiles_indexes.coords[other_dim]
        for i, coord_other in enumerate(coords_other_dim):
            profiles_to_merge = []
            weights_to_merge = []
            for coord_of_merging in coords_of_merging:
                # Get the profiles of this category
                sel_dict = {other_dim: coord_other, dimension: coord_of_merging}
                profile_index = profiles_indexes.sel(**sel_dict).to_numpy()
                # Add the weights and profile we will merge
                profiles_to_merge.append(profiles[profile_index])
                weights_to_merge.append(weights.sel(**sel_dict).data)

            # Perform the combination
            new_profiles.append(
                weighted_combination_time(profiles_to_merge, weights_to_merge)
            )
            new_indexes.append(i)

        new_indexes = xr.DataArray(
            new_indexes, coords=[coords_other_dim], dims=[other_dim]
        )

    else:
        raise TypeError(
            f"Unknown profile type {type(profiles)}, must be VerticalProfiles or list"
        )

    # Set the invalid profiles back
    new_indexes = xr.where(mask_sum_0, -1, new_indexes)

    return new_profiles, new_indexes


def get_weights_of_gdf_profiles(
    gdf: gpd.GeoDataFrame, profiles_indexes: xr.DataArray
) -> xr.DataArray:
    """Read the emission values from the gdf to get the weights.

    Weights for missing data will be 0. This makes sure that if later
    group different profiles, the data will the merged profile from
    the other categories.

    :arg gdf: The gdf of an inventory, following the standard definition.
    :arg profiles_indexes: Indexes of the profiles to use.

    :return weights: A xarray containing the weights of the given profiles_indexes.
    """

    weights = xr.zeros_like(profiles_indexes, dtype=float)
    for col in gdf.columns:
        if isinstance(gdf[col].dtype, gpd.array.GeometryDtype):
            continue
        cat, sub = col

        # Case indexes is a subset of the data
        if (
            "category" in profiles_indexes.dims
            and cat not in profiles_indexes.coords["category"]
        ) or (
            "substance" in profiles_indexes.dims
            and sub not in profiles_indexes.coords["substance"]
        ):
            continue

        serie = gdf.loc[:, col]

        col_value = (
            # if depend dont get the total of the serie
            serie
            if "cell" in profiles_indexes.dims
            else sum(serie)
        )
        if "category" in profiles_indexes.dims and "substance" in profiles_indexes.dims:
            weights.loc[dict(category=cat, substance=sub)] = col_value
        elif "category" in profiles_indexes.dims:
            weights.loc[dict(category=cat)] += col_value
        elif "substance" in profiles_indexes.dims:
            weights.loc[dict(substance=sub)] += col_value
        else:
            weights += col_value

    # Where we there is no profile, we should not use the weight
    weights = xr.where(profiles_indexes == -1, 0, weights)

    return weights


def group_profiles_indexes(
    profiles: VerticalProfiles | list[list[TemporalProfile]],
    profiles_indexes: xr.DataArray,
    indexes_weights: xr.DataArray,
    categories_group: dict[str, list[str]],
    groupping_dimension: str = "category",
) -> tuple[VerticalProfiles | list[list[TemporalProfile]], xr.DataArray]:
    """Groups profiles and their indexes according to the given mapping.

    It is possible to group categories or substances or cells by setting
    the groupping_dimension to the corresponding dimension.

    :arg profiles: The profiles to use for merging.
    :arg profiles_indexes: The profiles indexes of the data.
    :arg indexes_weights: The weights of the data. In terms of emissions, it means
        the total emission of that data.
        Weights can be obtained through :py:func:`get_weights_of_gdf_profiles`
    :arg categories_group: A dictionary containing the mapping of the categories.
        The keys are the new categories and the values are the list of categories
        that should be grouped together.
    :arg groupping_dimension: The dimension along which the combination should be done.
        Default is "category".
    """
    if not groupping_dimension in profiles_indexes.dims:
        # if they don't depend on category, we don't need to do anything
        return profiles, profiles_indexes

    groups_indexes_list = []
    merged_profiles = []
    n_profiles = 0
    for group, categories in categories_group.items():
        cats_with_profiles = [
            c for c in categories if c in profiles_indexes.coords[groupping_dimension]
        ]
        group_coord = {groupping_dimension: [group]}
        sel_dict = {groupping_dimension: cats_with_profiles}
        if not cats_with_profiles:
            # No categories to group
            # Add unknown profiles
            group_indexes = profiles_indexes.sum(dim=groupping_dimension).expand_dims(
                group_coord
            )
            # set all the values to -1, no matter what the value is
            groups_indexes_list.append(
                xr.where(
                    group_indexes != -1,
                    -1,
                    group_indexes,
                )
            )

            continue

        # Combine the profiles that have to be grouped
        new_profiles, group_indexes = combine_profiles(
            profiles,
            profiles_indexes.sel(**sel_dict),
            dimension=groupping_dimension,
            weights=indexes_weights.sel(**sel_dict),
        )
        # Set that new coord to the dimension
        group_indexes = group_indexes.expand_dims(group_coord)

        # Offset the indexes for merging with the profiles
        groups_indexes_list.append(
            xr.where(
                group_indexes != -1,
                group_indexes + n_profiles,
                -1,
            )
        )
        n_profiles += len(new_profiles)
        merged_profiles.append(new_profiles)

    if not merged_profiles:
        coord_values = profiles_indexes.coords[groupping_dimension].to_numpy()
        raise ValueError(
            f"No profiles to group with {categories_group=}, on dimension"
            f" {groupping_dimension=} with {coord_values=}"
        )

    if isinstance(merged_profiles[0], VerticalProfiles):
        merged_profiles = sum(
            merged_profiles,
        )
    elif isinstance(merged_profiles[0], CompositeTemporalProfiles):
        merged_profiles = CompositeTemporalProfiles.join(
            *merged_profiles,
        )
    else:
        merged_profiles = sum(merged_profiles, [])

    new_indices = xr.concat(
        groups_indexes_list,
        dim=groupping_dimension,
    )

    return merged_profiles, new_indices


def country_to_cells(
    profiles: VerticalProfiles
    | list[list[TemporalProfile]]
    | CompositeTemporalProfiles,
    profiles_indexes: xr.DataArray,
    grid: Grid,
    country_mask_kwargs: dict[str, any] = {},
) -> tuple[VerticalProfiles | CompositeTemporalProfiles, xr.DataArray]:
    """Takes profiles given for countries and transform them to cells.

    The input profiles must have a 'country' dimension.
    The output profiles will have a 'cell' dimension.

    country_mask_kwargs are passed to :py:func:`emiproc.utilities.get_country_mask`.
    """

    # First check that the profiles are for countries
    if not isinstance(profiles_indexes, xr.DataArray):
        raise TypeError(
            f"Expected {profiles_indexes=} to be a xr.DataArray, got"
            f" {type(profiles_indexes)}"
        )
    if "country" not in profiles_indexes.dims:
        raise ValueError(
            f"Expected {profiles_indexes=} to have a country dimension, got"
            f" {profiles_indexes.dims}"
        )
    if "cell" in profiles_indexes.dims:
        raise ValueError(
            f"Expected {profiles_indexes=} to not have a cell dimension, got"
            f" {profiles_indexes.dims}"
        )

    # These will be the weights of the profiles
    countries_fractions: xr.DataArray = get_country_mask(
        grid, return_fractions=True, **country_mask_kwargs
    )

    # Check that the countries are all given
    # All the countries of the grid must be in the profiles
    missing_countries = set(countries_fractions.coords["country"].values) - set(
        profiles_indexes.coords["country"].values
    )

    if missing_countries:
        raise ValueError(
            f"Missing countries {missing_countries=} in {profiles_indexes=}"
        )

    if isinstance(profiles, list):
        # Make it composite temporal profiles
        profiles = CompositeTemporalProfiles(profiles)

    current_profile_index = len(profiles)
    new_indexes = []
    new_profiles = []
    new_profiles.append(profiles)
    # Now what we want to do is similar to groupping the indexes, but we want to
    # change the new dimension to cells instead of countries
    for cell in countries_fractions.coords["cell"]:
        # Get the indexes of the countries that are in this cell
        cell_fractions = countries_fractions.sel(cell=cell)
        # Get the indexes of the countries that are in this cell (non zero fracs)
        mask_non_zero = cell_fractions != 0
        cell_countries = cell_fractions.coords["country"].loc[mask_non_zero]
        if len(cell_countries) == 0:
            # No countries in this cell
            continue
        if len(cell_countries) == 1:
            # Only one country, no need to group, simply convert country to cell
            new_indexes.append(
                profiles_indexes.sel(country=cell_countries)
                .squeeze("country")
                .expand_dims({"cell": [cell]})
                .drop_vars("country")
            )
            continue
        # Combine the profiles according to the weights
        merged_profiles, merged_indexes = combine_profiles(
            profiles=profiles,
            profiles_indexes=profiles_indexes.sel(country=cell_countries).drop_vars(
                ["cell"]
            ),
            dimension="country",
            weights=cell_fractions.sel(country=cell_countries).drop_vars(["cell"]),
        )
        merged_indexes += current_profile_index
        current_profile_index += len(merged_profiles)
        new_profiles.append(merged_profiles)
        new_indexes.append(merged_indexes.expand_dims({"cell": [cell]}))

    new_indexes = xr.concat(new_indexes, dim="cell")

    return concatenate_profiles(new_profiles), new_indexes
