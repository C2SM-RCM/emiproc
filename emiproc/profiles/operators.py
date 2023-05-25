from __future__ import annotations
import numpy as np
import xarray as xr
import geopandas as gpd

from emiproc.profiles.vertical_profiles import VerticalProfiles, VerticalProfile
from emiproc.profiles.temporal_profiles import SpecificDayProfile, TemporalProfile
from emiproc.profiles.utils import get_objects_of_same_type_from_list


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
            f"The number of profiles and weights must be the same: {len(profiles)=} != {len(weights)=}"
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
                f"All profiles must be of the same type, got {[type(p) for p in profiles]}"
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
    profiles: VerticalProfiles | list[list[TemporalProfile]],
    profiles_indexes: xr.DataArray,
    dimension: str,
    weights: xr.DataArray,
) -> tuple[VerticalProfiles, xr.DataArray]:
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
    # assert len(indexes.coords[dimension]) > 1, "Cannot combine over 1 dimension"
    if dimension not in profiles_indexes.dims:
        raise ValueError(
            f"Dimension {dimension} not in {profiles_indexes.dims}, cannot combine over it."
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

    new_coords = [c for dim, c in profiles_indexes.coords.items() if dim != dimension]

    # Get the ratios to use depending on the profile type
    if isinstance(profiles, VerticalProfiles):
        # Perform the average on the profiles
        new_profiles = np.average(
            # Access the profile data (this adds a new dim to the array)
            profiles.ratios[profiles_indexes, :],
            # Find the specified dimension of the profiles
            axis=profiles_indexes.dims.index(dimension),
            # Weights must be extended on the last dimension such that a weight can take care of the whole time index
            weights=np.repeat(
                weights.to_numpy().reshape(*weights.shape, 1), len(profiles.height), -1
            ),
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
        new_indexes = xr.DataArray(
            inverse.reshape(shape),
            # Remove the coord of the given dimension
            coords=new_coords,
        )

        new_profiles = VerticalProfiles(unique_profiles, profiles.height)

    elif isinstance(profiles, list):
        if len(profiles_indexes.dims) > 2:
            raise NotImplementedError(
                f"Currently no implementation for time profiles varying on more than 2 dimensions. Got {profiles_indexes.dims=}"
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
            new_profiles.append(weighted_combination_time(profiles_to_merge, weights_to_merge))
            new_indexes.append(i)
        
        new_indexes = xr.DataArray(
            new_indexes,
            coords=[coords_other_dim],
            dims=[other_dim]
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
        if not cats_with_profiles:
            # No categories to group
            continue

        sel_dict = {groupping_dimension: cats_with_profiles}
        # Combine the profiles that have to be grouped
        new_profiles, group_indexes = combine_profiles(
            profiles,
            profiles_indexes.sel(**sel_dict),
            dimension=groupping_dimension,
            weights=indexes_weights.sel(**sel_dict),
        )
        # Set that new coord to the dimension
        group_indexes = group_indexes.expand_dims({groupping_dimension: [group]})

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
            f"No profiles to group with {categories_group=}, on dimension {groupping_dimension=} with {coord_values=}"
        )

    if isinstance(merged_profiles[0], VerticalProfiles):
        merged_profiles = sum(
            merged_profiles,
        )
    else:
        merged_profiles = sum(merged_profiles, [])

    new_indices = xr.concat(
        groups_indexes_list,
        dim=groupping_dimension,
    )

    return merged_profiles, new_indices
