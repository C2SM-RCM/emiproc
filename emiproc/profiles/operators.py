import numpy as np
import xarray as xr
import geopandas as gpd

from emiproc.profiles.vertical_profiles import VerticalProfiles, VerticalProfile


def weighted_combination(
    profiles: VerticalProfiles, weights: np.ndarray
) -> VerticalProfile:
    """Combine the different profiles according to the specified weights."""

    return VerticalProfile(
        np.average(profiles.ratios, axis=0, weights=weights), profiles.height.copy()
    )


def combine_profiles(
    profiles: VerticalProfiles,
    profiles_indexes: xr.DataArray,
    dimension: str,
    weights: xr.DataArray,
) -> tuple[VerticalProfile, xr.DataArray]:
    """Combine profiles from multidimensional array over a specified dimension.

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

    # Perform the average on the profiles
    new_profiles = np.average(
        # Access the profile data
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
        new_profiles.reshape(-1, new_profiles.shape[-1]), axis=0, return_inverse=True
    )

    # Reshape the indexes of the new profiles
    shape = list(profiles_indexes.shape)
    shape.pop(profiles_indexes.dims.index(dimension))
    # These are now the indexes of this category
    new_indexes = xr.DataArray(
        inverse.reshape(shape),
        # Remove the coord of the given dimension
        coords=[c for dim, c in profiles_indexes.coords.items() if dim != dimension],
    )

    new_profiles = VerticalProfiles(unique_profiles, profiles.height)

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
