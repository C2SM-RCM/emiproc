from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import xarray as xr

from emiproc.profiles.temporal.profiles import (
    SpecificDayProfile,
    TemporalProfile,
)
from emiproc.profiles.temporal.composite import CompositeTemporalProfiles

from emiproc.profiles.utils import (
    get_objects_of_same_type_from_list,
    profiles_to_scalingfactors_dataarray,
    ratios_dataarray_to_profiles,
)
from emiproc.profiles.vertical_profiles import (
    VerticalProfile,
    VerticalProfiles,
    resample_vertical_profiles,
)
from emiproc.utilities import get_country_mask

if TYPE_CHECKING:
    from emiproc.grids import Grid
    from emiproc.inventories import Inventory

logger = logging.getLogger(__name__)


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
    profiles: (
        VerticalProfiles | list[list[TemporalProfile]] | CompositeTemporalProfiles
    ),
    profiles_indexes: xr.DataArray,
    dimension: str,
    weights: xr.DataArray,
) -> tuple[VerticalProfiles | CompositeTemporalProfiles, xr.DataArray]:
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

    # Aligns exisiting coordinates
    weights = weights.broadcast_like(profiles_indexes).fillna(0)
    profiles_indexes = profiles_indexes.broadcast_like(weights).fillna(-1).astype(int)

    # Make sure that where the index is -1, the weights are 0
    mask_missing_profile = profiles_indexes == -1
    weights = xr.where(mask_missing_profile, 0.0, weights)

    if isinstance(profiles, list):
        # Make it composite temporal profiles
        profiles = CompositeTemporalProfiles(profiles)
    # Get the ratios to use depending on the profile type
    if not isinstance(profiles, (VerticalProfiles, CompositeTemporalProfiles)):
        raise TypeError(
            f"Unknown profile type {type(profiles)}, must be VerticalProfiles or"
            " CompositeTemporalProfiles"
        )
    # Access the profiles
    da_sf = profiles_to_scalingfactors_dataarray(profiles, profiles_indexes)

    logger.debug(
        f"Creating averaged profiles with {da_sf=} and" f" {weights=} on {dimension=}"
    )
    # Perform the average on the profiles
    new_scaling_factors = (da_sf * weights).sum(dimension)

    # Recover the profiles and indexes from the new scaling factors

    new_profiles, new_indexes = ratios_dataarray_to_profiles(
        new_scaling_factors.rename({"scaling_factors": "ratio"})
    )

    if isinstance(profiles, VerticalProfiles):
        # Rescale the unique profiles to sum to 1
        new_profiles = new_profiles / new_profiles.sum(axis=1).reshape(-1, 1)
        new_profiles = VerticalProfiles(new_profiles, profiles.height)
    elif isinstance(profiles, CompositeTemporalProfiles):
        new_profiles = CompositeTemporalProfiles.from_ratios(
            ratios=new_profiles,
            types=profiles.types,
            rescale=True,
        )
    else:
        raise TypeError(f"Unknown profile type {type(profiles)}")

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
            serie.loc[profiles_indexes.coords["cell"].values]
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

    new_indices = (
        xr.concat(
            groups_indexes_list,
            dim=groupping_dimension,
        )
        .fillna(-1)
        .astype(int)
    )

    return merged_profiles, new_indices


def country_to_cells(
    profiles: (
        VerticalProfiles | list[list[TemporalProfile]] | CompositeTemporalProfiles
    ),
    profiles_indexes: xr.DataArray,
    grid: Grid,
    country_mask_kwargs: dict[str, any] = {},
    ignore_missing_countries: bool = False,
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
    if isinstance(profiles, list):
        # Make it composite temporal profiles
        profiles = CompositeTemporalProfiles(profiles)

    # Check that the set makes sense
    if not np.all(profiles_indexes >= -1):
        raise ValueError(
            f"Expected {profiles_indexes=} to have all values >= -1, got"
            f" {profiles_indexes.min()}"
        )
    if not np.all(profiles_indexes < len(profiles)):
        raise ValueError(
            f"Expected {profiles_indexes=} to have all values <= {len(profiles)}, got"
            f" {profiles_indexes.max()}"
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

    if missing_countries and not ignore_missing_countries:
        raise ValueError(
            f"Missing countries {missing_countries=} in {profiles_indexes=}. "
            "Please check the profiles or set `ignore_missing_countries=False` ."
        )

    da_sf = profiles_to_scalingfactors_dataarray(profiles, profiles_indexes)

    sf_on_cell = da_sf.dot(countries_fractions, dim=["country"])

    profiles_array, new_indexes = ratios_dataarray_to_profiles(
        sf_on_cell.rename({"scaling_factors": "ratio"})
    )

    if isinstance(profiles, VerticalProfiles):
        new_profiles = VerticalProfiles(
            profiles_array / profiles_array.sum(axis=1).reshape((-1, 1)),
            profiles.height,
        )
    elif isinstance(profiles, CompositeTemporalProfiles):
        new_profiles = CompositeTemporalProfiles.from_ratios(
            profiles_array, types=profiles.types, rescale=True
        )
    else:
        raise TypeError(f"Unknown profile type {type(profiles)}")

    return new_profiles, new_indexes


def remap_profiles(
    profiles: VerticalProfiles | CompositeTemporalProfiles,
    profiles_indexes: xr.DataArray,
    emissions_weights: xr.DataArray,
    weights_mapping: dict[str, np.ndarray],
) -> tuple[VerticalProfiles | CompositeTemporalProfiles, xr.DataArray]:
    """Remap the profiles on a new grid.

    :arg profiles: The profiles to remap.
    :arg profiles_indexes: The indexes of the profiles.
    :arg emissions_weights: The weights of the emissions.
        Can be calculated using :py:func:`get_weights_of_gdf_profiles`.
    :arg weights_mapping: A dictionary containing the weights for the remapping.
        This is the result of :py:func:`emiproc.utilities.get_weights_mapping`.

    :return: The remapped profiles and the new indexes.

    """
    if isinstance(profiles, VerticalProfiles):
        raise NotImplementedError("Vertical profiles remapping not implemented yet.")
    if not isinstance(profiles, CompositeTemporalProfiles):
        raise TypeError(
            f"Invalid profile type {type(profiles)}, must be {CompositeTemporalProfiles}"
        )

    assert "cell" in profiles_indexes.dims, "Expected cell dimension in indexes"
    assert "cell" in emissions_weights.dims, "Expected cell dimension in weights"

    # Check that all the cells are in the profiles
    missing_cells_in_profiles = set(weights_mapping["inv_indexes"]) - set(
        profiles_indexes["cell"].values
    )
    if missing_cells_in_profiles:
        # Get a new weights mapping without the missing cells
        mask_missing = np.isin(
            weights_mapping["inv_indexes"], list(missing_cells_in_profiles)
        )
        weights_mapping = {k: v[~mask_missing] for k, v in weights_mapping.items()}

    # Get the emissions weights at the remapping index
    da_weights_of_remapping = xr.DataArray(
        weights_mapping["weights"],
        dims=["cell"],
        coords={"cell": weights_mapping["inv_indexes"]},
    )
    da_weights = (
        emissions_weights.sel(cell=weights_mapping["inv_indexes"])
        * da_weights_of_remapping
    )

    # Set the profiles needed on the remapping cells
    profiles_to_get_remapped = profiles_indexes.sel(cell=weights_mapping["inv_indexes"])

    # access the profiles to get the ratios at each inpout
    da_ratios = (
        xr.DataArray(
            profiles.ratios[profiles_to_get_remapped],
            dims=[*profiles_to_get_remapped.dims, "ratio"],
            coords={
                **profiles_to_get_remapped.coords,
                "ratio": range(profiles.ratios.shape[1]),
            },
            # Remove the profiles with no ratios (will be set to nan)
            # This assumes that no profile = no contribution, so only the other ratios in the cell will have an impact
        )
        .where(profiles_to_get_remapped != -1)
        .assign_coords(output_cell=("cell", weights_mapping["output_indexes"]))
    )

    # Do the actual remapping calculations
    da_remapped_ratios = (
        (da_ratios * da_weights)
        .groupby("output_cell")
        .mean()
        .rename({"output_cell": "cell"})
    )

    new_ratios, new_indices = ratios_dataarray_to_profiles(da_remapped_ratios)
    new_profiles = CompositeTemporalProfiles.from_ratios(
        new_ratios, profiles.types, rescale=True
    )

    return new_profiles, new_indices


def add_profiles(
    inv1: Inventory,
    inv2: Inventory,
) -> tuple[CompositeTemporalProfiles, xr.DataArray]:
    """Add the profiles of two inventories together.

    :arg inv1: The first inventory.
    :arg inv2: The second inventory.

    :return: The sum of the two profiles.
    """

    indexes_name = "t_profiles_indexes"
    profiles_name = "t_profiles_groups"

    indexes1 = getattr(inv1, indexes_name)
    indexes2 = getattr(inv2, indexes_name)

    # Aligns exisiting coordinates
    indexes1, indexes2 = xr.broadcast(indexes1, indexes2)
    # Add the missing and convert again to integers
    indexes1 = indexes1.fillna(-1).astype(int)
    indexes2 = indexes2.fillna(-1).astype(int)

    weights1 = get_weights_of_gdf_profiles(inv1.gdf, profiles_indexes=indexes1)
    weights2 = get_weights_of_gdf_profiles(inv2.gdf, profiles_indexes=indexes2)
    weights1, weights2 = xr.broadcast(weights1, weights2)
    # Add the missing
    weights1 = weights1.fillna(0)
    weights2 = weights2.fillna(0)

    profiles_name = "t_profiles_groups"
    profiles1: CompositeTemporalProfiles = getattr(inv1, profiles_name)
    profiles2: CompositeTemporalProfiles = getattr(inv2, profiles_name)

    # Make the profiles have the same sub-profiles included
    # This will make scaling factors of 1 when a sub-profile is missing
    all_types = set(sum([p.types for p in [profiles1, profiles2]], []))
    # Careful here, becaue the types will change the order of position
    profiles1 = profiles1.broadcast(all_types)
    profiles2 = profiles2.broadcast(all_types)

    # Make sure they are the same in the new profiles
    assert (
        profiles1.types == profiles2.types
    ), "Profiles not same type. Please raise an issue on Github."

    sf1 = profiles_to_scalingfactors_dataarray(profiles1, indexes1)
    sf2 = profiles_to_scalingfactors_dataarray(profiles2, indexes2)

    # Multiply the scaling factors by the weights
    sf_total = sf1 * weights1 + sf2 * weights2

    ratios, indexes = ratios_dataarray_to_profiles(
        sf_total.rename({"scaling_factors": "ratio"})
    )

    # Create the new profiles object
    profiles = CompositeTemporalProfiles.from_ratios(
        ratios, rescale=True, types=profiles1.types
    )

    return profiles, indexes


def add_constant_profile_to_missing_cells(inv: Inventory) -> Inventory:

    out_inv = inv.copy(profiles=False)

    for profile_name, indexes_name in [
        ("t_profiles_groups", "t_profiles_indexes"),
        ("v_profiles", "v_profiles_indexes"),
    ]:
        profiles = getattr(inv, profile_name)
        indexes = getattr(inv, indexes_name)

        if profiles is None or indexes is None:
            continue

        if "cell" not in indexes.dims:
            # No profile means constant profile
            out_inv.set_profiles(
                profiles,
                indexes=indexes,
            )
            continue

        # Append the constant profile
        if isinstance(profiles, CompositeTemporalProfiles):
            # new_profile = CompositeTemporalProfiles.join(
            #    profiles, CompositeTemporalProfiles([[]])
            # )
            new_profile = profiles
        else:
            raise NotImplementedError(
                "function `add_constant_profile_to_missing_cells` not implemented for "
                f"{type(profiles)}"
            )

        missing_cells = set(np.arange(len(inv.grid))) - set(indexes["cell"].values)

        new_indexes = xr.concat(
            [
                indexes,
                xr.full_like(indexes.isel(cell=0), -1).expand_dims(
                    {"cell": list(missing_cells)}
                ),
            ],
            dim="cell",
        )

        out_inv.set_profiles(
            new_profile,
            indexes=new_indexes,
        )

    return out_inv
