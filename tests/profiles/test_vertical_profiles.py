"""Test some basic features of the inventory profiles."""
import xarray as xr
from pathlib import Path
import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import Polygon

from emiproc.inventories import Inventory
from emiproc.inventories.utils import group_categories
from emiproc.profiles.vertical_profiles import (
    VerticalProfile,
    VerticalProfiles,
    check_valid_vertical_profile,
    get_mid_heights,
    resample_vertical_profiles,
    get_weights_profiles_interpolation,
    get_delta_h,
)
from emiproc.profiles.operators import (
    group_profiles_indexes,
    weighted_combination,
    combine_profiles,
    get_weights_of_gdf_profiles,
)

from emiproc.tests_utils import vertical_profiles


def test_test_profiles():
    for p in vertical_profiles.list_of_three:
        check_valid_vertical_profile(p)
    check_valid_vertical_profile(vertical_profiles.VerticalProfiles_instance)


def test_weights_interpolation():
    from_h = np.array([5, 10, 40, 100, 120])
    levels = np.array([2, 8, 10, 50, 80])
    w = get_weights_profiles_interpolation(from_h, levels)

    assert w[0, 0] == 2 / 5
    assert w[-1, -1] == 1.0
    assert w[1, 0] == 3 / 5


def test_rescale():
    # We just test it runs
    new_profiles = resample_vertical_profiles(
        *vertical_profiles.list_of_three,
        vertical_profiles.VerticalProfiles_instance,
        specified_levels=[0, 10, 20, 40, 50, 60],
    )

    # Visual check
    # start_from_0 = lambda arr: np.hstack([0, arr])
    # start_from_1 = lambda arr: np.hstack([1, arr])
    # for i, profile in enumerate(list_of_three):
    #     plt.figure()
    #     plot_profile = lambda h, r, **kwargs: plt.step(
    #         start_from_0(h),
    #         # np.cumsum(start_from_0(r)),
    #         start_from_0(r) / start_from_1(get_delta_h(h)),
    #         where="pre",
    #         **kwargs
    #     )
    #     plot_profile(new_profiles.height, new_profiles.ratios[i])
    #     plot_profile(profile.height, profile.ratios, linestyle='--')
    #     plt.show()


def test_mid_heights():
    assert np.allclose(
        get_mid_heights(np.array([10, 20, 40, 50])), np.array([5, 15, 30, 45])
    )


# test addition
def test_addition_vertical_profiles():
    new_profiles = (
        vertical_profiles.VerticalProfiles_instance
        + vertical_profiles.VerticalProfiles_instance
    )
    assert (
        new_profiles.n_profiles
        == 2 * vertical_profiles.VerticalProfiles_instance.n_profiles
    )

    # Check we can also do +=
    new_profiles += vertical_profiles.VerticalProfiles_instance
    assert (
        new_profiles.n_profiles
        == 3 * vertical_profiles.VerticalProfiles_instance.n_profiles
    )


#
def test_weighted_combination():
    weights = np.array([1, 2, 3])
    new_profile = weighted_combination(
        vertical_profiles.VerticalProfiles_instance, weights=weights
    )
    check_valid_vertical_profile(new_profile)
    new_total_emissions = np.sum(weights) * new_profile.ratios
    previous_total_emissions = weights.dot(
        vertical_profiles.VerticalProfiles_instance.ratios
    )
    # The combination should give the same as summing the emissions one by one
    assert np.allclose(
        new_total_emissions, previous_total_emissions
    ), f"{new_total_emissions},{previous_total_emissions}"


#
def test_invalid_vertical_profiles():
    # invalid profiles should raise some errors, let's check the main ones
    with pytest.raises(AssertionError):
        # Height 0
        check_valid_vertical_profile(VerticalProfile(np.array([1]), np.array([0])))
    with pytest.raises(AssertionError):
        # Sum of ratio != 1
        check_valid_vertical_profile(VerticalProfile(np.array([2]), np.array([10])))
    with pytest.raises(AssertionError):
        # non increasing heights
        check_valid_vertical_profile(
            VerticalProfile(np.array([0.5, 0.0, 0.5]), np.array([10, 40, 30]))
        )
    with pytest.raises(AssertionError):
        # ration < 0
        check_valid_vertical_profile(
            VerticalProfile(np.array([-0.5, 1.5]), np.array([10, 20]))
        )
    with pytest.raises(AssertionError):
        # Different length
        check_valid_vertical_profile(
            VerticalProfile(np.array([0.5, 0.5]), np.array([10, 20, 30]))
        )
    with pytest.raises(AssertionError):
        # nan value
        check_valid_vertical_profile(
            VerticalProfile(np.array([0.5, 0.5, np.nan]), np.array([10, 20, 30]))
        )


# select a subset

categories = ["test_cat", "test_cat2"]
selected_profile_indexes = vertical_profiles.corresponding_vertical_profiles.sel(
    {"category": categories}
)

# get the weights associatied to each profile


weights = get_weights_of_gdf_profiles(vertical_profiles.gdf, selected_profile_indexes)

# Make an average over the category taking care of putting the weights on the profiles where they should go


def test_combination_over_dimensions():
    new_profiles, new_indexes = combine_profiles(
        vertical_profiles.VerticalProfiles_instance,
        selected_profile_indexes,
        "cell",
        weights,
    )

    # Missing data
    assert all(new_indexes.sel(dict(substance="NH3")) == -1)
    # The missing data must have been merged (not all were missing)
    assert not any(new_indexes.sel(dict(substance="CH4")) == -1)
    # Access the profiles we want to compare (orginal of cat 0 and 1 for CH4 )
    assert np.allclose(
        new_profiles.ratios[
            new_indexes.sel(dict(substance="CH4", category="test_cat"))
        ],
        np.array([0.0, 0.3, 6.8 / 14, 3.0 / 14]),
    )
    assert np.allclose(
        new_profiles.ratios[
            new_indexes.sel(dict(substance="CH4", category="test_cat2"))
        ],
        np.array([0.08, 0.22, 0.7, 0.0]),
    )

    check_valid_vertical_profile(new_profiles)
    new_profiles, new_indexes = combine_profiles(
        vertical_profiles.VerticalProfiles_instance,
        selected_profile_indexes,
        "category",
        weights,
    )
    # Missing data
    assert all(new_indexes.sel(dict(substance="NH3")) == -1)
    assert new_indexes.sel(dict(cell=1, substance="CH4")) == -1
    # Access the profiles we want to compare (orginal of cells 0, 2, 3 for CH4 )
    indexes_of_CH4 = new_indexes.sel(dict(substance="CH4"))
    assert np.all(
        new_profiles.ratios[indexes_of_CH4[0]]
        == vertical_profiles.VerticalProfiles_instance.ratios[0]
    )
    assert np.all(
        new_profiles.ratios[indexes_of_CH4[2]]
        == vertical_profiles.VerticalProfiles_instance.ratios[0]
    )
    assert np.allclose(
        new_profiles.ratios[indexes_of_CH4[3]], np.array([0.04, 0.26, 0.4, 0.3])
    )

    check_valid_vertical_profile(new_profiles)
    new_profiles, new_indexes = combine_profiles(
        vertical_profiles.VerticalProfiles_instance,
        selected_profile_indexes,
        "substance",
        weights,
    )
    check_valid_vertical_profile(new_profiles)

def test_group_profiles_categories():

    weights = get_weights_of_gdf_profiles(vertical_profiles.inv.gdf, vertical_profiles.inv.v_profiles_indexes)
    new_profiles,new_indices = group_profiles_indexes(
        profiles = vertical_profiles.inv.v_profiles,
        profiles_indexes = vertical_profiles.inv.v_profiles_indexes,
        indexes_weights=weights,
        categories_group=vertical_profiles.inv_groups_dict,
    )
    
def test_group_profiles_substances():

    weights = get_weights_of_gdf_profiles(vertical_profiles.inv.gdf, vertical_profiles.inv.v_profiles_indexes)
    new_profiles,new_indices = group_profiles_indexes(
        profiles = vertical_profiles.inv.v_profiles,
        profiles_indexes = vertical_profiles.inv.v_profiles_indexes,
        indexes_weights=weights,
        categories_group=vertical_profiles.inv_groups_subs_dict,
        groupping_dimension="substance"
    )
    
    

def test_can_group_profiles_in_inv():
    group_categories(
        vertical_profiles.inv, vertical_profiles.inv_groups_dict
    )



def test_point_sources_are_just_appened():
    g_inv = group_categories(vertical_profiles.inv, {"new_cat": ["test_cat", "test_cat2", "test_cat3"]})

    assert len(g_inv.gdfs["new_cat"]) == 8
    # Check that the data was correctly appended
    assert np.all(g_inv.gdfs["new_cat"]["CO2"] == [1, 2, 3, 1.2, 2.7, 8, 1, 2])
    assert np.all(g_inv.gdfs["new_cat"]["CH4"] == [0, 0, 0, 4, 2, 8, 0, 0])
    assert np.all(g_inv.gdfs["new_cat"]["__v_profile__"] == [-1, 2, 1, 1, 2, -1, -1, -1])


#  Create hourly profiles
def check_valid_profiles_indexes_array(inv: Inventory, index_array: xr.DataArray):
    for cat, sub in inv._gdf_columns:
        print(cat)
        indexes = {}
        # get the values from the profile
        if "substance" in index_array.dims:
            if sub not in index_array["substance"]:
                raise ValueError(f"Missing profile for substance {sub}")
            indexes["substance"] = sub
        if "category" in index_array.dims:
            if cat not in index_array["category"]:
                raise ValueError(f"Missing profile for category {cat}")
            indexes["category"] = sub

        total_emission_da = xr.DataArray(
            inv.gdf[(cat, sub)], {"cells": np.arange(len(inv.gdf))}
        )

        profiles_this_catsub = index_array.loc[indexes]

        gridded_data = total_emission_da * profiles_this_catsub


if __name__ == "__main__":
    pytest.main([__file__])
    #test_group_profiles_substances()
