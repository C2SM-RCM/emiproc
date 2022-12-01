"""Test some basic features of the inventory profiles."""
#%%
import xarray as xr
from pathlib import Path
import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import Polygon

from emiproc.inventories import Inventory
from emiproc.tests_utils.test_inventories import inv, inv_with_pnt_sources
from emiproc.profiles.vertical_profiles import (
    VerticalProfile,
    VerticalProfiles,
    check_valid_vertical_profile,
)
from emiproc.profiles.operators import weighted_combination

# %%
inv, inv_with_pnt_sources
#%%
test_profiles = [
    VerticalProfile(np.array([0, 0.3, 0.7, 0.0]), np.array([15, 30, 60, 100])),
    VerticalProfile(
        np.array([0.1, 0.3, 0.5, 0.0, 0.1]), np.array([10, 30, 40, 65, 150])
    ),
    VerticalProfile(np.array([1]), np.array([20])),
]
test_profiles2 = VerticalProfiles(
    np.array(
        [
            [0.0, 0.3, 0.7, 0.0],
            [0.1, 0.2, 0.7, 0.0],
            [0.0, 0.3, 0.2, 0.5],
        ]
    ),
    np.array([15, 30, 60, 100]),
)

for p in test_profiles:
    check_valid_vertical_profile(p)
check_valid_vertical_profile(test_profiles2)

#%%
def test_weighted_combination():
    weights = np.array([1, 2, 3])
    new_profile = weighted_combination(test_profiles2, weights=weights)
    check_valid_vertical_profile(new_profile)
    new_total_emissions = np.sum(weights) * new_profile.ratios
    previous_total_emissions = weights.dot(test_profiles2.ratios)
    # The combination should give the same as summing the emissions one by one
    assert np.allclose(
        new_total_emissions, previous_total_emissions
    ), f"{new_total_emissions},{previous_total_emissions}"


#%%
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


test_invalid_vertical_profiles()

#%%
gdf = gpd.GeoDataFrame(
    {
        ("test_cat", "CO2"): [i for i in range(4)],
        ("test_cat", "CH4"): [i + 3 for i in range(4)],
        ("test_cat", "NH3"): [2 * i for i in range(4)],
        ("test_cat2", "CO2"): [i + 1 for i in range(4)],
        ("test_cat2", "CH4"): [i + 1 for i in range(4)],
        ("test_cat2", "NH3"): [i + 1 for i in range(4)],
    },
    geometry=gpd.GeoSeries(
        [
            Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),
            Polygon(((0, 1), (0, 2), (1, 2), (1, 1))),
            Polygon(((1, 0), (1, 1), (2, 1), (2, 0))),
            Polygon(((1, 1), (1, 2), (2, 2), (2, 1))),
        ]
    ),
)
gdf
#%% Corresponding profiles integer array
# -1 is when the profile is not defined
corresponding_vertical_profiles = xr.DataArray(
    [
        [[0, -1, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0]],
        [[0, 0, -1, 0], [0, 0, -1, 0], [0, 1, 0, 0]],
    ],
    dims=("category", "substance", "cell"),
    coords={
        "category": ["test_cat", "test_cat2"],
        "substance": (substances := ["CH4", "CO2", "NH3"]),
        "cell": (cells :=[0, 1, 2, 3]),
    },
)
corresponding_vertical_profiles



#%% get the weights associatied to each profile
categories = ["test_cat", "test_cat2"]
sa = corresponding_vertical_profiles.sel({"category": categories})
weights = xr.full_like(sa, np.nan)
for cat in categories:
    for sub in substances:
        serie = gdf.loc[:, (cat,sub)]
        weights.loc[dict(category=cat, substance=sub)] = (
            # if depend dont get the total of the serie
            serie if 'cell' in sa.dims else sum(serie)
        )
weights

#%% Make an average over the category taking care of putting the weights on the profiles where they should go

new_profiles = np.average(
    # Access the profile data 
    test_profiles2.ratios[corresponding_vertical_profiles, :],
    axis=sa.dims.index('category'),
    # Weights must be extended on the last dimension such that a weight can take care of the whole time index
    weights=np.repeat(weights.to_numpy().reshape(*weights.shape, 1), len(test_profiles2.height), -1)
)
new_profiles

#%% Get a unique set of profiles

unique_profiles, inverse = np.unique(new_profiles.reshape(-1, new_profiles.shape[-1]), axis=0, return_inverse=True)
unique_profiles, inverse

#%% Create the new profiles xr.datarray of this group
shape = list(sa.shape)
shape.pop(sa.dims.index('category'))
# These are now the indexes of this category
this_category= inverse.reshape(shape)

# TODO: make it a loop over the different groups from the dict
# TODO: make sure to concatenate correctly the indexes
# TODO: try to make it a function working for merging over categories or cell as well
# %% Create hourly profiles
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


# %%
