"""Test some basic features of the inventory profiles."""
#%%
import xarray as xr
from pathlib import Path
import numpy as np
import pytest

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
    weights= np.array([1, 2, 3])
    new_profile = weighted_combination(test_profiles2, weights=weights)
    check_valid_vertical_profile(new_profile)
    new_total_emissions = np.sum(weights) * new_profile.ratios
    previous_total_emissions = weights.dot(test_profiles2.ratios)
    # The combination should give the same as summing the emissions one by one
    assert np.allclose(new_total_emissions, previous_total_emissions), f"{new_total_emissions},{previous_total_emissions}"

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
        "substance": ["CH4", "CO2", "NH3"],
        "cell": [0, 1, 2, 3],
    },
)
corresponding_vertical_profiles
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
