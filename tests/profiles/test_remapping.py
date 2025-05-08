from emiproc.tests_utils import test_inventories, test_grids, temporal_profiles
import pandas as pd
from emiproc.regrid import calculate_weights_mapping
from emiproc.profiles.operators import get_weights_of_gdf_profiles, remap_profiles
import xarray as xr
import numpy as np


def test_remap_profiles():

    # Get the test variables required
    inv = test_inventories.inv.copy()
    weights_mapping = calculate_weights_mapping(
        inv.grid.gdf, test_grids.basic_grid_2.gdf
    )
    profiles_indexes = temporal_profiles.indexes_inv_catsubcell
    da_emission_weights = get_weights_of_gdf_profiles(
        inv.gdf,
        profiles_indexes,
    )
    profiles = temporal_profiles.three_composite_profiles

    new_profiles, new_indexes = remap_profiles(
        profiles,
        profiles_indexes,
        da_emission_weights,
        weights_mapping,
    )

    # check that the results are correct
    sel_dict = dict(substance="CH4", category="adf")
    this_new_indices = new_indexes.sel(**sel_dict)
    this_old_indices = profiles_indexes.sel(**sel_dict)

    # All cells have the same index in the given data, make sure still true for the test
    assert np.all(this_old_indices.values == this_old_indices.sel(cell=0).values)
    old_ratios = profiles.ratios[this_old_indices.sel(cell=0).values]
    for new_index in this_new_indices.values:
        assert new_index != -1, "all cells should have a valid index"
        this_ratios = new_profiles.ratios[new_index]
        np.testing.assert_almost_equal(
            this_ratios,
            old_ratios,
            err_msg=f"The new profiles should be the same as the old ones {this_ratios=} != {old_ratios=}",
        )

    # Check that the profiles with no emissions are set to -1
    assert all(
        new_indexes.sel(substance="CH4", category="liku") == -1
    ), "no emissions should give invalid profiles"

    # Check that this profile is a mixture with the correct weights
    sel_dict = dict(substance="CO2", category="liku")
    index_out = new_indexes.sel(**sel_dict, cell=0).values
    assert index_out != -1, "The profile should be valid"
    # This one has just one input cell with a profile
    index_in = profiles_indexes.sel(**sel_dict, cell=4).values
    # Check that the profiles are the same
    np.testing.assert_almost_equal(
        new_profiles.ratios[index_out],
        profiles.ratios[index_in],
        err_msg="The profiles should be the same",
    )

    # Now test some merged data with real combination
    sel_dict = dict(substance="CO2", category="adf")
    index_out = new_indexes.sel(**sel_dict, cell=0).values
    assert index_out != -1, "The profile should be valid"
    # This one has just one input cell with a profile
    index_in = profiles_indexes.sel(**sel_dict, cell=3).values
    assert index_in != -1, "The profile should be valid"
    # Check that the profiles are the same
    np.testing.assert_almost_equal(
        new_profiles.ratios[index_out],
        profiles.ratios[index_in],
        err_msg="The profiles should be the same",
    )
    # This one should be a mixture of the two profiles
    index_out2 = new_indexes.sel(**sel_dict, cell=1).values
    assert index_out2 != -1, "The profile should be valid"

    index_in2 = profiles_indexes.sel(**sel_dict, cell=4).values
    assert index_in2 != -1, "The profile should be valid"

    # It is a combination of the two cells
    # One weight is the area weight and the second is the emission weight
    # weights must be scaled with the respective area of that cell used
    weigths = np.array([(1.0 / 8.0) * (3.0 / 7.0), (3.0 / 8.0) * (4.0 / 7.0)])
    # As these are the only weights used, the sum of the weights should be 1
    weigths = weigths / weigths.sum()
    expected_profile = (
        profiles.ratios[index_in] * weigths[0] + profiles.ratios[index_in2] * weigths[1]
    )

    np.testing.assert_almost_equal(
        new_profiles.ratios[index_out2],
        expected_profile,
        err_msg="The profiles should be the same",
    )
