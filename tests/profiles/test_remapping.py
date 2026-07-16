import pytest
import xarray as xr
import numpy as np

from emiproc.profiles.temporal.composite import CompositeTemporalProfiles
from emiproc.profiles.temporal.profiles import WeeklyProfile
from emiproc.tests_utils import test_inventories, test_grids, temporal_profiles
from emiproc.regrid import calculate_weights_mapping
from emiproc.profiles.operators import get_weights_of_gdf_profiles, remap_profiles


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


profiles_test = CompositeTemporalProfiles.from_ratios(
    np.array(
        [
            # Just ones
            [1] * 7,
            # Weekend is negative
            [1] * 5 + [-1] * 2,
            # Majority is negative, will turn out the same for ratios, because sum to 1
            [1] * 5 + [-1] * 2,
        ]
    ),
    rescale=True,
    types=[WeeklyProfile],
)

profiles_indexes_test = xr.DataArray(
    np.array([0, 1, 2]),
    dims=["cell"],
    coords={"cell": np.arange(3)},
)
da_unit_weights = xr.DataArray(
    np.array([1, 1, 1]),
    dims=["cell"],
    coords={"cell": np.arange(3)},
)
# Weights of the series with actual emissions
da_real_weights = xr.DataArray(
    np.array([7, 3, -3]),
    dims=["cell"],
    coords={"cell": np.arange(3)},
)
kwargs_test = dict(
    profiles=profiles_test,
    profiles_indexes=profiles_indexes_test,
)


def test_remap_same():

    new_profiles, new_indexes = remap_profiles(
        **kwargs_test,
        emissions_weights=da_unit_weights,
        weights_mapping={
            "output_indexes": np.array([0, 0]),
            "inv_indexes": np.array([0, 0]),
            "weights": np.array([1, 1]),
        },
    )

    assert np.allclose(
        new_profiles.ratios[0],
        profiles_test.ratios[0],
    )


def test_remap_profiles_with_negatives():
    weights_mapping = {
        "output_indexes": np.array([0, 0]),
        "inv_indexes": np.array([0, 1]),
        "weights": np.array([1, 1]),
    }

    new_profiles, new_indexes = remap_profiles(
        **kwargs_test,
        emissions_weights=da_unit_weights,
        weights_mapping=weights_mapping,
    )

    total_expected = np.array([[1 / 7 + 1 / 3] * 5 + [1 / 7 - 1 / 3] * 2])

    np.testing.assert_almost_equal(
        new_profiles.ratios,
        # Get to ratios
        total_expected / np.sum(total_expected),
        err_msg="The profiles should be the same",
    )


def test_remap_profiles_with_negatives_real():
    weights_mapping = {
        "output_indexes": np.array([0, 0]),
        "inv_indexes": np.array([0, 1]),
        "weights": np.array([1, 1]),
    }

    new_profiles, new_indexes = remap_profiles(
        **kwargs_test,
        emissions_weights=da_real_weights,
        weights_mapping=weights_mapping,
    )

    total_expected = np.array([[1 + 1] * 5 + [1 - 1] * 2])

    np.testing.assert_almost_equal(
        new_profiles.ratios,
        # Get to ratios
        total_expected / np.sum(total_expected),
        err_msg="The profiles should be the same",
    )


def test_remap_when_total_emission_is_negative():
    weights_mapping = {
        "output_indexes": np.array([0, 0]),
        "inv_indexes": np.array([0, 2]),
        "weights": np.array([1, 1]),
    }

    new_profiles, new_indexes = remap_profiles(
        **kwargs_test,
        emissions_weights=da_real_weights,
        weights_mapping=weights_mapping,
    )

    total_expected = np.array([[1 - 1] * 5 + [1 + 1] * 2])

    np.testing.assert_almost_equal(
        new_profiles.ratios,
        # Get to ratios
        total_expected / np.sum(total_expected),
        err_msg="The profiles should be the same",
    )


def test_with_negative_and_dont_merge():
    # test remapping now with the negative profile

    weights_mapping_with_negative = {
        "output_indexes": np.array([0, 0]),
        "inv_indexes": np.array([0, 2]),
        "weights": np.array([1, 2]),
    }

    with pytest.raises(ValueError):
        new_profiles, new_indexes = remap_profiles(
            **kwargs_test,
            weights_mapping=weights_mapping_with_negative,
            emissions_weights=da_real_weights,
            dont_merge=True,
        )
