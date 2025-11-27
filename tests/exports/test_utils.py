import numpy as np
import pandas as pd
import pytest
import xarray as xr

from emiproc.exports.utils import get_temporally_scaled_array
from emiproc.tests_utils import temporal_profiles, test_grids, test_inventories


time_range = pd.date_range("2017-12-30", "2018-01-02", freq="h")


def test_temporally_scaled_array_on_simple_inv():

    inv = test_inventories.inv.copy()

    profiles = temporal_profiles.three_composite_profiles
    profiles_indexes = temporal_profiles.indexes_inv_catsubcell

    inv.set_profiles(profiles, indexes=profiles_indexes)
    scaled = get_temporally_scaled_array(inv, time_range, sum_over_cells=False)

    assert "time" in scaled.dims
    assert len(scaled.time) == len(time_range)

    assert "category" in scaled.dims
    assert len(scaled.category) == len(inv.categories)

    assert "substance" in scaled.dims
    assert len(scaled.substance) == len(inv.substances)

    # scaled.sum("cell").stack(catsub=["category", "substance"]).plot.line(x="time")


def test_temporally_scaled_array_sum_over_cells():

    inv = test_inventories.inv.copy()

    profiles = temporal_profiles.three_composite_profiles
    profiles_indexes = temporal_profiles.indexes_inv_catsubcell

    inv.set_profiles(profiles, indexes=profiles_indexes)

    scaled = get_temporally_scaled_array(inv, time_range, sum_over_cells=True)


def test_temporally_scaled_array_missing_cell_profile_fails():

    bad_index = temporal_profiles.indexes_inv_catsubcell.copy()
    # Drop one cell on purpose
    bad_index = bad_index.sel(cell=[2, 3])

    inv = test_inventories.inv.copy()

    profiles = temporal_profiles.three_composite_profiles

    inv.set_profiles(profiles, indexes=bad_index)

    with pytest.raises(ValueError, match="Some cells have emissions but no profiles"):

        scaled = get_temporally_scaled_array(inv, time_range, sum_over_cells=False)


def test_temporally_scaled_array_missing_cell_profile_okay():

    bad_index = temporal_profiles.indexes_inv_catsubcell.copy()
    # Drop one cell on purpose
    bad_index = bad_index.sel(cell=[1, 2, 3, 4])

    inv = test_inventories.inv.copy()
    # Remove the first line of each column
    for col in inv._gdf_columns:
        inv.gdf.loc[0, col] = 0.0

    profiles = temporal_profiles.three_composite_profiles

    inv.set_profiles(profiles, indexes=bad_index)

    scaled = get_temporally_scaled_array(inv, time_range, sum_over_cells=False)
