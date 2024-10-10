import numpy as np
import pandas as pd
import xarray as xr

from emiproc.exports.utils import get_temporally_scaled_array
from emiproc.tests_utils import temporal_profiles, test_grids, test_inventories


def test_temporally_scaled_array_on_simple_inv():

    inv = test_inventories.inv.copy()

    profiles = temporal_profiles.three_composite_profiles
    profiles_indexes = temporal_profiles.indexes_inv_catsubcell

    inv.set_profiles(profiles, indexes=profiles_indexes)
    time_range = pd.date_range("2017-12-30", "2018-01-02", freq="h")
    scaled = get_temporally_scaled_array(inv, time_range, sum_over_cells=False)

    assert "time" in scaled.dims
    assert len(scaled.time) == len(time_range)

    assert "category" in scaled.dims
    assert len(scaled.category) == len(inv.categories)

    assert "substance" in scaled.dims
    assert len(scaled.substance) == len(inv.substances)

    # scaled.sum("cell").stack(catsub=["category", "substance"]).plot.line(x="time")
