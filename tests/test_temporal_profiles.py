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
from emiproc.profiles.temporal_profiles import (
    TemporalProfile,
    DailyProfile,
    WeeklyProfile,
    MounthsProfile,
    create_time_serie
)
from emiproc.profiles.operators import (
    weighted_combination,
    combine_profiles,
    get_weights_of_gdf_profiles,
)

#%% test 

DailyProfile()
#%% Create test profiles
test_profiles = [
    DailyProfile(),
    WeeklyProfile(),
    MounthsProfile(),
    TemporalProfile(
        size=3,
        ratios=np.array([0.3, 0.2, 0.5]),
    )
]

#%%
ratios = create_time_serie(
    start_time="2020-01-01",
    end_time="2020-01-31",
    profiles=[DailyProfile(), WeeklyProfile(), MounthsProfile()],
)
ratios

