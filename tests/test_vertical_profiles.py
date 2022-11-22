"""Test some basic features of the inventory profiles."""
#%%
import pandas as pd
import xarray as xr
from pathlib import Path
import numpy as np
import geopandas as gpd
from typing import Any, Iterable
from shapely.geometry import Point, MultiPolygon, Polygon
from emiproc.inventories.utils import add_inventories, crop_with_shape
from emiproc.plots import explore_inventory, explore_multilevel
from emiproc.utilities import ProgressIndicator
from emiproc.regrid import (
    calculate_weights_mapping,
    geoserie_intersection,
    get_weights_mapping,
    remap_inventory,
)
from emiproc.inventories import Inventory
from emiproc.tests_utils.test_inventories import inv, inv_with_pnt_sources


# %%
inv, inv_with_pnt_sources
# %%
vertical_profiles = xr.DataArray(
    [[0, 0.3, 0.7, 0.0], [0.1, 0.8, 0.0, 0.1], [0.2, 0.1, 0.6, 0.1]],
    dims=(
        "substances",
        "levels",
    ),
    coords={
        "levels": (
            "levels",
            [0, 12, 18, 45],
            {"doc": "height (m) at which the level start"},
        ),
        "substances": ["CH4", "NH3", "CO2"],
    },
)
# %% Create hourly profiles 

for cat, sub in inv._gdf_columns:
    print(cat)
    indexes = {}
    # get the values from the profile 
    if 'substances' in vertical_profiles.dims:
        if sub not in vertical_profiles['substances']:
            raise ValueError(f"Missing profile for substance {sub}")
        indexes['substances'] = sub
    if 'categories' in vertical_profiles.dims:
        if cat not in vertical_profiles['categories']:
            raise ValueError(f"Missing profile for category {cat}")
        indexes['categories'] = sub
    
    total_emission_da = xr.DataArray(inv.gdf[(cat, sub)], {'cells': np.arange(len(inv.gdf))})

    profiles_this_catsub = vertical_profiles.loc[indexes]

    gridded_data = total_emission_da * profiles_this_catsub
    

# %%
