"""Test some basic features of the inventory profiles."""
# %%
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
    create_time_serie,
    from_csv,
    from_yaml
)
from emiproc.profiles.operators import (
    weighted_combination,
    combine_profiles,
    get_weights_of_gdf_profiles,
)

# %% test

DailyProfile()
# %% Create test profiles
test_profiles = [
    DailyProfile(),
    WeeklyProfile(),
    MounthsProfile(),
    TemporalProfile(
        size=3,
        ratios=np.array([0.3, 0.2, 0.5]),
    ),
]

# %%
ratios = create_time_serie(
    start_time="2020-01-01",
    end_time="2020-01-31",
    profiles=[DailyProfile(), WeeklyProfile(), MounthsProfile()],
)
ratios

# TODO: move the following to examples
# %%
profiles_month_in_year = from_csv(
    r"C:\Users\coli\Documents\emiproc\files\profiles\copernicus\month_in_year.csv"
)
profiles_day_in_week = from_csv(
    r"C:\Users\coli\Documents\emiproc\files\profiles\copernicus\day_in_week.csv"
)
profiles_hour_in_day = from_csv(
    r"C:\Users\coli\Documents\emiproc\files\profiles\copernicus\hour_in_day.csv"
)
# %% plot the profiles
import matplotlib.pyplot as plt
import pandas as pd

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for name, profile in profiles_month_in_year.items():
    ax.plot(profile.ratios, label=name)
ax.legend()
# %%
tss = {}
for categorie in profiles_month_in_year.keys():
    tss[categorie] = create_time_serie(
        start_time="2020-01-01",
        end_time="2020-02-28",
        profiles=[
            profiles_month_in_year[categorie],
            profiles_day_in_week[categorie],
            profiles_hour_in_day[categorie],
        ],
    )

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for name, ts in tss.items():
    ax.plot(ts, label=name)
ax.legend()
# %%
yaml_dir = Path(r"C:\Users\coli\Documents\emiproc\files\profiles\yamls")

yaml_profiles = {}
for yml_file in yaml_dir.glob("*.yaml"):
    yaml_profiles[yml_file.stem] = from_yaml(yml_file)

# %%
tss = {}
for categorie in yaml_profiles.keys():
    tss[categorie] = create_time_serie(
        start_time="2020-01-01",
        end_time="2020-02-28",
        profiles=yaml_profiles[categorie],
    )

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for name, ts in tss.items():
    ax.plot(ts, label=name)
ax.legend()
# %%
