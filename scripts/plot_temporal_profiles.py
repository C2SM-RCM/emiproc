"""Show some basic features of the temporal profiles and plot."""

# %%
from datetime import datetime, timedelta
from os import PathLike
import xarray as xr
from pathlib import Path
import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import Polygon
from emiproc.exports.icon import TemporalProfilesTypes, make_icon_time_profiles

from emiproc.inventories import Inventory
from emiproc.tests_utils.test_inventories import inv, inv_with_pnt_sources
from emiproc.profiles.temporal_profiles import (
    TemporalProfile,
    DailyProfile,
    WeeklyProfile,
    MounthsProfile,
    create_scaling_factors_time_serie,
    from_csv,
    from_yaml,
)
from emiproc import FILES_DIR

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
ratios = create_scaling_factors_time_serie(
    start_time="2020-01-01",
    end_time="2020-01-31",
    profiles=[DailyProfile(), WeeklyProfile(), MounthsProfile()],
)
ratios

# %%
profiles_dir = FILES_DIR / "profiles" / "copernicus"
profiles_month_in_year = from_csv(
    profiles_dir / "timeprofiles-month_in_year.csv"
)
profiles_day_in_week = from_csv(
    profiles_dir / "timeprofiles-day_in_week.csv"
)
profiles_hour_in_day = from_csv(
    profiles_dir / "timeprofiles-hour_in_day.csv"
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
csv_profiles = {}
for categorie in profiles_month_in_year.keys():
    csv_profiles[categorie] = [
        profiles_month_in_year[categorie],
        profiles_day_in_week[categorie],
        profiles_hour_in_day[categorie],
    ]

    tss[categorie] = create_scaling_factors_time_serie(
        start_time="2020-01-01",
        end_time="2020-02-28",
        profiles=csv_profiles[categorie],
    )


fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for name, ts in tss.items():
    ax.plot(ts, label=name)
ax.legend()
# %%
yaml_dir = Path(FILES_DIR / "profiles" / "yamls")

yaml_profiles = {}
for yml_file in yaml_dir.glob("*.yaml"):
    yaml_profiles[yml_file.stem] = from_yaml(yml_file)

# %%
tss = {}
for categorie in yaml_profiles.keys():
    tss[categorie] = create_scaling_factors_time_serie(
        start_time="2020-01-01",
        end_time="2020-02-28",
        profiles=yaml_profiles[categorie],
    )

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for name, ts in tss.items():
    ax.plot(ts, label=name)
ax.legend()
# %%


dss = make_icon_time_profiles(
    csv_profiles,
    {"DE": 0, "FR": 1, "USA": -5},
    #{"DE": 0, "FR": 1},
    TemporalProfilesTypes.HOUR_OF_YEAR,
    2020,
    out_dir="./test_icon_oem_profiles",
)
dss['hourofyear']
# %%

# plot each of the countries
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for country in dss["hourofyear"]["country"].values:
    ax.plot(
        dss["hourofyear"]["Road_Transport"].sel(country=country),
        label=country,
        alpha=0.5,
    )
ax.legend()

# %%
