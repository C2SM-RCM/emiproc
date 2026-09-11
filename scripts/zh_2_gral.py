"""Convert the mapluft inventory to gral.

For this exercise we need an additional package called pygg.
"""

# %%
# %load_ext autoreload
# %autoreload 2
# %%
from pathlib import Path
from emiproc.exports.gral import export_to_gral
from emiproc.inventories.zurich.duck import DuckDBInventory
from emiproc.tests_utils import TEST_OUTPUTS_DIR
from emiproc.inventories.utils import crop_with_shape, drop
from emiproc.inventories.utils import group_categories, validate_group
from emiproc.inventories.zurich.gral_groups import ZH_CO2_DUCK_GROUPS
from emiproc.inventories.zurich.categories_info import (
    ZURICH_GROUPPED_SOURCES,
)

# pygg module for gram gral preprocessing
from pygg.grids import GralGrid
import numpy as np

# %%

YEAR = 2024
duckdbs_dir = Path("/input/CH_EMISSIONS/MapLuft/Emissions/duckdbs")
gral_run_dir = Path("/project/leob/gg/zurich/Zurich_CO2_clean")
inv_file = duckdbs_dir / f"emikat_v2026a.db"
substances = ["CO2"]

# %%
inv_zh = DuckDBInventory(
    inv_file,
    year=YEAR,
    substances=[s.lower() for s in substances],
)
inv_zh.to_crs("LV03")

# %% Process inventory
ignore_categories = ["brandfeuerschaden", "feuerwerk", "tabakwaren"]
zh_cleaned = drop(inv_zh, categories=ignore_categories)
zh_groupped = group_categories(zh_cleaned, ZH_CO2_DUCK_GROUPS, ignore_missing=False)

# %% Read the gral grid from a generated geb
grid = GralGrid.from_gral_rundir(gral_run_dir)

# %%
inv_export = zh_groupped
inv_export.emission_infos = ZURICH_GROUPPED_SOURCES

out_dir = TEST_OUTPUTS_DIR / "test_gral_emissions"
out_dir.mkdir(exist_ok=True)
export_to_gral(inv_export, grid, out_dir, polygon_raster_size=5)
# %%
