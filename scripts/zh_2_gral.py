"""Convert the mapluft inventory to gral."""
# %%
from pathlib import Path
from emiproc.inventories.zurich import MapLuftZurich, ZURICH_SOURCES
from emiproc.exports.gral import export_to_gral
from emiproc.tests_utils import TEST_OUTPUTS_DIR
# pygg module for gram gral preprocessing
from pygg.grids import GralGrid
import numpy as np

# %%
file = Path("/store/empa/em05/mapluft/mapLuft_2020_v2021.gdb")

zh_inv = MapLuftZurich(file, ['CO2'], convert_lines_to_polygons=False)
zh_inv.emission_infos = ZURICH_SOURCES
zh_inv 
#%% Read the gral grid from a generated geb
from emiproc.exports.gral import export_to_gral
from pygg.grids import GralGrid
grid = GralGrid.from_gral_rundir("/scratch/snx3000/lconstan/gramm_gral2/")

#%% Crop the invenotry over the gral grid 
from emiproc.inventories.utils import crop_with_shape
from emiproc.inventories.utils import get_total_emissions

zh_cropped = crop_with_shape(zh_inv, grid.get_bounding_polygon())

#%%
out_dir = TEST_OUTPUTS_DIR / 'test_gral_emissions'
out_dir.mkdir(exist_ok=True)
export_to_gral(
    zh_cropped,
    grid,
    out_dir,
    polygon_raster_size = 2
)
# %%
