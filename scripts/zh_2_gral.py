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
#%%

import importlib
import emiproc.exports.gral
importlib.reload(emiproc.exports.gral)
from emiproc.exports.gral import export_to_gral
import pygg.grids
importlib.reload(pygg.grids)
from pygg.grids import GralGrid
grid = GralGrid(
        dz=2.0,
        stretch=1.0,
        nx=1368,
        ny=1296,
        nslice=9,
        sourcegroups="",
        xmin=2676060,
        xmax=2689740,
        ymin=1241540,
        ymax=1254500,
        latitude=47,
        crs=zh_inv.crs,
    )
grid.building_heights = np.zeros((grid.nx, grid.ny))
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
