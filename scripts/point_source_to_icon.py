# %%
from pathlib import Path
from emiproc.inventories import Inventory
from emiproc.grids import WGS84, ICONGrid, WGS84_PROJECTED

from emiproc.inventories.utils import group_categories
from emiproc.regrid import remap_inventory
from emiproc.utilities import SEC_PER_YR
from emiproc.exports.icon import export_icon_oem

from shapely.geometry import Point 
import geopandas as gpd
import yaml

# invpath = "/scratch/snx3000/dbrunner/grid4OEM/emission_pntSrc_h202.xml"

# %%
inv=Inventory.from_gdf(gdfs={'point_sources': gpd.GeoDataFrame(
    {
    "H2O2": [4*SEC_PER_YR,4*SEC_PER_YR],
    },
    geometry=[Point(8.4,49.0), Point(9.2,48.8)],
              crs="EPSG:4326",
              )}
)


#%% Load the icon grid
grid_file = Path(
    r"/scratch/snx3000/dbrunner/grid4OEM/BASE_KA_DOM01.nc"
)
icon_grid = ICONGrid(grid_file)

#%%
# Convert to a planar crs for the remapping to work
inv.to_crs(WGS84_PROJECTED)

#%%
remapped_point = remap_inventory(inv, icon_grid, grid_file.parent / f"remapped_point_{grid_file.stem}")


#%%
export_icon_oem(remapped_point, grid_file, grid_file.with_stem(f"{grid_file.stem}_with_h2o2_emissions"))

