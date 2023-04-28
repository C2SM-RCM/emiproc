# %%
from pathlib import Path
from emiproc.grids import WGS84, ICONGrid, WGS84_PROJECTED
from emiproc.inventories.tno import TNO_Inventory
from emiproc.inventories.categories_groups import TNO_2_GNFR

from emiproc.plots import explore_inventory, explore_multilevel, plot_inventory
from emiproc.inventories.utils import group_categories
from emiproc.regrid import remap_inventory

# %%
nc_file = r"C:\Users\coli\Documents\emiproc\files\TNO_6x6_GHGco_v4_0\TNO_GHGco_v4_0_year2018.nc"

# %%


inv = TNO_Inventory(nc_file)


# %% Load the icon grid
grid_file = Path(r"C:\Users\coli\Documents\ZH-CH-emission\icon_europe_DOM01.nc")
icon_grid = ICONGrid(grid_file)

# %%
# Convert to a planar crs for the remapping to work

inv.to_crs(WGS84_PROJECTED)

# %%
remaped_tno = remap_inventory(
    inv, icon_grid, grid_file.parent / f"remap_tno2{grid_file.stem}"
)

# %%
groupped = group_categories(remaped_tno, TNO_2_GNFR)

# %%
from emiproc.exports.icon import export_icon_oem

export_icon_oem(
    remaped_tno, grid_file, grid_file.with_name(f"{grid_file.stem}_with_tno_emissions")
)
# %%
