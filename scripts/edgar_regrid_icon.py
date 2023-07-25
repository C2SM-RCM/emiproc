import pygeos
from pathlib import Path

from emiproc.grids import ICONGrid, WGS84_NSIDC
from emiproc.inventories.edgar import EDGAR_Inventory

from emiproc.regrid import remap_inventory
from emiproc.exports.icon import export_icon_oem


nc_file = r"C:\Users\thjo\Documents\EMPA\DATA_DAVID\EDGAR_v4.3.2\EDGAR_v4.3.2\CH4\ALL\v432_CH4_2012_IPCC_ALL.0.1x0.1.nc"
inv = EDGAR_Inventory(nc_file)

grid_file = Path(r"C:\Users\thjo\Documents\EMPA\DATA_DAVID\icon_grid_0053_R03B08_L.nc")
icon_grid = ICONGrid(grid_file)

print("icon grid loaded")

inv.to_crs(WGS84_NSIDC)

print("coordinates changed")

remaped_edgar = remap_inventory(inv, icon_grid, r"C:\Users\thjo\Documents\EMPA\scripts\weights.nc")

print("inventory remapped")

export_icon_oem(remaped_edgar, grid_file, r"C:\Users\thjo\Documents\EMPA\scripts\output_emiproc", country_resolution="110m")

print("inventory loaded")