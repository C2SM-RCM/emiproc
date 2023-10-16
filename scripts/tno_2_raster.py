"""Convert the TNO inventory to a netcdf file on a regular raster grid."""

#%%
from pathlib import Path

from emiproc.inventories.tno import TNO_Inventory

from emiproc.utilities import Units
from emiproc.grids import RegularGrid
from emiproc.inventories.utils import group_categories

from emiproc.exports.netcdf import nc_cf_attributes
from emiproc.exports.rasters import export_raster_netcdf
from emiproc.exports.profiles import export_inventory_profiles

#%%
# path to input inventory
input_path = Path("/scratch/snx3000/lconstan/TNO_GHGco_v4_1_highres_year2018.nc")
# output path and filename
output_dir = input_path.parent
output_path = output_dir / f"{input_path.stem}_with_emissions.nc"

#%%
inv = TNO_Inventory(
    input_path,
    # Take only CH4
    substances=["CH4", 'CO2'],
)


# %%  mapping from input to output species
mapping = {
    "energy": ["A"],
    "industrial": ["B"],
    "residential": ["C"],
    "natural_gas": ["D", "E"],
    "transport": ["F1", "F2", "F3", "F4", "G", "H", "I"],
    "waste": ["J"],
    "agriculture": ["K", "L"],
}


groupped_tno = group_categories(inv, mapping)


# %% Regular grid domain, If you want to use the TNO grid, use output_grid = inv.grid


output_grid = RegularGrid(
    xmin=4.97, ymin=45.4875, xmax=11.21, ymax=48.5, nx=312, ny=208
)

#%%


nc_metadata = nc_cf_attributes(
    author="Stephan Henne",
    contact="stephan.henne@empa.ch",
    title="Gridded annual emissions",
    source=input_path.name,
    institution="Empa Duebendorf, Switzerland",
)

# Export to a netcdf raster on the output grid, remapping will be performed
export_raster_netcdf(
    groupped_tno, output_path, output_grid, nc_metadata, unit=Units.KG_PER_M2_PER_S
)


# %% Export the temporal profiles of TNO
export_inventory_profiles(
    inv=groupped_tno,
    output_dir=output_dir,
    grid=output_grid,
    netcdf_attributes=nc_metadata,
)
# %%
