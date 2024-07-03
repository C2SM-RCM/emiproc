"""Convert the TNO inventory to a netcdf file on a regular raster grid.

By scaling the emissions with the time profiles, we export hourly emissions values.
"""
# %%
%load_ext autoreload
%autoreload 2
# %%
from datetime import datetime
from pathlib import Path

from emiproc.inventories.tno import TNO_Inventory
from emiproc.regrid import remap_inventory

from emiproc.utilities import Units
from emiproc.grids import RegularGrid
from emiproc.inventories.utils import group_categories

from emiproc.exports.netcdf import nc_cf_attributes
from emiproc.exports.hourly import export_hourly_emissions

from emiproc import FILES_DIR


# %%
# path to input inventory
input_path = FILES_DIR / r"test\tno\tno_test_minimal.nc"
profiles_dir = FILES_DIR / r"profiles\tno"
# output path and filename
output_dir = input_path.parent
output_path = output_dir / f"{input_path.stem}_with_hourly_emissions"

# %%
inv = TNO_Inventory(
    input_path,
    # Take only CH4
    substances_mapping={"co2_ff": "CO2","co2_bf": "CO2","ch4": "CH4"},
    profiles_dir=profiles_dir
)



# %% Regular grid domain, If you want to use the TNO grid, use output_grid = inv.grid


output_grid = RegularGrid(
    xmin=-40, ymin=20, xmax=80, ymax=80, nx=312, ny=208
)


remaped_tno = remap_inventory(
    inv=inv,
    grid=output_grid,
)
# %%


nc_metadata = nc_cf_attributes(
    author="LastName FirstName",
    contact="name@domain.com",
    title="Hourly Emissions for ...",
    source=input_path.name,
    institution="Instiution, Country",
)


# %% Export the temporal profiles of TNO
export_hourly_emissions(
    inv=remaped_tno,
    path=output_dir,
    netcdf_attributes=nc_metadata,
    start_time=datetime(2018, 1, 1),
    end_time=datetime(2018, 1, 2),
)
