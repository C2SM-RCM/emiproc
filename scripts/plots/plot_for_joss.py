"""Script to produce the plot in the emiproc paper in the joss."""

# %%
from pathlib import Path
import matplotlib.pyplot as plt
from emiproc.plots import plot_inventory
from emiproc.inventories.edgar import EDGARv8, download_edgar_files
from emiproc.inventories.utils import group_categories
from emiproc.regrid import remap_inventory
from emiproc.grids import RegularGrid

plt.style.use("default")
# %%
# path to input inventory
local_dir = Path("./edgar")
local_dir.mkdir(exist_ok=True)
download_edgar_files(local_dir, year=2022, substances=["CO2"])
inv = EDGARv8(local_dir / "v8.0_*.nc")
# output path and filename
output_dir = local_dir

# %% remap on a regular grid over Europe
grid = RegularGrid(
    name="Europe",
    xmin=-30,
    xmax=60,
    ymin=30,
    ymax=74,
    dx=0.1,
    dy=0.1,
)
# %% Rempa the point sources
remapped = remap_inventory(inv, grid)

# %%
grouped = group_categories(remapped, {"total": remapped.categories})


# %%
plot_inventory(
    grouped,
    cmap="magma_r",
    total_only=True,
    figsize=(14, 7),
    vmin=1e-8,
    vmax=1e2,
    out_dir=output_dir,
)

# %%
