"""Plot the tno inventory."""

# %%
from pathlib import Path
import matplotlib.pyplot as plt
from emiproc.plots import plot_inventory
from emiproc.inventories.tno import TNO_Inventory
from emiproc.inventories.utils import group_categories
from emiproc.regrid import remap_inventory

plt.style.use("default")
# %%
# path to input inventory
input_path = Path(r"C:\Users\coli\Documents\Data\TNO\TNO_GHGco_v1_1_year2015.nc")
# output path and filename
output_dir = input_path.parent
output_path = output_dir / f"{input_path.stem}_with_emissions.nc"

# %%
inv = TNO_Inventory(
    input_path,
    # Take only CH4
    substances_mapping={"co2_ff": "CO2", "co2_bf": "CO2"},
)

# %% Rempa the point sources
remapped = remap_inventory(inv, inv.grid)

# %%
grouped = group_categories(remapped, {"total": remapped.categories})


# %%
plot_inventory(
    grouped,
    cmap="magma_r",
    total_only=True,
    figsize=(16, 7),
    vmin=1e-8,
    vmax=1e2,
)

# %%
