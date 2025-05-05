"""Read WetCHARTs inventory and plot it."""

from emiproc.inventories.wetcharts import WetCHARTs
from emiproc.plots import plot_inventory
from emiproc import FILES_DIR
import xarray as xr


# Download the file and change the path here or anywhere you want.
file_path = FILES_DIR / "wetcharts" / "WetCHARTs_v1_3_3_2021.nc"

inv = WetCHARTs(
    file_path,
    # Specify the model number to select from the dataset.
    model=2924,
    # Set the name of the category (default is "wetcharts").
    category="wetland_emissions",
)


plot_inventory(inv)
