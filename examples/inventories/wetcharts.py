"""Read WetCHARTs inventory and plot it."""

from emiproc.inventories.wetcharts import WetCHARTs
from emiproc.plots import plot_inventory
from emiproc import FILES_DIR


# Go to the website and download the file
# Change the path here to where you downloaded the file
file_path = FILES_DIR / "wetcharts" / "WetCHARTs_v1_3_3_2021.nc"

inv = WetCHARTs(
    file_path,
    # Specify the model number to select from the dataset.
    # If not specified, a mean of all models is used.
    # model=2924,
    # Set the name of the category (default is "wetcharts").
    category="wetland_emissions",
)


plot_inventory(inv)
