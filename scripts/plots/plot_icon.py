"""plot an icon exported file.

This will generate a plot for each oem variable in the file.
"""

# %%
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import geopandas as gpd
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm

from emiproc import TESTS_DIR
from emiproc.plots.nclcmaps import cmap
from emiproc.grids import ICONGrid


# %% Load the file of icon
icon_file = TESTS_DIR / "export_icon" / "oem_gridded_emissions.nc"
out_folder = icon_file.parent / "plots"
out_folder.mkdir(exist_ok=True)
ds = xr.load_dataset(icon_file)

# %% choose the variables to plot
emiproc_generated_variables = [
    v for v in ds.variables if "created_by_emiproc" in ds[v].attrs
] + ["country_ids"]
# %% Loads the icon grid
n_cells = ds["cell"].size
corners = np.zeros((n_cells, 3, 2))
corners[:, :, 0] = ds["vlon"][ds["vertex_of_cell"] - 1].T
corners[:, :, 1] = ds["vlat"][ds["vertex_of_cell"] - 1].T
corners = np.rad2deg(corners)


# %%
# %matplotlib qt
matplotlib.style.use("default")
plt.ioff()
for var in emiproc_generated_variables:
    fig, ax = plt.subplots()

    da = ds[var]
    if not np.any(da):
        # No emissions at all
        continue

    min_val = da.min().values
    max_val = da.max().values

    if np.isnan(min_val) or np.isnan(max_val):
        print("skipping", var, "because of nan values in the data.")
        continue
    if min_val < 0:
        print("skipping", var, "because of negative values in the data.")
        continue
    poly_coll = PolyCollection(
        corners,
        cmap=cmap("WhViBlGrYeOrRe"),
        norm=LogNorm(),
        edgecolors="black",
        linewidth=0.04,
        antialiased=True,  # AA will help show the line when set to true
        alpha=0.6,
    )
    # Add the collection to the ax
    ax.add_collection(poly_coll)
    ax.set_ylim(np.min(corners[:, :, 1]), np.max(corners[:, :, 1]))
    ax.set_xlim(np.min(corners[:, :, 0]), np.max(corners[:, :, 0]))

    poly_coll.set_array(da)
    ax.set_title(da.attrs["long_name"])
    fig.colorbar(poly_coll)
    fig.savefig(out_folder / f"{var}.png")
    # fig.show()

# %% extra variables on the regions defined by country ids


icon_grid = ICONGrid(icon_file)
gdf = icon_grid.gdf
# %%

gdf["country_ids"] = ds["country_ids"].astype(int)
# Spatial join of the cell to get the regions=country
countries_shapes = {
    c_id: gdf[gdf["country_ids"] == c_id].geometry.union_all()
    for c_id in sorted(gdf["country_ids"].unique())
}
# %%
gdf_country = gpd.GeoDataFrame(
    {
        "geometry": countries_shapes.values(),
    }
    | {var: ds[var].values for var in ds.data_vars if ds[var].dims == ("region",)},
    index=countries_shapes.keys(),
)
gdf_country
# %%
gdf_country.explore("tz_shift")
# %%
