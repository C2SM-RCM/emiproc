"""plot an icon exported file."""
# %%
from pathlib import Path
import xarray as xr
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib
from emiproc.plots.nclcmaps import cmap

#%% Load the file of icon
icon_file = r"C:\Users\coli\Documents\emiproc\scripts\.emiproc_weights_swiss_2_icon\icon_Zurich_R19B9_wide_DOM01_zh_ch_tno_combined.nc"
out_folder = Path(r"C:\Users\coli\Pictures\icon_zh_ch_tno_combined")
out_folder.mkdir(exist_ok=True)
ds = xr.load_dataset(icon_file)

#%% choose the variables to plot
emiproc_generated_variables = [
    v for v in ds.variables if "created_by_emiproc" in ds[v].attrs
] + ['country_ids']
# %% Loads the icon grid
n_cells = ds["cell"].size
corners = np.zeros((n_cells, 3, 2))
corners[:, :, 0] = ds["vlon"][ds["vertex_of_cell"] - 1].T
corners[:, :, 1] = ds["vlat"][ds["vertex_of_cell"] - 1].T
corners =  np.rad2deg(corners)



#%%
#%matplotlib qt
matplotlib.style.use('default')
plt.ioff()
for var in emiproc_generated_variables:
    fig, ax = plt.subplots()

    da = ds[var]
    if not np.any(da):
        # No emissions at all
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
    ax.set_title(da.attrs['long_name'])
    fig.colorbar(poly_coll)
    fig.savefig(out_folder/ f"{var}.png")
    #fig.show()
#%%
# %%
