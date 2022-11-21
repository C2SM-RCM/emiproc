#%%
from emiproc.inventories.tno import TNO_Inventory


from emiproc.plots import plot_inventory


# %%
nc_file = "TNO_GHGco_v4_0_year2018.nc"

# %%


inv = TNO_Inventory(nc_file)

#%%

plot_inventory(
    inv,
    figsize=(20,9),
    out_dir=r"C:\Users\coli\Documents\ZH-CH-emission\Data\CHEmissionen\plots\TNO"
)

# %%
