# %% [markdown]
# # Plot Profiles
# This notebook demonstrates how to use the `plot_profile` function to visualize different temporal profiles.

# %%
from emiproc.profiles.plot import plot_profile
from emiproc.tests_utils.temporal_profiles import read_test_copernicus, read_test_yamls


# %% [markdown]
# ## Copernicus Profiles
# Let's start by plotting the Copernicus profiles.

# %% [code]
copernicus_profiles, copernicus_indexes = read_test_copernicus()

plot_profile(copernicus_profiles, ignore_limit=True)


# %% [markdown]
# ## YAML Profiles
# Now, let's plot the YAML profiles.

# %% [code]
yaml_profiles = read_test_yamls()
for name, profile in yaml_profiles.items():
    if not profile:
        print(f"Skipping empty profile list for {name}")
        continue
    fig, ax = plot_profile(profile)
    fig.suptitle(name)
    fig.tight_layout()
fig


# %%  plot a single profile

plot_profile(profile[0])


# %%
