from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr

from emiproc.grids import RegularGrid
from emiproc.inventories import Inventory
from emiproc.profiles.temporal.composite import CompositeTemporalProfiles
from emiproc.profiles.temporal.profiles import MounthsProfile


class SaunoisInventory(Inventory):
    """Inventory based on Saunois estimates of methane emissions.

    https://doi.org/10.5194/essd-12-1561-2020

    You can download the data there but note that the current implementation
    had some other files as sources, but very similar.
    https://www.icos-cp.eu/GCP-CH4-2024

    """

    def __init__(self, saunois_files: list[Path]):
        """Initialize the inventory.


        Parameters
        ----------

        saunois_files :
            List of paths to the Saunois files.
            Here each netcdf is named after the category.
            If you donwload from the ICOS website, you will have to rename the files.
            Or change the code for this inventory.
        """
        super().__init__()

        da = xr.concat(
            [
                xr.open_dataset(file)["flux"]
                .rename(file.stem)
                .expand_dims(category=[file.stem])
                for file in saunois_files
            ],
            dim="category",
        )

        # Drop the lev dimension and add the substance dimension
        assert da["lev"].size == 1
        da = da.squeeze("lev").expand_dims(substance=["CH4"])

        # Set the coords to be str
        da["substance"] = da["substance"].astype(str)
        da["category"] = da["category"].astype(str)

        # replace the lat lon by cell
        da_stacked_all = da.stack(cell=("lon", "lat"))
        da_stacked = da_stacked_all.drop_vars(["lat", "lon"])
        # Use a simple integer index for the cell
        da_stacked["cell"] = np.arange(da_stacked.sizes["cell"])

        self.grid = RegularGrid.from_centers(
            x_centers=da["lon"].values,
            y_centers=da["lat"].values,
            name="Saunois_Grid",
            rounding=2,
        )

        # Unit conversion
        # Units are gCH4/m2/day
        # "g CH4 m-2 d-1" -> "kg / year / cell"
        # To convert from day to year, we have to multiplly each month by the number of days in the month and then sum the months totals
        da_stacked_total = (
            da_stacked
            * xr.DataArray(
                np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]),
                dims="time",
                coords={"time": da_stacked.coords["time"]},
            )
        ).sum(dim="time")
        # kg/g * m2/cell
        conversion_factor = 1e-3 * self.grid.cell_areas
        da_total = da_stacked_total * xr.DataArray(
            conversion_factor, dims="cell", coords={"cell": da_stacked["cell"]}
        )

        # Convert to pandas
        df = (
            da_total.stack(catsub=("category", "substance"))
            .drop_vars(["cell"])
            .to_pandas()
        )
        self.gdf = gpd.GeoDataFrame(df, geometry=self.grid.gdf.geometry)
        self.gdfs = {}

        # Generate the profiles
        da_profiles = da_stacked.stack(profiles=("substance", "category", "cell"))
        # Convert to ratios
        da_ratios = (da_profiles / da_profiles.sum(dim="time")).fillna(0.0)
        mask_valid = da_ratios.sum(dim="time") != 0

        # Create the profiles indexes
        da_profiles_indexes = da_ratios.sum(dim="time")
        da_profiles_indexes.values = -np.ones(da_profiles_indexes.shape, dtype=int)

        # Set the values to linear indices
        da_valid_profiles = da_ratios.sel(profiles=mask_valid)

        # Many profiles are exactly the same, so we will simplify the profiles by grouping the same profiles
        unique_profiles, unique_indices = np.unique(
            da_valid_profiles.values, axis=-1, return_inverse=True
        )

        da_profiles_indexes.loc[mask_valid] = unique_indices
        profiles_indexes = da_profiles_indexes.unstack()

        self.set_profiles(
            profiles=CompositeTemporalProfiles.from_ratios(
                unique_profiles.T, types=[MounthsProfile]
            ),
            indexes=profiles_indexes.drop_vars("lev"),
        )
