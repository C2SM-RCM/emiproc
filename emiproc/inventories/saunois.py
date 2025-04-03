from pathlib import Path


import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.creation import polygons

from emiproc.grids import WGS84_PROJECTED, GeoPandasGrid
from emiproc.inventories import Inventory
from emiproc.profiles.temporal.profiles import MounthsProfile
from emiproc.profiles.temporal.composite import CompositeTemporalProfiles


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

        # Make the geometry
        lon = da_stacked_all["lon"].values
        lat = da_stacked_all["lat"].values

        lon_range = da["lon"].values
        lat_range = da["lat"].values

        d_lon = np.round(np.diff(lon_range)[0], 8)
        d_lat = np.round(np.diff(lat_range)[0], 8)
        # Ensure the spacing is constant
        assert np.allclose(
            np.diff(lon_range), d_lon
        ), "Longitude spacing is not constant"
        assert np.allclose(
            np.diff(lat_range), d_lat
        ), "Latitude spacing is not constant"

        # Reconstruct the grid vertices
        coords = np.array(
            [
                # Bottom left
                [
                    lon - d_lon / 2,
                    lat - d_lat / 2,
                ],
                # Bottom right
                [
                    lon + d_lon / 2,
                    lat - d_lat / 2,
                ],
                # Top right
                [
                    lon + d_lon / 2,
                    lat + d_lat / 2,
                ],
                # Top left
                [
                    lon - d_lon / 2,
                    lat + d_lat / 2,
                ],
            ]
        )

        coords = np.rollaxis(coords, -1, 0)

        geometry = gpd.GeoSeries(polygons(coords), crs="WGS84")
        self.grid = GeoPandasGrid(geometry, shape=(len(lon_range), len(lat_range)))

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
        converstion_factor = 1e-3 * geometry.to_crs(WGS84_PROJECTED).area
        da_total = da_stacked_total * converstion_factor.values

        # Convert to pandas
        df = (
            da_total.stack(catsub=("category", "substance"))
            .drop_vars(["cell"])
            .to_pandas()
        )
        self.gdf = gpd.GeoDataFrame(df, geometry=geometry)
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
