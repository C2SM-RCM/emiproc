from pathlib import Path


import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.creation import polygons

from emiproc.grids import WGS84_PROJECTED, GeoPandasGrid
from emiproc.inventories import Inventory
from emiproc.profiles.temporal.profiles import DayOfYearProfile
from emiproc.profiles.temporal.composite import CompositeTemporalProfiles


class LPJ_GUESS_Inventory(Inventory):
    """LPJ-GUESS inventory.

    This is an implementation to read netcdf outputs from LPJ-GUESS.

    https://web.nateko.lu.se/lpj-guess/

    """

    def __init__(self, lpj_guess_files: list[Path]):
        """Initialize the inventory.

        Parameters
        ----------
        lpj_guess_files:
            List of paths to the LPJ-GUESS files.
        """
        super().__init__()

        ds = xr.open_mfdataset(lpj_guess_files, combine="by_coords")

        varnames = [
            k
            for k in ds.variables.mapping.keys()
            if k not in ["longitude", "latitude", "time"]
        ]

        varname_to_catsub = {
            "_".join(splitted): ("_".join(splitted[1:]), splitted[0])
            for splitted in map(lambda x: x.split("_"), varnames)
        }

        # Check that the units are all as expceted
        expected = "mg CH4 m-2 d-1"
        for varname in varnames:
            assert (
                ds[varname].units == expected
            ), f"{varname} has units {ds[varname].units} instead of {expected}"

        # Resample the dataset as a dataarray with the categories and substances added as dimensions
        da = ds.to_dataarray(dim="variable")
        # Transform the variable dimension to a multiindex
        da_merged = xr.combine_by_coords(
            [
                da.sel(variable=varname)
                .drop_vars("variable")
                .expand_dims(dim=dict(category=[cat], substance=[sub]))
                for varname, (cat, sub) in varname_to_catsub.items()
            ],
        )
        da_merged["substance"] = da_merged["substance"].astype(str)
        da_merged["category"] = da_merged["category"].astype(str)

        # replace the lat lon by cell
        da_stacked_all = da_merged.stack(cell=("longitude", "latitude"))

        da_stacked = da_stacked_all.drop_vars(["latitude", "longitude"])
        # Use a simple integer index for the cell
        da_stacked["cell"] = np.arange(da_stacked.sizes["cell"])

        # Make the geometry
        lon = da_stacked_all["longitude"].values
        lat = da_stacked_all["latitude"].values

        lon_range = da_merged["longitude"].values
        lat_range = da_merged["latitude"].values

        d_lon = np.diff(lon_range)[0]
        d_lat = np.diff(lat_range)[0]
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

        # Create the polygons
        geometry = gpd.GeoSeries(polygons(coords), crs="WGS84")
        self.grid = GeoPandasGrid(geometry, shape=(len(lon_range), len(lat_range)))

        # Unit conversion
        # "mg CH4 m-2 d-1" -> "kg / year / cell"
        # Day to Year is already included when we summed
        # kg/mg * m2/cell
        converstion_factor = 1e-6 * geometry.to_crs(WGS84_PROJECTED).area
        da_total = da_stacked.sum(dim="time") * converstion_factor.values

        # Convert to pandas
        df = (
            da_total.stack(catsub=("category", "substance"))
            .drop_vars(["cell"])
            .to_pandas()
        )
        self.gdf = gpd.GeoDataFrame(df, geometry=geometry)
        self.gdfs = {}

        # Gerenerate the profiles
        da_profiles = da_stacked.stack(profiles=("substance", "category", "cell"))
        # Convert to ratios
        da_ratios = (da_profiles / da_profiles.sum(dim="time")).fillna(0.0)
        mask_valid = da_ratios.sum(dim="time") != 0

        # Create the profiles indexes
        da_profiles_indexes = da_ratios.sum(dim="time")
        da_profiles_indexes.values = -np.ones(da_profiles_indexes.shape, dtype=int)

        # Set the values to linear indices
        da_valid_profiles = da_ratios.sel(profiles=mask_valid)
        da_profiles_indexes.loc[mask_valid] = np.arange(
            da_valid_profiles.sizes["profiles"]
        )
        profiles_indexes = da_profiles_indexes.unstack()

        self.set_profiles(
            profiles=CompositeTemporalProfiles.from_ratios(
                da_valid_profiles.values.T, types=[DayOfYearProfile]
            ),
            indexes=profiles_indexes,
        )
