import calendar
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
import pandas as pd

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

        self.year = int(pd.to_datetime(da_stacked["time"].values[0]).year)

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


class Saunois(Inventory):
    """Inventory based on the Global Carbon Project methane budget prior fluxes.

    Unlike :py:class:`SaunoisInventory`, which reads one file per category,
    this reads a single merged file (e.g. ``GCP_Prior_CH4_fluxes.nc``) where
    each category is a ``flux_ch4_<category>`` data variable in units of
    kg CH4/m2/s.

    https://doi.org/10.5194/essd-12-1561-2020
    https://www.icos-cp.eu/GCP-CH4-2024

    Most categories have a monthly time series covering several years
    (dim ``time``); the requested ``year`` is sliced out of those. A few
    categories (e.g. ``freshwaters``, ``geological``, ``ocean``, ``termites``,
    ``wetlands``) are only available as a fixed 12-month climatology
    (dim ``time_climato``), which is reused as-is regardless of the
    requested year.
    """

    def __init__(self, saunois_file: Path, year: int):
        """Initialize the inventory.

        Parameters
        ----------

        saunois_file :
            Path to the merged Saunois/GCP-CH4 netcdf file.
        year :
            The year to extract from the categories with a yearly time series.
        """
        super().__init__()

        saunois_file = Path(saunois_file)
        ds = xr.open_dataset(saunois_file)

        self.year = year

        categories = sorted(
            var[len("flux_ch4_") :]
            for var in ds.data_vars
            if var.startswith("flux_ch4_")
        )

        days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        if calendar.isleap(year):
            days_in_month[1] = 29

        das = []
        for cat in categories:
            da = ds[f"flux_ch4_{cat}"]
            units = da.attrs.get("units")
            if units != "kg/m2/s":
                raise ValueError(
                    f"Variable 'flux_ch4_{cat}' has units {units!r}, but this "
                    "class only supports 'kg/m2/s' (the unit conversion "
                    "assumes it). See emiproc.utils.units for other supported "
                    "units and adapt the conversion if needed."
                )
            if "time" in da.dims:
                da = da.sel(time=da["time"].dt.year == year)
                if da.sizes["time"] != 12:
                    raise ValueError(
                        f"Expected 12 months of data for category {cat!r} and"
                        f" {year=}, got {da.sizes['time']}. Is the year within"
                        " the file's time range?"
                    )
                da = da.rename(time="month")
            else:
                # No year-specific data available: reuse the climatology.
                da = da.rename(time_climato="month")
            da = da.assign_coords(month=np.arange(1, 13))
            das.append(da.expand_dims(category=[cat]))

        da = xr.concat(das, dim="category")

        # Add the substance dimension
        da = da.expand_dims(substance=["CH4"])

        # Set the coords to be str
        da["substance"] = da["substance"].astype(str)
        da["category"] = da["category"].astype(str)

        # replace the lat lon by cell
        da_stacked_all = da.stack(cell=("lon", "lat"))
        da_stacked = da_stacked_all.drop_vars(
            [
                c
                for c in ["lat", "lon", "lat_bnds", "lon_bnds"]
                if c in da_stacked_all.coords
            ]
        )
        # Use a simple integer index for the cell
        da_stacked["cell"] = np.arange(da_stacked.sizes["cell"])

        self.grid = RegularGrid.from_centers(
            x_centers=da["lon"].values,
            y_centers=da["lat"].values,
            name="Saunois_Grid",
            rounding=2,
        )

        # Unit conversion
        # Units are kg CH4 / m2 / s -> kg / year / cell
        # Multiply each month by the number of seconds in that month and
        # sum the months, then scale by the grid cell area.
        seconds_in_month = days_in_month * 86400
        da_stacked_total = (
            da_stacked
            * xr.DataArray(
                seconds_in_month,
                dims="month",
                coords={"month": da_stacked.coords["month"]},
            )
        ).sum(dim="month")

        da_total = da_stacked_total * xr.DataArray(
            self.grid.cell_areas, dims="cell", coords={"cell": da_stacked["cell"]}
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
        da_ratios = (da_profiles / da_profiles.sum(dim="month")).fillna(0.0)
        mask_valid = da_ratios.sum(dim="month") != 0

        # Create the profiles indexes
        da_profiles_indexes = da_ratios.sum(dim="month")
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
            indexes=profiles_indexes,
        )
