import urllib.request
from datetime import date, datetime
from os import PathLike
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr

from emiproc.grids import Grid, RegularGrid
from emiproc.inventories import Inventory
from emiproc.profiles.temporal.profiles import (
    DayOfYearProfile,
    Hour3OfDayPerMonth,
    MounthsProfile,
    get_leap_year_or_normal,
)
from emiproc.profiles.temporal.composite import CompositeTemporalProfiles
from emiproc.profiles.utils import ratios_dataarray_to_profiles


def download_gfed5(
    data_dir: PathLike,
    year: int,
    link_template: str = "https://surfdrive.surf.nl/files/index.php/s/VPMEYinPeHtWVxn/download?path=%2Fdaily&files=GFED5_Beta_daily_{year}{month}.nc",
):
    """Download the GFED5 files for a given year.

    The files are downloaded in the data_dir folder.

    :param data_dir: The directory where to download the files
    :param year: The year to download
    :param link_template: The template for the download link. The template should contain the year and month placeholders.
    """

    data_dir = Path(data_dir)

    for month in range(1, 13):
        link = link_template.format(year=year, month=f"{month:02d}")
        filename = link.split("=")[-1]
        filepath = data_dir / filename
        try:
            urllib.request.urlretrieve(link, filepath)
        except urllib.error.HTTPError as e:
            raise ValueError(
                f"Link {link} does not exist. Maybe the inventory is not available for {year=}."
            ) from e

    print(f"Downloaded gfed5 files for {year=}.")


class GFED_Grid(RegularGrid):
    """Grid for the GFED inventory.


    The hdf5 file contains all the information about the grid coordinates.
    It is a regular grid.


    Not also that the grid cell area is provided in the file,
    but we do not use it here.
    Also the values are not exactly the same (around 10% errors).
    Geopandas calculated areas are a bit larger it seems.

    .. note::

        The profiles implemented are only the day of year profile.
        If you want hour of day, you will have to implement it.

    """

    def __init__(self, gfed_filepath: PathLike):

        gfed_filepath = Path(gfed_filepath)
        ds = xr.open_dataset(gfed_filepath)

        # Get the lon lat coordinates
        # This assumes (but checks) that the grid is regular
        unique_lons = np.unique(ds["lon"].values, axis=0)
        unique_lats = np.unique(ds["lat"].values.T, axis=0)
        assert len(unique_lons) == 1
        assert len(unique_lats) == 1

        self.lon_range = unique_lons[0]
        self.lat_range = unique_lats[0]

        # Get the grid cell size (here also ensure that the grid is regular)
        unique_dx = np.unique(np.diff(self.lon_range))
        unique_dy = np.unique(np.diff(self.lat_range))

        assert len(unique_dx) == 1
        assert len(unique_dy) == 1

        self.dx = abs(unique_dx[0])
        self.dy = abs(unique_dy[0])

        self.nx = len(self.lon_range)
        self.ny = len(self.lat_range)

        # Bypass the RegularGrid __init__ method because we already have the grid coordinates
        Grid.__init__(self, gfed_filepath.stem)


class GFED4_Inventory(Inventory):
    """Global Fire Emissions Database.

    Global inventory based on satellite data, burned areas, fuel consumption.

    https://www.globalfiredata.org/

    You can download the input data for various year at
    https://www.geo.vu.nl/~gwerf/GFED/GFED4/

    This uses the area inside the GFED file to calculate the total emissions, but
    we found out that the area is not exactly the same as the one calculated by geopandas.
    See :py:class:`GFED_Grid` for the grid information.

    The data set contains two variables:
        * C: Carbon emissions
        * DM: Dry matter emissions

    .. note:: This inventory applies only for GFED4 .
        GFED5 has changed the format and is not supported by this class.
    """

    def __init__(self, gfed_filepath: PathLike, year: int):

        super().__init__()

        self.gfed_filepath = Path(gfed_filepath)
        gfed_file = self.gfed_filepath
        self.grid = GFED_Grid(gfed_filepath)

        self.year = year

        # Units of C var: g C / m^2 / month
        # Units of DM var: kg DM / m^2 / month

        das = []
        phony_dims = lambda ds: [dim for dim in ds.dims if dim.startswith("phony_dim")]
        lat_lon_dims = lambda ds: {phony_dims(ds)[0]: "lat", phony_dims(ds)[1]: "lon"}
        rename_phony_dims = lambda ds: ds.rename(lat_lon_dims(ds))
        for month in range(1, 13):
            da_dm = xr.open_dataset(gfed_file, group=f"/emissions/{month:02}")["DM"]
            ds_partion = xr.open_dataset(
                gfed_file, group=f"/emissions/{month:02}/partitioning"
            )
            # Get teh phony dims and renmae them
            ds_partion = rename_phony_dims(ds_partion)
            da_dm = rename_phony_dims(da_dm)
            da_partition = ds_partion.to_dataarray(dim="category")
            das.append((da_dm * da_partition).expand_dims(month=[month]))
        da = xr.concat(das, dim="month")

        # Rename the category to remove the `DM_` prefix
        da["category"] = [str(cat).split("_")[-1] for cat in da["category"].values]

        # Get the grid cell areas
        grid_areas = xr.open_dataset(gfed_file, group="/ancill/")["grid_cell_area"]
        grid_areas = rename_phony_dims(grid_areas)
        # Scale with the grid cell area to get kg / year / cell
        da = da * grid_areas

        da_stacked = da.stack(cell=("lon", "lat")).drop_vars(["lon", "lat"])
        cell_index = np.array(range(da_stacked.sizes["cell"]))
        da_stacked = da_stacked.assign_coords(cell=cell_index)

        da_total = da_stacked.sum(dim="month")

        mask_cell = da_total.sum(dim="category") > 0

        columns = {}

        for category in da_total["category"].values:
            columns[(category, "DM")] = da_total.sel(category=category).values

        self.gdf = gpd.GeoDataFrame(columns, geometry=self.grid.gdf.geometry)
        self.gdfs = {}

        # Now we make the profiles

        das_daily = []

        for month in range(1, 13):
            ds_daily = xr.open_dataset(
                gfed_file, group=f"/emissions/{month:02}/daily_fraction"
            )

            # Merge the days into a new dimension
            da_daily = xr.concat(
                [
                    ds_daily[f"day_{i:d}"].expand_dims(
                        day=[datetime(year=year, month=month, day=i)]
                    )
                    for i in range(1, 32)
                    if f"day_{i:d}" in ds_daily
                ],
                dim="day",
            )
            da_daily = rename_phony_dims(da_daily)
            das_daily.append(da_daily)

        da_day_of_year = xr.concat(das_daily, dim="day")
        da_day_of_year = da_day_of_year / da_day_of_year.sum(dim="day")

        # Stack to have on the cell dimension
        da_day_of_year_stacked = (
            da_day_of_year.stack(cell=("lon", "lat"))
            .drop_vars(["lon", "lat"])
            .assign_coords(cell=cell_index)
        )

        dss_diurnal = []
        for month in range(1, 13):
            ds_diurnal = xr.open_dataset(
                gfed_file, group=f"/emissions/{month:02}/diurnal_cycle"
            )
            dss_diurnal.append(rename_phony_dims(ds_diurnal).expand_dims(month=[month]))
        ds_diurnal = xr.concat(dss_diurnal, dim="month")
        da_diurnal = xr.concat(
            [
                ds_diurnal[hour_name]
                for hour_name in [
                    "UTC_0-3h",
                    "UTC_3-6h",
                    "UTC_6-9h",
                    "UTC_9-12h",
                    "UTC_12-15h",
                    "UTC_15-18h",
                    "UTC_18-21h",
                    "UTC_21-24h",
                ]
            ],
            dim="hour",
        )
        da_month_diurnal = xr.concat(
            [
                da_diurnal.sel(month=month).assign_coords(
                    hour=8 * (month - 1) + np.arange(8)
                )
                for month in range(1, 13)
            ],
            dim="hour",
        ).rename(hour="hour3_per_month")
        da_diurnal3_stacked = (
            da_month_diurnal.stack(cell=("lon", "lat"))
            .drop_vars(["lon", "lat"])
            .assign_coords(cell=cell_index)
        )

        # Below whe should use the function that generates composite profiles

        profiles_arrays = {
            Hour3OfDayPerMonth: da_diurnal3_stacked.rename(
                {"hour3_per_month": "ratio"}
            ).drop_vars("month"),
            get_leap_year_or_normal(
                DayOfYearProfile, year=year
            ): da_day_of_year_stacked.rename({"day": "ratio"}),
            MounthsProfile: da_stacked.rename({"month": "ratio"}),
        }

        parsed_profiles = {
            profile_type: (
                # Convert to scaling factors to ease nan handling
                da
                / da.mean(dim="ratio")
            )
            .fillna(1.0)
            .sel(cell=mask_cell)
            # Add the category dim
            .broadcast_like(da_stacked, exclude=["cell", "month"])
            # Drop the ratio coordinate values, so we can concat on the ratio dimension
            .drop_vars("ratio")
            for profile_type, da in profiles_arrays.items()
        }

        # Set the ratios all together and build the composite profiles
        profiles_ratios = xr.concat(parsed_profiles.values(), dim="ratio")
        ratios, indices = ratios_dataarray_to_profiles(profiles_ratios)

        profiles = CompositeTemporalProfiles.from_ratios(
            ratios, list(parsed_profiles.keys()), rescale=True
        )

        self.set_profiles(profiles, indices)


class GFED5(Inventory):
    """Global Fire Emissions Database.

    Global inventory based on satellite data, burned areas, fuel consumption.

    https://www.globalfiredata.org/

    You can download the input data for various year using :py:func:`download_gfed5`.
    """

    def __init__(self, file_dir: PathLike, year: int, substances: list[str]):
        super().__init__()

        files_dir = Path(file_dir)
        files = [
            files_dir / f"GFED5_Beta_daily_{year}{month:02d}.nc"
            for month in range(1, 13)
        ]

        # Check that all files exists
        for file in files:
            if not file.exists():
                raise ValueError(f"File {file} does not exist.")

        ds = xr.open_mfdataset(files, combine="by_coords")

        self.grid = RegularGrid.from_centers(
            x_centers=ds["lon"].values,
            y_centers=ds["lat"].values,
            name="GFED5",
        )

        profiles = {}

        for sub in substances:
            if sub not in ds.data_vars:
                raise ValueError(f"Substance {sub} not in the dataset.")

            profiles[sub] = ds[sub].expand_dims(substance=[sub])

        da_profiles: xr.DataArray = (
            xr.concat(list(profiles.values()), dim="substance")
            .stack(cell=("lon", "lat"))
            .drop_vars(["lon", "lat"])
            .rename({"time": "ratio"})
        )

        self.gdf = gpd.GeoDataFrame(
            {
                ("gfed", sub): da_profiles.sel(substance=sub).sum(dim=["ratio"]).values
                # Convert from kg/m2 to kg/cell
                * 1e-3 * self.grid.cell_areas
                for sub in substances
            },
            geometry=self.grid.gdf.geometry,
        )
        self.gdfs = {}

        ratios, indices = ratios_dataarray_to_profiles(da_profiles)

        profiles = CompositeTemporalProfiles.from_ratios(
            ratios, [get_leap_year_or_normal(DayOfYearProfile, year=year)], rescale=True
        )

        self.set_profiles(profiles, indices)
