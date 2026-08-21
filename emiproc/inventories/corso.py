"""Read and process the CORSO point-source emissions database."""

from collections.abc import Hashable, Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Optional, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

DateValue: TypeAlias = str | date | datetime | np.datetime64 | pd.Timestamp
DateValues: TypeAlias = Iterable[DateValue]


def interp_monthly_values(
    monthly_values: npt.ArrayLike,
    requested_date: datetime | pd.Timestamp,
) -> float:
    """Interpolate monthly profile values to a requested date.

    Monthly values are treated as valid at approximately the middle of each
    month and are extended cyclically across the year boundary.

    Args:
        monthly_values: Values for January through December, in that order.
        requested_date: Date and time at which to interpolate the profile.

    Returns:
        The linearly interpolated profile value.
    """
    # Approximate center day-of-year for each month
    month_centers = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])

    # Make values cyclic.
    days = np.concatenate(
        ([month_centers[-1] - 365], month_centers, [month_centers[0] + 365])
    )
    monthly_array = np.asarray(monthly_values)
    values = np.concatenate(([monthly_array[-1]], monthly_array, [monthly_array[0]]))

    # Compute fractional day of year.
    day_of_year = requested_date.timetuple().tm_yday + (
        requested_date.hour / 24.0
        + requested_date.minute / 1440.0
        + requested_date.second / 86400.0
    )

    # Perform linear interpolation
    return float(np.interp(day_of_year, days, values))


class CorsoEmissions:
    """Generate MicroHH emissions from the CORSO database.

    Args:
        csv_path: Directory containing the CORSO point-source CSV files.
    """

    def __init__(self, csv_path: str | Path) -> None:
        csv_directory = Path(csv_path)

        def read_csv(csv_file: str, drop_last_col: bool = False) -> pd.DataFrame:
            """Read one of the semicolon-delimited CORSO files.

            Args:
                csv_file: Name of the file within ``csv_path``.
                drop_last_col: Whether to remove the final column.

            Returns:
                The parsed file contents.
            """
            # Parse CSV with Pandas.
            df = pd.read_csv(
                csv_directory / csv_file,
                sep=";",
                header=0,
                index_col=0,
            )
            return df.iloc[:, :-1] if drop_last_col else df

        # Read hour/week/month and vertical profiles.
        self.df_hour = read_csv("corso_ps_hourly_profiles_v2.0.csv", True)
        self.df_week = read_csv("corso_ps_weekly_profiles_v2.0.csv", True)
        self.df_month = read_csv("corso_ps_monthly_profiles_v2.0.csv", True)
        self.df_vert = read_csv("corso_ps_vertical_profiles_v2.0.csv", True)

        # Read emissions.
        self.df_emiss = read_csv("corso_ps_catalogue_v2.0.csv")

    def filter_emissions(
        self,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        min_emission: Optional[float] = None,
    ) -> None:
        """Filter the emission catalogue to an open longitude/latitude box.

        Args:
            lon_min: Exclusive lower longitude bound.
            lon_max: Exclusive upper longitude bound.
            lat_min: Exclusive lower latitude bound.
            lat_max: Exclusive upper latitude bound.
            min_emission: Minimal emission required to be imported in kt per year.
        """
        if min_emission is not None:
            emissions = self.df_emiss["co2_kty"]
            plant_emissions = emissions.groupby(self.df_emiss["plant_name"]).transform(
                "sum"
            )
            emission_mask = emissions.ge(min_emission) | plant_emissions.ge(
                min_emission
            )
        lat = self.df_emiss.latitude
        lon = self.df_emiss.longitude

        mask = (lon > lon_min) & (lon < lon_max) & (lat > lat_min) & (lat < lat_max)
        if min_emission is not None:
            mask &= emission_mask

        self.df_emiss = self.df_emiss[mask]

    def get_dataset(self) -> xr.Dataset:
        """This methods returns the raw Corso emission database as dataset.

        Return:
            Dataset with corso emissions
        """
        # return xr.Dataset.from_dataframe(self.df_emiss)
        return self.df_emiss.to_xarray()

    def get_emission_tser(
        self,
        emission_id: Hashable,
        specie: str,
        dates: DateValues,
        interpolate_month: bool = True,
    ) -> npt.NDArray[np.float64]:
        """Calculate an emission time series.

        Args:
            emission_id: Catalogue index identifying the point source.
            specie: Species prefix used by the catalogue's ``*_kty`` column.
            dates: Dates and times for which to calculate emissions.
            interpolate_month: Whether to interpolate monthly profile values.

        Returns:
            Emission rates in kilograms per second, one per requested date.
        """
        # Cast dates to `datetime` format.
        converted_dates = pd.to_datetime(list(dates))

        emission = np.zeros(len(converted_dates), dtype=np.float64)
        for i, requested_date in enumerate(converted_dates):
            emission[i] = self.get_emission(
                emission_id,
                specie,
                requested_date,
                interpolate_month,
            )

        return emission

    def get_emission(
        self,
        emission_id: Hashable,
        specie: str,
        requested_date: datetime | pd.Timestamp,
        interpolate_month: bool = True,
    ) -> float:
        """Calculate a source's emission rate at a requested date and time.

        Args:
            emission_id: Catalogue index identifying the point source.
            specie: Species prefix used by the catalogue's ``*_kty`` column.
            requested_date: Full-hour date and time for the calculation.
            interpolate_month: Whether to interpolate monthly profile values.

        Returns:
            The emission rate in kilograms per second.

        Raises:
            ValueError: If ``requested_date`` is not exactly on a full hour.
        """
        # Only full hours are supported for now to avoid interpolations.
        if not (
            requested_date.minute == 0
            and requested_date.second == 0
            and requested_date.microsecond == 0
        ):
            raise ValueError("Only full hours are supported for now")

        # Month, weekday, and hour as integers.
        # 0 = Januari and 0 = Monday, 0 = 00:00 UTC.
        index_month = requested_date.month - 1
        index_day = requested_date.weekday()
        index_hour = requested_date.hour

        # Filter Dataframe on emission ID.
        dfs = self.df_emiss.loc[emission_id]

        # Get time profile IDs.
        id_month = dfs["ID_MonthFact"]
        id_week = dfs["ID_WeekFact"]
        id_hour = dfs["ID_HourFact"]

        # Get time profiles as Numpy arrays with floats.
        def get_prof(
            df: pd.DataFrame,
            profile_id: Hashable,
        ) -> npt.NDArray[np.float64]:
            return df[df.index == profile_id].values.astype(float).flatten()

        prof_month = get_prof(self.df_month, id_month)
        prof_week = get_prof(self.df_week, id_week)
        prof_hour = get_prof(self.df_hour, id_hour)

        # Total emission in kg per year.
        emiss_y = dfs[f"{specie}_kty"] * 10**6

        # Emission in kg / month.
        if interpolate_month:
            month_scale_fac = interp_monthly_values(prof_month, requested_date)
            emiss_m = emiss_y * month_scale_fac / 12.0
        else:
            emiss_m = emiss_y * prof_month[index_month] / 12.0

        # Emission in kg / day.
        days_per_month = 365.25 / 12
        emiss_d = emiss_m / days_per_month * prof_week[index_day]

        # Emission in kg / hour.
        emiss_h = emiss_d * prof_hour[index_hour] / 24.0

        # Check: OKAY!
        # print(emiss_h, emiss_y / 365.25 / 24)

        # Return emission in kg / second.
        return float(emiss_h / 3600)

    def get_profile(
        self,
        emission_id: Hashable,
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float64]]:
        """Get the vertical emission profile for a catalogue source.

        Args:
            emission_id: Catalogue index identifying the point source.

        Returns:
            Emission heights and corresponding vertical profile values.
        """
        # Filter Dataframe on emission ID.
        dfs = self.df_emiss.loc[emission_id]

        # Get time profile IDs.
        id_prof = dfs["ID_VertProf"]

        # Get time profiles as Numpy arrays with floats.
        def get_prof(
            df: pd.DataFrame,
            profile_id: Hashable,
        ) -> npt.NDArray[np.float64]:
            return df[df.index == profile_id].values.astype(float).flatten()

        # Emission heights.
        z = np.arange(0, 1501, 100)

        return z, get_prof(self.df_vert, id_prof)

    def save_sources_to_netcdf(
        self,
        output_path: str | Path,
        sum_nearby_same_name: bool = False,
        distance_threshold_m: float = 1000.0,
    ) -> xr.Dataset:
        """Save the CORSO point-source catalogue as a NetCDF file.

        Emission columns ending in ``_kty`` are converted from kilotonnes per
        year to kilograms per second. If requested, sources with the same plant
        name are grouped when all sources in a group are less than the
        specified distance apart. Their emissions are summed and their
        coordinates are averaged.

        Args:
            output_path: Path of the NetCDF file to create.
            sum_nearby_same_name: Combine nearby sources sharing a plant name.
            distance_threshold_m: Maximum separation within a combined group.

        Returns:
            The dataset written to ``output_path``.
        """
        if distance_threshold_m <= 0:
            raise ValueError("distance_threshold_m must be greater than zero")

        sources = self.df_emiss.reset_index().copy()
        id_column = sources.columns[0]
        if id_column == "ID":
            sources[id_column] = sources[id_column].astype(str).str.zfill(5)
        emission_columns = [
            column for column in sources.columns if column.lower().endswith("_kty")
        ]
        required_columns = {"plant_name", "longitude", "latitude"}
        missing_columns = required_columns.difference(sources.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"CORSO catalogue is missing required columns: {missing}")
        if not emission_columns:
            raise ValueError("CORSO catalogue contains no '*_kty' emission columns")

        if sum_nearby_same_name:
            sources = self._sum_nearby_sources(
                sources,
                emission_columns,
                distance_threshold_m,
            )

        seconds_per_year = 365.25 * 24 * 3600
        kilotonnes_to_kg = 1_000_000
        conversion_factor = kilotonnes_to_kg / seconds_per_year

        data_vars = {}
        for column in sources.columns:
            if column in emission_columns:
                species = column[:-4]
                values = sources[column].astype(float).to_numpy() * conversion_factor
                data_vars[species] = ("source", values)
            else:
                values = sources[column].to_numpy()
                if values.dtype == object:
                    values = sources[column].fillna("").astype(str).to_numpy()
                data_vars[column] = ("source", values)

        dataset = xr.Dataset(
            data_vars=data_vars,
            coords={"source": np.arange(len(sources), dtype=np.int64)},
            attrs={
                "title": "CORSO point-source emissions catalogue",
                "emission_conversion": (
                    "kilotonnes per year converted using 365.25 days per year"
                ),
                "nearby_same_name_sources_summed": str(sum_nearby_same_name).lower(),
                "nearby_source_distance_threshold_m": distance_threshold_m,
            },
        )
        dataset[id_column].attrs["long_name"] = "CORSO point-source identifier"
        dataset["longitude"].attrs["units"] = "degrees_east"
        dataset["latitude"].attrs["units"] = "degrees_north"
        for column in emission_columns:
            species = column[:-4]
            dataset[species].attrs.update(
                {
                    "units": "kg s-1",
                    "long_name": f"{species.upper()} emission rate",
                    "original_units": "kt yr-1",
                }
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_netcdf(output_path)
        return dataset

    @staticmethod
    def _sum_nearby_sources(
        sources: pd.DataFrame,
        emission_columns: list[str],
        distance_threshold_m: float,
    ) -> pd.DataFrame:
        """Combine close, same-name sources using complete-linkage clustering."""
        grouped_rows = []
        named_sources = sources.loc[sources["plant_name"].notna(), :]
        plant_names = named_sources["plant_name"].drop_duplicates()
        for plant_name in plant_names:
            same_name = named_sources.loc[
                named_sources["plant_name"] == plant_name, :
            ].reset_index(drop=True)
            if len(same_name) == 1:
                grouped_rows.append(same_name.iloc[0].to_dict())
                continue

            distances = CorsoEmissions._pairwise_haversine_distance(same_name)
            clusters = fcluster(
                linkage(squareform(distances), method="complete"),
                t=distance_threshold_m,
                criterion="distance",
            )
            for cluster_id in np.unique(clusters):
                cluster = same_name.loc[clusters == cluster_id]
                row = cluster.iloc[0].to_dict()
                longitude_radians = np.radians(
                    cluster["longitude"].astype(float).to_numpy()
                )
                row["longitude"] = np.degrees(
                    np.arctan2(
                        np.sin(longitude_radians).mean(),
                        np.cos(longitude_radians).mean(),
                    )
                )
                row["latitude"] = cluster["latitude"].astype(float).mean()
                for column in emission_columns:
                    row[column] = cluster[column].astype(float).sum()
                if len(cluster) > 1:
                    id_column = sources.columns[0]
                    row[id_column] = ",".join(cluster[id_column].astype(str))
                grouped_rows.append(row)

        unnamed_sources = sources.loc[sources["plant_name"].isna(), :]
        for _, source in unnamed_sources.iterrows():
            grouped_rows.append(source.to_dict())

        return pd.DataFrame(grouped_rows, columns=sources.columns)

    @staticmethod
    def _pairwise_haversine_distance(sources: pd.DataFrame) -> np.ndarray:
        """Return pairwise great-circle distances in metres."""
        longitude = np.radians(sources["longitude"].astype(float).to_numpy())
        latitude = np.radians(sources["latitude"].astype(float).to_numpy())
        delta_lon = longitude[:, None] - longitude[None, :]
        delta_lat = latitude[:, None] - latitude[None, :]
        haversine = (
            np.sin(delta_lat / 2) ** 2
            + np.cos(latitude[:, None])
            * np.cos(latitude[None, :])
            * np.sin(delta_lon / 2) ** 2
        )
        return 2 * 6_371_000 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))


if __name__ == "__main__":
    """
    Just for testing..
    """
    import matplotlib.pyplot as plt

    plt.close("all")

    e = CorsoEmissions("/home/scratch1/bart/emissions/corso")
    e.filter_emissions(4, 5, 50, 51)

    row = e.df_emiss.iloc[4]
    name = row.name

    dates = pd.date_range("2020-01-01T00:00", "2020-04-01T00:00", freq="1h")

    em1 = e.get_emission_tser(name, "co2", dates, interpolate_month=True)
    em2 = e.get_emission_tser(name, "co2", dates, interpolate_month=False)

    plt.figure(figsize=(6, 5), layout="constrained")

    plt.subplot(211)
    plt.title("Linear interpolation monthly values", loc="left")
    plt.plot(dates, em1)

    plt.subplot(212)
    plt.title("NN interpolation monthly values", loc="left")
    plt.plot(dates, em2)


__all__ = ["CorsoEmissions", "interp_monthly_values"]
