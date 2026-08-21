from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from emiproc.inventories.corso import CorsoEmissions, interp_monthly_values
from emiproc.utilities import SEC_PER_YR


@pytest.fixture
def corso_directory(tmp_path: Path) -> Path:
    def write_profile(filename: str, values: np.ndarray) -> None:
        columns = [f"value_{index}" for index in range(values.size)] + ["unused"]
        table = pd.DataFrame(
            [np.append(values, np.nan)],
            index=pd.Index([1], name="ID"),
            columns=columns,
        )
        table.to_csv(tmp_path / filename, sep=";")

    write_profile("corso_ps_hourly_profiles_v2.0.csv", np.ones(24))
    write_profile("corso_ps_weekly_profiles_v2.0.csv", np.ones(7))
    write_profile("corso_ps_monthly_profiles_v2.0.csv", np.ones(12))
    write_profile("corso_ps_vertical_profiles_v2.0.csv", np.arange(16))

    one_kg_s_in_kty = SEC_PER_YR / 1_000_000
    catalogue = pd.DataFrame(
        {
            "plant_name": ["nearby", "nearby", "separate"],
            "longitude": [8.0000, 8.0010, 9.0],
            "latitude": [47.0000, 47.0010, 48.0],
            "sector": ["power", "power", "cement"],
            "ID_MonthFact": [1, 1, 1],
            "ID_WeekFact": [1, 1, 1],
            "ID_HourFact": [1, 1, 1],
            "ID_VertProf": [1, 1, 1],
            "co2_kty": one_kg_s_in_kty * np.array([1.0, 2.0, 4.0]),
        },
        index=pd.Index([1, 2, 3], name="ID"),
    )
    catalogue.to_csv(tmp_path / "corso_ps_catalogue_v2.0.csv", sep=";")
    return tmp_path


def test_corso_public_api_and_profiles(corso_directory: Path) -> None:
    corso = CorsoEmissions(corso_directory)
    requested_date = datetime(2021, 1, 4, 0)

    assert corso.get_dataset().sizes == {"ID": 3}
    assert corso.get_emission(1, "co2", requested_date) == pytest.approx(1.0)
    np.testing.assert_allclose(
        corso.get_emission_tser(2, "co2", [requested_date, requested_date]),
        [2.0, 2.0],
    )
    heights, profile = corso.get_profile(1)
    np.testing.assert_array_equal(heights, np.arange(0, 1501, 100))
    np.testing.assert_array_equal(profile, np.arange(16))
    assert interp_monthly_values(np.ones(12), requested_date) == 1.0


def test_corso_filter_preserves_open_bounds(corso_directory: Path) -> None:
    corso = CorsoEmissions(corso_directory)
    corso.filter_emissions(8.0, 8.01, 47.0, 47.01)

    assert corso.df_emiss.index.tolist() == [2]


def test_corso_netcdf_export_and_source_grouping(
    corso_directory: Path, tmp_path: Path
) -> None:
    corso = CorsoEmissions(corso_directory)
    dataset = corso.save_sources_to_netcdf(
        tmp_path / "corso.nc",
        sum_nearby_same_name=True,
        distance_threshold_m=1_000,
    )

    assert dataset.sizes["source"] == 2
    assert dataset["co2"].attrs["units"] == "kg s-1"
    np.testing.assert_allclose(np.sort(dataset["co2"]), [3.0, 4.0])
    assert (tmp_path / "corso.nc").is_file()
