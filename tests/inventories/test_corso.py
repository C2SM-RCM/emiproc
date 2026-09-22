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


def test_corso_inventory_is_a_regular_inventory(corso_directory: Path) -> None:
    """The CORSO database is usable like any other emiproc inventory."""
    from emiproc.inventories import Inventory
    from emiproc.inventories.corso import CorsoInventory
    from emiproc.inventories.utils import get_total_emissions

    inv = CorsoInventory(corso_directory)

    assert isinstance(inv, Inventory)
    assert sorted(inv.categories) == ["cement", "power"]
    assert inv.substances == ["CO2"]
    # Point sources live in `gdfs`, not on a grid.
    assert inv.gdf is None
    assert all(geom.geom_type == "Point" for geom in inv.gdfs["power"].geometry)

    # The catalogue holds 1 + 2 + 4 kg/s, converted to kg/y.
    totals = get_total_emissions(inv)
    assert totals["CO2"]["__total__"] == pytest.approx(7.0 * SEC_PER_YR)
    assert totals["CO2"]["power"] == pytest.approx(3.0 * SEC_PER_YR)
    assert totals["CO2"]["cement"] == pytest.approx(4.0 * SEC_PER_YR)


def test_corso_inventory_sets_profiles(corso_directory: Path) -> None:
    from emiproc.inventories.corso import CorsoInventory
    from emiproc.profiles.temporal.profiles import (
        DailyProfile,
        MounthsProfile,
        WeeklyProfile,
    )

    inv = CorsoInventory(corso_directory)

    assert inv.t_profiles_groups is not None
    assert set(inv.t_profiles_groups.types) == {
        DailyProfile,
        WeeklyProfile,
        MounthsProfile,
    }
    assert inv.t_profiles_indexes.dims == ("category", "substance")

    assert inv.v_profiles is not None
    # Ratios are normalised and the heights are the tops of the layers.
    np.testing.assert_allclose(inv.v_profiles.ratios.sum(axis=1), 1.0)
    assert inv.v_profiles.height[0] == pytest.approx(100.0)
    assert len(inv.v_profiles.height) == inv.v_profiles.ratios.shape[1]


def test_corso_inventory_mapping_and_filtering(corso_directory: Path) -> None:
    from emiproc.inventories.corso import CorsoInventory
    from emiproc.inventories.utils import get_total_emissions

    mapped = CorsoInventory(
        corso_directory,
        categories_mapping={"power": "ene", "cement": "ind"},
        substances=["CO2"],
    )
    assert sorted(mapped.categories) == ["ene", "ind"]

    # Only the two sources around (8.0, 47.0) are inside the box.
    cropped = CorsoInventory(corso_directory, bbox=(7.5, 8.5, 46.5, 47.5))
    assert get_total_emissions(cropped)["CO2"]["__total__"] == pytest.approx(
        3.0 * SEC_PER_YR
    )

    with pytest.raises(ValueError, match="not in the CORSO catalogue"):
        CorsoInventory(corso_directory, substances=["CH4"])


def test_corso_inventory_can_be_subtracted_from_a_gridded_inventory(
    corso_directory: Path,
) -> None:
    """The end to end double counting workflow."""
    import geopandas as gpd

    from emiproc.grids import RegularGrid, WGS84
    from emiproc.inventories import Inventory
    from emiproc.inventories.corso import CorsoInventory
    from emiproc.inventories.utils import (
        get_total_emissions,
        merge_inventories_without_double_counting,
    )

    points = CorsoInventory(corso_directory, bbox=(7.5, 8.5, 46.5, 47.5))

    grid = RegularGrid(
        xmin=7.5, ymin=46.5, xmax=8.5, ymax=47.5, nx=2, ny=2, name="coarse", crs=WGS84
    )
    coarse = Inventory.from_gdf(
        gpd.GeoDataFrame(
            {("power", "CO2"): np.full(4, 10.0 * SEC_PER_YR)},
            geometry=grid.gdf.geometry,
            crs=WGS84,
        ),
        name="coarse",
    )
    coarse.grid = grid

    before = get_total_emissions(coarse)["CO2"]["__total__"]
    merged = merge_inventories_without_double_counting(coarse, points)

    # Total unchanged, but the point sources are now resolved individually.
    assert get_total_emissions(merged)["CO2"]["__total__"] == pytest.approx(before)
    assert len(merged.gdfs["power"]) == 2
    # The cell holding the sources lost exactly their emissions.
    assert merged.gdf[("power", "CO2")].sum() == pytest.approx(
        before - 3.0 * SEC_PER_YR
    )
