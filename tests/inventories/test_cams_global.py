from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import geopandas as gpd

from emiproc.exports.utils import get_temporally_scaled_array
from emiproc.grids import WGS84
from emiproc.inventories import Inventory
from emiproc.inventories.cams_global import CAMSGlobal
from emiproc.inventories.utils import subtract_inventories
from emiproc.utilities import SEC_PER_YR


@pytest.fixture
def cams_file(tmp_path: Path) -> Path:
    path = tmp_path / "CAMS-GLOB-ANT_v6.2_methane_2025.nc"
    shape = (12, 2, 2)
    month_factor = np.arange(1, 13, dtype=np.float32)[:, None, None]
    ds = xr.Dataset(
        data_vars={
            "ene": (
                ("time", "lat", "lon"),
                np.broadcast_to(month_factor, shape).copy(),
                {
                    "units": "kg m-2 s-1",
                    "sector_name": "Power generation",
                    "long_name": "Emissions of CH4 for ene sector",
                },
            ),
            "res": (
                ("time", "lat", "lon"),
                np.broadcast_to(2 * month_factor, shape).copy(),
                {
                    "units": "kg m-2 s-1",
                    "sector_name": "Residential buildings",
                    "long_name": "Emissions of CH4 for res sector",
                },
            ),
            "sum": (
                ("time", "lat", "lon"),
                np.broadcast_to(3 * month_factor, shape).copy(),
                {
                    "units": "kg m-2 s-1",
                    "sector_name": "Sum of sectors",
                    "long_name": "Emissions of CH4 for sum of sectors",
                },
            ),
        },
        coords={
            "time": ("time", np.arange(12), {"units": "months since 2025-01-01"}),
            "lat": [-0.05, 0.05],
            "lon": [-0.05, 0.05],
        },
    )
    ds.to_netcdf(path)
    return path


def test_import_cams_global(cams_file: Path) -> None:
    inventory = CAMSGlobal(cams_file)

    assert inventory.year == 2025
    assert inventory.substance_name == "CH4"
    assert inventory.monthly_flux.dims == ("time", "category", "cell")
    assert inventory.monthly_flux.shape == (12, 2, 4)
    assert list(inventory.monthly_flux.category.values) == ["ene", "res"]
    assert pd.Timestamp(inventory.monthly_flux.time.values[0]).month == 1
    assert np.all(inventory.monthly_flux.sel(category="ene").isel(time=0) == 1)


def test_previous_inventory_name_remains_compatible() -> None:
    from emiproc.inventories.cams_glob_ant import CAMSGlobAnt

    assert CAMSGlobAnt is CAMSGlobal


def test_import_accepts_float32_coordinate_roundoff(tmp_path: Path) -> None:
    ds = xr.Dataset(
        data_vars={
            "sum": (
                ("time", "lat", "lon"),
                np.ones((12, 4, 2), dtype=np.float32),
                {
                    "units": "kg m-2 s-1",
                    "long_name": "Emissions of CH4 for sum of sectors",
                },
            )
        },
        coords={
            "time": ("time", np.arange(12), {"units": "months since 2025-01-01"}),
            "lon": np.array([8.35, 8.45], dtype=np.float32),
            "lat": np.array([47.15, 47.25, 47.35, 47.45], dtype=np.float32),
        },
    )
    # These are nominal 0.1 degree increments, but their float32 differences
    # are not necessarily equal within RegularGrid's default tolerance.
    path = tmp_path / "CAMS-GLOB-ANT_v6.2_methane_2025_float32.nc"
    ds.to_netcdf(path)

    inventory = CAMSGlobal(path, variables=["sum"])

    assert inventory.grid.dx == pytest.approx(0.1)
    assert inventory.grid.dy == pytest.approx(0.1)


def _point_inventory(
    longitudes: list[float],
    latitudes: list[float],
    sectors: list[str],
    emissions_kg_s: list[float],
    substance: str = "CH4",
) -> Inventory:
    """Build a point source inventory in kg/y, as CorsoInventory would."""
    inv = Inventory()
    inv.name = "points"
    inv.year = 2025
    inv.gdf = None
    inv.gdfs = {}
    frame = pd.DataFrame(
        {
            "sector": sectors,
            substance: np.asarray(emissions_kg_s, dtype=float) * SEC_PER_YR,
            "longitude": longitudes,
            "latitude": latitudes,
        }
    )
    for sector, group in frame.groupby("sector"):
        inv.gdfs[str(sector)] = gpd.GeoDataFrame(
            {substance: group[substance].to_numpy()},
            geometry=gpd.points_from_xy(group["longitude"], group["latitude"]),
            crs=WGS84,
        )
    return inv


def test_subtract_point_sources_maps_point_to_cell(cams_file: Path) -> None:
    """The generic operator removes the point emissions from the right cells."""
    original = CAMSGlobal(cams_file)
    points = _point_inventory(
        longitudes=[-0.04, -0.04, 1.0],
        latitudes=[-0.04, 0.04, 0.0],
        sectors=["power", "cement", "power"],
        emissions_kg_s=[1.0, 2.0, 4.0],
    )

    adjusted = subtract_inventories(
        original,
        points,
        category_mapping={"power": "ene", "cement": "res"},
        negative_policy="allow",
    )

    # Only the cells holding a source change, by exactly that source's rate.
    ene_removed = (
        original.gdf[("ene", "CH4")] - adjusted.gdf[("ene", "CH4")]
    ).to_numpy()
    res_removed = (
        original.gdf[("res", "CH4")] - adjusted.gdf[("res", "CH4")]
    ).to_numpy()
    expected_ene = np.zeros(4)
    expected_res = np.zeros(4)
    expected_ene[0] = 1.0 * SEC_PER_YR
    expected_res[1] = 2.0 * SEC_PER_YR
    np.testing.assert_allclose(ene_removed, expected_ene, rtol=1e-9)
    np.testing.assert_allclose(res_removed, expected_res, rtol=1e-9)

    # The third source is outside the CAMS domain and is therefore dropped.
    summary = adjusted.subtraction_summary
    assert summary["removed_kg_y"]["ene/CH4"] == pytest.approx(1.0 * SEC_PER_YR)
    assert summary["clipped_values"] == 0


def test_subtract_all_point_sectors_from_cams_sum(cams_file: Path) -> None:
    original = CAMSGlobal(cams_file, variables=["sum"])
    points = _point_inventory(
        longitudes=[-0.04, -0.04],
        latitudes=[-0.04, 0.04],
        sectors=["power", "cement"],
        emissions_kg_s=[1.0, 2.0],
    )

    adjusted = subtract_inventories(
        original,
        points,
        category_mapping={"power": "sum", "cement": "sum"},
        negative_policy="allow",
    )

    removed = (original.gdf[("sum", "CH4")] - adjusted.gdf[("sum", "CH4")]).to_numpy()
    expected = np.zeros(4)
    expected[0] = 1.0 * SEC_PER_YR
    expected[1] = 2.0 * SEC_PER_YR
    np.testing.assert_allclose(removed, expected, rtol=1e-9)


def test_negative_policy_clip_and_raise(cams_file: Path) -> None:
    original = CAMSGlobal(cams_file)
    excessive = _point_inventory(
        longitudes=[-0.04],
        latitudes=[-0.04],
        sectors=["power"],
        emissions_kg_s=[1e20],
    )

    with pytest.warns(RuntimeWarning, match="clipped to zero"):
        clipped = subtract_inventories(
            original, excessive, category_mapping={"power": "ene"}
        )
    assert clipped.gdf[("ene", "CH4")].iloc[0] == 0
    assert clipped.subtraction_summary["clipped_values"] == 1
    # The summary reports that far less was removed than was requested.
    assert (
        clipped.subtraction_summary["removed_kg_y"]["ene/CH4"]
        < clipped.subtraction_summary["requested_kg_y"]["ene/CH4"]
    )

    with pytest.raises(ValueError, match="negative values"):
        subtract_inventories(
            original,
            excessive,
            category_mapping={"power": "ene"},
            negative_policy="raise",
        )


def test_profiles_reproduce_the_monthly_fluxes(cams_file: Path) -> None:
    """gdf + temporal profiles must carry the same information as the input."""
    inventory = CAMSGlobal(cams_file)
    areas = np.asarray(inventory.grid.cell_areas)

    da = get_temporally_scaled_array(
        inventory,
        pd.date_range("2025-01-01", "2026-01-01", freq="MS", inclusive="left"),
        sum_over_cells=False,
    )
    # kg/y per cell back to the kg m-2 s-1 of the CAMS file.
    flux = da / SEC_PER_YR / xr.DataArray(areas, dims="cell")

    for category in ("ene", "res"):
        np.testing.assert_allclose(
            flux.sel(category=category, substance="CH4").transpose("time", "cell"),
            inventory.monthly_flux.sel(category=category),
            rtol=1e-12,
        )
