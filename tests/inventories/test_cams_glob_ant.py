from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from emiproc.inventories.cams_glob_ant import (
    CAMSGlobAnt,
)
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


@pytest.fixture
def corso_dataset() -> xr.Dataset:
    one_kg_per_second_in_kty = SEC_PER_YR / 1_000_000
    return xr.Dataset(
        data_vars={
            "longitude": ("ID", [-0.04, -0.04, 1.0]),
            "latitude": ("ID", [-0.04, 0.04, 0.0]),
            "sector": ("ID", ["power", "cement", "power"]),
            "ch4_kty": (
                "ID",
                one_kg_per_second_in_kty * np.array([1.0, 2.0, 4.0]),
            ),
        },
        coords={"ID": [1, 2, 3]},
    )


def test_import_cams_glob_ant(cams_file: Path) -> None:
    inventory = CAMSGlobAnt(cams_file)

    assert inventory.year == 2025
    assert inventory.substance_name == "CH4"
    assert inventory.monthly_flux.dims == ("time", "category", "cell")
    assert inventory.monthly_flux.shape == (12, 2, 4)
    assert list(inventory.monthly_flux.category.values) == ["ene", "res"]
    assert pd.Timestamp(inventory.monthly_flux.time.values[0]).month == 1
    assert np.all(inventory.monthly_flux.sel(category="ene").isel(time=0) == 1)


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

    inventory = CAMSGlobAnt(path, variables=["sum"])

    assert inventory.grid.dx == pytest.approx(0.1)
    assert inventory.grid.dy == pytest.approx(0.1)


def test_subtract_corso_converts_units_and_maps_point_to_cell(
    cams_file: Path, corso_dataset: xr.Dataset
) -> None:
    original = CAMSGlobAnt(cams_file)
    adjusted = CAMSGlobAnt(
        cams_file,
        corso_dataset=corso_dataset,
        corso_sector_mapping={"cement": "res"},
        corso_negative_policy="allow",
    )

    areas = np.asarray(adjusted.grid.cell_areas)
    ene_difference = (
        original.monthly_flux.sel(category="ene")
        - adjusted.monthly_flux.sel(category="ene")
    )
    res_difference = (
        original.monthly_flux.sel(category="res")
        - adjusted.monthly_flux.sel(category="res")
    )
    expected_ene = np.zeros((12, 4))
    expected_res = np.zeros((12, 4))
    expected_ene[:, 0] = 1.0 / areas[0]
    expected_res[:, 1] = 2.0 / areas[1]
    np.testing.assert_allclose(ene_difference, expected_ene)
    np.testing.assert_allclose(res_difference, expected_res)
    assert adjusted.corso_subtraction_summary["mapped_sources"] == 2
    assert adjusted.corso_subtraction_summary["outside_sources"] == 1
    assert adjusted.corso_subtraction_summary[
        "requested_annual_mean_kg_s"
    ] == pytest.approx(3.0)


def test_subtract_all_corso_sectors_from_cams_sum(
    cams_file: Path, corso_dataset: xr.Dataset
) -> None:
    original = CAMSGlobAnt(cams_file, variables=["sum"])
    adjusted = CAMSGlobAnt(
        cams_file,
        variables=["sum"],
        corso_dataset=corso_dataset,
        corso_negative_policy="allow",
    )

    areas = np.asarray(adjusted.grid.cell_areas)
    difference = original.monthly_flux - adjusted.monthly_flux
    expected = np.zeros((12, 1, 4))
    expected[:, 0, 0] = 1.0 / areas[0]
    expected[:, 0, 1] = 2.0 / areas[1]
    # Subtracting a small flux from the synthetic O(1) CAMS values incurs
    # cancellation error in the recovered difference.
    np.testing.assert_allclose(difference, expected, rtol=1e-6)


def test_corso_negative_policy_clip_and_raise(
    cams_file: Path, corso_dataset: xr.Dataset
) -> None:
    excessive = corso_dataset.isel(ID=[0]).copy(deep=True)
    excessive["ch4_kty"][:] = 1e20

    with pytest.warns(RuntimeWarning, match="residuals were clipped"):
        clipped = CAMSGlobAnt(cams_file, corso_dataset=excessive)
    assert np.all(clipped.monthly_flux.sel(category="ene").isel(cell=0) == 0)
    assert clipped.corso_subtraction_summary["clipped_values"] == 12

    with pytest.raises(ValueError, match="negative CAMS values"):
        CAMSGlobAnt(
            cams_file,
            corso_dataset=excessive,
            corso_negative_policy="raise",
        )
