"""Test file for the Saunois inventory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import emiproc
from emiproc.inventories.saunois import Saunois

saunois_path = emiproc.FILES_DIR / "saunois" / "GCP_Prior_CH4_fluxes.nc"

lon = np.array([10.0, 11.0])
lat = np.array([45.0, 46.0])


def _make_flux_da(time: pd.DatetimeIndex | np.ndarray, time_dim: str) -> xr.DataArray:
    rng = np.random.default_rng(0)
    data = rng.random((len(time), len(lat), len(lon)))
    return xr.DataArray(
        data,
        dims=(time_dim, "lat", "lon"),
        coords={time_dim: time, "lat": lat, "lon": lon},
        attrs={"units": "kg/m2/s"},
    )


@pytest.fixture(scope="module")
def saunois_test_file(tmp_path_factory) -> Path:
    file = tmp_path_factory.mktemp("saunois") / "GCP_Prior_CH4_fluxes_test.nc"

    time = pd.date_range("2018-01-01", periods=24, freq="MS")

    ds = xr.Dataset(
        {
            # A category with a yearly time series spanning several years.
            "flux_ch4_wetlands": _make_flux_da(time, "time"),
            # A category only available as a fixed 12-month climatology.
            "flux_ch4_termites": _make_flux_da(np.arange(12), "time_climato"),
        }
    )
    ds.to_netcdf(file)
    return file


def test_saunois_read(saunois_test_file):
    inv = Saunois(saunois_test_file, year=2018)

    assert inv.year == 2018
    assert set(inv.categories) == {"wetlands", "termites"}
    assert set(inv.substances) == {"CH4"}
    # 2x2 grid
    assert len(inv.gdf) == 4
    assert not inv.gdf[("wetlands", "CH4")].isna().any()
    assert not inv.gdf[("termites", "CH4")].isna().any()


def test_saunois_missing_year_raises(saunois_test_file):
    with pytest.raises(ValueError):
        Saunois(saunois_test_file, year=2025)


def test_saunois_wrong_units_raises(tmp_path_factory):
    file = tmp_path_factory.mktemp("saunois_wrong_units") / "bad_units.nc"
    time = pd.date_range("2018-01-01", periods=12, freq="MS")
    da = _make_flux_da(time, "time")
    da.attrs["units"] = "g CH4 m-2 d-1"
    xr.Dataset({"flux_ch4_wetlands": da}).to_netcdf(file)

    with pytest.raises(ValueError):
        Saunois(file, year=2018)


def test_saunois_inconsistent_units_raises(tmp_path_factory):
    file = tmp_path_factory.mktemp("saunois_inconsistent_units") / "mixed_units.nc"
    time = pd.date_range("2018-01-01", periods=12, freq="MS")

    da_wetlands = _make_flux_da(time, "time")
    da_termites = _make_flux_da(time, "time")
    da_termites.attrs["units"] = "g CH4 m-2 d-1"

    xr.Dataset(
        {"flux_ch4_wetlands": da_wetlands, "flux_ch4_termites": da_termites}
    ).to_netcdf(file)

    with pytest.raises(ValueError, match="Inconsistent units"):
        Saunois(file, year=2018)


@pytest.mark.slow
def test_saunois_read_real_file():
    if not saunois_path.exists():
        raise FileNotFoundError(
            f"File {saunois_path} not found, please add it to {saunois_path}"
        )

    inv = Saunois(saunois_path, year=2015)

    assert inv.year == 2015
    assert set(inv.categories) == {
        "biofuels",
        "biomass",
        "coal",
        "freshwaters",
        "geological",
        "livestock",
        "ocean",
        "oilgasind",
        "rice",
        "soils",
        "termites",
        "waste",
        "wetlands",
    }
    assert set(inv.substances) == {"CH4"}
    # 180x360 global 1deg grid
    assert len(inv.gdf) == 180 * 360
    for cat in inv.categories:
        assert not inv.gdf[(cat, "CH4")].isna().all()
