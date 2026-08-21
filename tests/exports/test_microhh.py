from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from emiproc.exports.microhh import (
    export_cams_glob_ant_to_microhh,
    import_and_export_cams_glob_ant,
)
from emiproc.grids import MicroHHGrid
from emiproc.inventories.cams_glob_ant import CAMSGlobAnt
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


def test_export_to_microhh_grid(cams_file: Path, tmp_path: Path) -> None:
    inventory = CAMSGlobAnt(cams_file)
    grid = MicroHHGrid(
        xsize=10_000,
        ysize=10_000,
        itot=2,
        jtot=2,
        lon=0,
        lat=0,
        crs="EPSG:3857",
    )

    output = export_cams_glob_ant_to_microhh(
        inventory,
        grid,
        tmp_path / "microhh_emissions",
        weights_file=tmp_path / "weights",
    )

    assert output.suffix == ".nc"
    assert (tmp_path / "weights.npz").exists()
    with xr.open_dataset(output) as ds:
        assert ds["emi_CH4"].dims == ("time", "category", "y", "x")
        assert ds["emi_CH4"].shape == (12, 2, 2, 2)
        np.testing.assert_allclose(ds["x"], [2500, 7500])
        np.testing.assert_allclose(ds["y"], [2500, 7500])
        np.testing.assert_allclose(ds["emi_CH4"].isel(time=0, category=0), 1)
        np.testing.assert_allclose(ds["emi_CH4"].isel(time=0, category=1), 2)
        np.testing.assert_allclose(ds["emi_CH4_total"].isel(time=0), 3)
        assert ds["emi_CH4"].attrs["units"] == "kg m-2 s-1"

        source_shapes = inventory.grid.gdf.geometry.to_crs(grid.crs)
        domain = grid.gdf.geometry.union_all()
        overlap_areas = source_shapes.intersection(domain).area.to_numpy()
        expected_rate = (
            inventory.monthly_flux.isel(time=0, category=0).values * overlap_areas
        ).sum()
        exported_rate = (
            ds["emi_CH4"].isel(time=0, category=0) * ds["cell_area"]
        ).sum()
        np.testing.assert_allclose(exported_rate, expected_rate, rtol=1e-12)


def test_one_call_import_crops_to_grid(cams_file: Path, tmp_path: Path) -> None:
    grid = MicroHHGrid(
        xsize=2_000,
        ysize=2_000,
        itot=2,
        jtot=2,
        lon=0,
        lat=0,
        crs="EPSG:3857",
    )
    output = import_and_export_cams_glob_ant(
        cams_file,
        grid,
        tmp_path / "one_call.nc",
        variables=["sum"],
    )

    with xr.open_dataset(output) as ds:
        assert ds.sizes["time"] == 12
        assert ds.sizes["category"] == 1
        assert ds.sizes["y"] == 2
        assert ds.sizes["x"] == 2
        assert ds.sizes["bounds"] == 2
        np.testing.assert_allclose(ds["emi_CH4"].isel(time=0), 3)


def test_one_call_import_subtracts_corso_and_records_metadata(
    cams_file: Path, tmp_path: Path
) -> None:
    corso = xr.Dataset(
        data_vars={
            "longitude": ("ID", [-0.04]),
            "latitude": ("ID", [-0.04]),
            "sector": ("ID", ["power"]),
            # Exactly 1 kg/s as an annual mean rate.
            "ch4_kty": ("ID", [SEC_PER_YR / 1_000_000]),
        },
        coords={"ID": [1]},
    )
    grid = MicroHHGrid(
        xsize=10_000,
        ysize=10_000,
        itot=2,
        jtot=2,
        lon=0,
        lat=0,
        crs="EPSG:3857",
    )

    baseline = import_and_export_cams_glob_ant(
        cams_file,
        grid,
        tmp_path / "baseline.nc",
        variables=["sum"],
    )
    adjusted = import_and_export_cams_glob_ant(
        cams_file,
        grid,
        tmp_path / "adjusted.nc",
        variables=["sum"],
        corso_dataset=corso,
        corso_negative_policy="allow",
    )

    with xr.open_dataset(baseline) as before, xr.open_dataset(adjusted) as after:
        removed_rate = (
            (before["emi_CH4"] - after["emi_CH4"]) * after["cell_area"]
        ).sum(("category", "y", "x"))
        source_cell = CAMSGlobAnt(cams_file, variables=["sum"]).grid.gdf.geometry
        source_cell = source_cell.to_crs(grid.crs).iloc[0]
        target_domain = grid.gdf.geometry.union_all()
        physical_cell_area = np.asarray(
            CAMSGlobAnt(cams_file, variables=["sum"]).grid.cell_areas
        )[0]
        expected_rate = (
            source_cell.intersection(target_domain).area / physical_cell_area
        )
        np.testing.assert_allclose(removed_rate, expected_rate, rtol=2e-6)
        assert after.attrs["corso_emission_variable"] == "ch4_kty"
        assert after.attrs["corso_mapped_sources"] == 1
        assert after.attrs["corso_requested_annual_mean_kg_s"] == pytest.approx(1.0)
