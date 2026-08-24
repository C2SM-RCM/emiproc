from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from emiproc.exports.microhh import (
    export_cams_global_to_microhh,
    import_and_export_cams_global,
)
from emiproc.grids import MicroHHGrid
from emiproc.inventories.cams_global import CAMSGlobal
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


def write_single_variable_cams_file(
    path: Path,
    value: float,
    variable: str = "custom",
    year: int = 2025,
    substance_label: str = "CH4",
    latitudes: list[float] | None = None,
    longitudes: list[float] | None = None,
    bare_long_name: bool = False,
) -> Path:
    latitudes = [-0.05, 0.05] if latitudes is None else latitudes
    longitudes = [-0.05, 0.05] if longitudes is None else longitudes
    long_name = (
        substance_label
        if bare_long_name
        else f"Emissions of {substance_label} for {variable} sector"
    )
    dataset = xr.Dataset(
        data_vars={
            variable: (
                ("time", "lat", "lon"),
                np.full(
                    (12, len(latitudes), len(longitudes)),
                    value,
                    dtype=np.float64,
                ),
                {
                    "units": "kg m-2 s-1",
                    "long_name": long_name,
                },
            )
        },
        coords={
            "time": (
                "time",
                np.arange(12),
                {"units": f"months since {year}-01-01"},
            ),
            "lat": latitudes,
            "lon": longitudes,
        },
    )
    dataset.to_netcdf(path)
    return path


def write_daily_cams_file(
    path: Path,
    variable: str = "shipping",
    latitudes: list[float] | None = None,
    substance_label: str = "CH4",
) -> Path:
    latitudes = [-0.05, 0.05] if latitudes is None else latitudes
    longitudes = [-0.05, 0.05] if len(latitudes) == 2 else [-1.0, 0.0, 1.0]
    daily_values = np.arange(365, dtype=np.float64)[:, None, None]
    values = np.broadcast_to(
        daily_values,
        (365, len(latitudes), len(longitudes)),
    ).copy()
    dataset = xr.Dataset(
        data_vars={
            variable: (
                ("time", "lat", "lon"),
                values,
                {
                    "units": "kg m-2 s-1",
                    "long_name": f"{substance_label} emissions (daily mean)",
                },
            )
        },
        coords={
            "time": (
                "time",
                np.arange(365),
                {"units": "days since 2025-01-01 00:00"},
            ),
            "lat": latitudes,
            "lon": longitudes,
        },
    )
    dataset.to_netcdf(path)
    return path


def test_previous_export_names_remain_compatible() -> None:
    from emiproc.exports.microhh import (
        export_cams_glob_ant_to_microhh,
        import_and_export_cams_glob_ant,
    )

    assert export_cams_glob_ant_to_microhh is export_cams_global_to_microhh
    assert import_and_export_cams_glob_ant is import_and_export_cams_global


def test_export_to_microhh_grid(cams_file: Path, tmp_path: Path) -> None:
    inventory = CAMSGlobal(cams_file)
    grid = MicroHHGrid(
        xsize=10_000,
        ysize=10_000,
        itot=2,
        jtot=2,
        lon=0,
        lat=0,
        crs="EPSG:3857",
    )

    output = export_cams_global_to_microhh(
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
    output = import_and_export_cams_global(
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

    baseline = import_and_export_cams_global(
        cams_file,
        grid,
        tmp_path / "baseline.nc",
        variables=["sum"],
    )
    adjusted = import_and_export_cams_global(
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
        source_cell = CAMSGlobal(cams_file, variables=["sum"]).grid.gdf.geometry
        source_cell = source_cell.to_crs(grid.crs).iloc[0]
        target_domain = grid.gdf.geometry.union_all()
        physical_cell_area = np.asarray(
            CAMSGlobal(cams_file, variables=["sum"]).grid.cell_areas
        )[0]
        expected_rate = (
            source_cell.intersection(target_domain).area / physical_cell_area
        )
        np.testing.assert_allclose(removed_rate, expected_rate, rtol=2e-6)
        assert after.attrs["corso_emission_variable"] == "ch4_kty"
        assert after.attrs["corso_mapped_sources"] == 1
        assert after.attrs["corso_requested_annual_mean_kg_s"] == pytest.approx(1.0)


def test_load_multiple_files_prefers_sum_and_accepts_sole_variable(
    cams_file: Path, tmp_path: Path
) -> None:
    single_file = write_single_variable_cams_file(
        tmp_path / "CAMS-GLOB-ANT_v6.2_methane-part_2025.nc",
        value=4.0,
    )
    grid = MicroHHGrid(
        xsize=2_000,
        ysize=2_000,
        itot=2,
        jtot=2,
        lon=0,
        lat=0,
        crs="EPSG:3857",
    )

    output = import_and_export_cams_global(
        [cams_file, single_file],
        grid,
        tmp_path / "summed.nc",
    )

    with xr.open_dataset(output) as dataset:
        assert dataset["category"].values.tolist() == ["sum"]
        np.testing.assert_allclose(dataset["emi_CH4"].isel(time=0), 7.0)
        assert dataset.attrs["cams_global_input_file_count"] == 2
        assert "CAMS-GLOB-ANT_v6.2_methane_2025.nc" in dataset.attrs[
            "cams_global_input_files"
        ]
        assert "CAMS-GLOB-ANT_v6.2_methane-part_2025.nc" in dataset.attrs[
            "cams_global_input_files"
        ]


def test_load_multiple_files_subtracts_corso_after_summation(
    tmp_path: Path,
) -> None:
    first_file = write_single_variable_cams_file(
        tmp_path / "CAMS-GLOB-ANT_v6.2_methane-part-a_2025.nc",
        value=1e-8,
    )
    second_file = write_single_variable_cams_file(
        tmp_path / "CAMS-GLOB-ANT_v6.2_methane-part-b_2025.nc",
        value=1e-8,
        latitudes=[-0.125, 0.125],
        longitudes=[-0.125, 0.125],
    )
    source_inventory = CAMSGlobal(first_file, variables=["custom"])
    requested_rate = float(source_inventory.grid.cell_areas[0]) * 1.5e-8
    corso = xr.Dataset(
        data_vars={
            "longitude": ("ID", [-0.04]),
            "latitude": ("ID", [-0.04]),
            "sector": ("ID", ["power"]),
            "ch4_kty": ("ID", [requested_rate * SEC_PER_YR / 1_000_000]),
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

    output = import_and_export_cams_global(
        [first_file, second_file],
        grid,
        tmp_path / "summed_then_subtracted.nc",
        corso_dataset=corso,
    )

    with xr.open_dataset(output) as dataset:
        assert dataset.attrs["corso_negative_values"] == 0
        assert dataset.attrs["corso_clipped_values"] == 0
        assert dataset.attrs["corso_requested_annual_mean_kg_s"] == pytest.approx(
            requested_rate
        )
        assert dataset.attrs["corso_actual_monthly_mean_kg_s"] == pytest.approx(
            requested_rate
        )


def test_load_multiple_files_rejects_ambiguous_variables(
    cams_file: Path, tmp_path: Path
) -> None:
    ambiguous_file = tmp_path / "CAMS-GLOB-ANT_v6.2_methane_ambiguous_2025.nc"
    with xr.open_dataset(cams_file, decode_times=False) as dataset:
        dataset[["ene", "res"]].load().to_netcdf(ambiguous_file)
    grid = MicroHHGrid(
        xsize=2_000,
        ysize=2_000,
        itot=2,
        jtot=2,
        lon=0,
        lat=0,
        crs="EPSG:3857",
    )

    with pytest.raises(ValueError, match="expected a 'sum' field"):
        import_and_export_cams_global(
            [ambiguous_file],
            grid,
            tmp_path / "ambiguous.nc",
        )

    with pytest.raises(ValueError, match="`variables` cannot be provided"):
        import_and_export_cams_global(
            [cams_file],
            grid,
            tmp_path / "variables.nc",
            variables=["sum"],
        )


def test_date_selects_monthly_and_daily_values_before_summation(
    cams_file: Path, tmp_path: Path
) -> None:
    shipping_file = write_daily_cams_file(
        tmp_path / "CAMS-GLOB-SHIP_v3.2_methane_2025.nc"
    )
    requested_date = datetime(2025, 2, 10, 12, 30)
    grid = MicroHHGrid(
        xsize=2_000,
        ysize=2_000,
        itot=2,
        jtot=2,
        lon=0,
        lat=0,
        crs="EPSG:3857",
    )

    output = import_and_export_cams_global(
        [cams_file, shipping_file],
        grid,
        tmp_path / "selected_date.nc",
        date=requested_date,
    )

    with xr.open_dataset(output) as dataset:
        # February CAMS sum is 6; February 10 is zero-based day 40.
        np.testing.assert_allclose(dataset["emi_CH4"].isel(time=0), 46.0)
        assert dataset.sizes["time"] == 1
        assert pd.Timestamp(dataset["time"].values[0]) == requested_date
        assert pd.Timestamp(dataset["time_bnds"].values[0, 0]) == datetime(
            2025, 2, 10
        )
        assert pd.Timestamp(dataset["time_bnds"].values[0, 1]) == datetime(
            2025, 2, 11
        )


def test_date_supports_daily_irregular_latitude_grid(tmp_path: Path) -> None:
    ocean_file = write_daily_cams_file(
        tmp_path / "CAMS-GLOB-OCE_v4.1_methane_2025.nc",
        variable="emi_oceans",
        latitudes=[-0.8, 0.0, 1.2],
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

    output = import_and_export_cams_global(
        ocean_file,
        grid,
        tmp_path / "ocean_daily.nc",
        date=datetime(2025, 1, 6),
    )

    with xr.open_dataset(output) as dataset:
        np.testing.assert_allclose(dataset["emi_CH4"].isel(time=0), 5.0)


def test_daily_inventory_requires_available_datetime(
    tmp_path: Path,
) -> None:
    shipping_file = write_daily_cams_file(
        tmp_path / "CAMS-GLOB-SHIP_v3.2_methane_2025.nc"
    )
    grid = MicroHHGrid(
        xsize=2_000,
        ysize=2_000,
        itot=2,
        jtot=2,
        lon=0,
        lat=0,
        crs="EPSG:3857",
    )

    with pytest.raises(ValueError, match="Pass a datetime with `date=`"):
        import_and_export_cams_global(
            shipping_file,
            grid,
            tmp_path / "missing_date.nc",
        )
    with pytest.raises(ValueError, match="is not available"):
        import_and_export_cams_global(
            shipping_file,
            grid,
            tmp_path / "outside_year.nc",
            date=datetime(2026, 1, 1),
        )


@pytest.mark.parametrize(
    ("monthly_label", "shipping_label", "canonical_name"),
    [
        ("CO2_s_c", "Carbon dioxide", "CO2"),
        ("CO", "Carbon monoxide", "CO"),
    ],
)
def test_substance_aliases_allow_co2_and_co_summation(
    tmp_path: Path,
    monthly_label: str,
    shipping_label: str,
    canonical_name: str,
) -> None:
    monthly_file = write_single_variable_cams_file(
        tmp_path / f"monthly_{canonical_name}.nc",
        value=2.0,
        variable="sum",
        substance_label=monthly_label,
    )
    shipping_file = write_daily_cams_file(
        tmp_path / f"shipping_{canonical_name}.nc",
        substance_label=shipping_label,
    )
    grid = MicroHHGrid(
        xsize=2_000,
        ysize=2_000,
        itot=2,
        jtot=2,
        lon=0,
        lat=0,
        crs="EPSG:3857",
    )

    output = import_and_export_cams_global(
        [monthly_file, shipping_file],
        grid,
        tmp_path / f"summed_{canonical_name}.nc",
        date=datetime(2025, 1, 2),
    )

    with xr.open_dataset(output) as dataset:
        emission = dataset[f"emi_{canonical_name}"]
        np.testing.assert_allclose(emission.isel(time=0), 3.0)


def test_voc_family_inference_and_mixed_grid_remapping(tmp_path: Path) -> None:
    anthropogenic = write_single_variable_cams_file(
        tmp_path / "CAMS-GLOB-ANT_v6.2_non-methane-vocs_2025.nc",
        value=2.0,
        variable="sum",
        substance_label="NMVOCs",
    )
    shipping = write_daily_cams_file(
        tmp_path / "CAMS-GLOB-SHIP_v3.2_vocs-all_2025.nc",
        substance_label="Volatile organic compounds (all)",
    )
    biogenic = write_single_variable_cams_file(
        tmp_path / "CAMS-GLOB-BIO_v4.0_isoprene_2025.nc",
        value=4.0,
        variable="emiss_bio",
        substance_label="isoprene",
        latitudes=[-0.125, 0.125],
        longitudes=[-0.125, 0.125],
        bare_long_name=True,
    )
    grid = MicroHHGrid(
        xsize=2_000,
        ysize=2_000,
        itot=2,
        jtot=2,
        lon=0,
        lat=0,
        crs="EPSG:3857",
    )

    output = import_and_export_cams_global(
        [shipping, biogenic, anthropogenic],
        grid,
        tmp_path / "summed_vocs.nc",
        date=datetime(2025, 1, 2),
    )

    with xr.open_dataset(output) as dataset:
        np.testing.assert_allclose(dataset["emi_VOCs"].isel(time=0), 7.0)
        details = dataset.attrs["cams_global_input_details"]
        assert "variable=shipping" in details
        assert "variable=emiss_bio" in details
        assert "variable=sum" in details


def test_emission_selection_ignores_ancillary_data_fields(tmp_path: Path) -> None:
    source = write_single_variable_cams_file(
        tmp_path / "source.nc",
        value=2.0,
    )
    with xr.open_dataset(source, decode_times=False) as dataset:
        with_ancillary = dataset.load().assign(
            quality_flag=("time", np.zeros(12, dtype=np.int8))
        )
    path = tmp_path / "source_with_quality_flag.nc"
    with_ancillary.to_netcdf(path)
    grid = MicroHHGrid(
        xsize=2_000,
        ysize=2_000,
        itot=2,
        jtot=2,
        lon=0,
        lat=0,
        crs="EPSG:3857",
    )

    output = import_and_export_cams_global(
        [path],
        grid,
        tmp_path / "selected_emission.nc",
    )

    with xr.open_dataset(output) as dataset:
        np.testing.assert_allclose(dataset["emi_CH4"].isel(time=0), 2.0)
