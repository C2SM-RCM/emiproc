from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import geopandas as gpd

from emiproc.exports.microhh import (
    export_cams_global_to_microhh,
    export_to_microhh,
    import_and_export_cams_global,
)
from emiproc.grids import WGS84, WGS84_NSIDC
from emiproc.inventories import Inventory
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


# The conservative remapping goes through `remap_inventory`, which computes the
# overlap weights in the inventory CRS while the emissions use true cell areas.
# That makes the exported fluxes agree with the analytic value to about 1e-5
# rather than to machine precision.
REMAP_RTOL = 1e-4


def _point_inventory(
    longitudes: list[float],
    latitudes: list[float],
    sectors: list[str],
    emissions_kg_s: list[float],
    substance: str = "CH4",
    year: int = 2025,
) -> Inventory:
    """Build a point source inventory in kg/y, as CorsoInventory would."""
    inv = Inventory()
    inv.name = "points"
    inv.year = year
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
        np.testing.assert_allclose(
            ds["emi_CH4"].isel(time=0, category=0), 1, rtol=REMAP_RTOL
        )
        np.testing.assert_allclose(
            ds["emi_CH4"].isel(time=0, category=1), 2, rtol=REMAP_RTOL
        )
        np.testing.assert_allclose(ds["emi_CH4_total"].isel(time=0), 3, rtol=REMAP_RTOL)
        assert ds["emi_CH4"].attrs["units"] == "kg m-2 s-1"

        # The target domain lies entirely inside the uniform source field, so
        # the exported mass rate is the source flux times the domain area.
        # `cell_areas` are true areas: EPSG:3857 metres are not ground metres,
        # so the projected area of the domain is about 0.7% larger.
        expected_rate = 1.0 * np.sum(grid.cell_areas)
        exported_rate = (ds["emi_CH4"].isel(time=0, category=0) * ds["cell_area"]).sum()
        np.testing.assert_allclose(exported_rate, expected_rate, rtol=REMAP_RTOL)


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
        np.testing.assert_allclose(ds["emi_CH4"].isel(time=0), 3, rtol=REMAP_RTOL)


def test_one_call_import_subtracts_point_sources_and_records_metadata(
    cams_file: Path, tmp_path: Path
) -> None:
    """Point sources already inside CAMS are removed to avoid double counting."""
    points = _point_inventory(
        longitudes=[-0.04],
        latitudes=[-0.04],
        sectors=["power"],
        # Exactly 1 kg/s as an annual mean rate.
        emissions_kg_s=[1.0],
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
        point_sources=points,
        category_mapping={"power": "sum"},
        negative_policy="allow",
    )

    with xr.open_dataset(baseline) as before, xr.open_dataset(adjusted) as after:
        # The subtraction happens on the annual emissions, so it is the annual
        # budget that must match. The monthly split of what was removed follows
        # the CAMS seasonality, see `subtract_inventories`.
        removed_rate = (
            ((before["emi_CH4"] - after["emi_CH4"]) * after["cell_area"])
            .sum(("category", "y", "x"))
            .mean("time")
        )

        source_grid = CAMSGlobal(cams_file, variables=["sum"]).grid
        source_cell = source_grid.gdf.geometry.to_crs(WGS84_NSIDC).iloc[0]
        target_domain = grid.gdf.geometry.to_crs(WGS84_NSIDC).union_all()
        # Only the part of the source cell inside the domain is exported.
        retained = source_cell.intersection(target_domain).area / source_cell.area
        np.testing.assert_allclose(removed_rate, 1.0 * retained, rtol=1e-3)

        # The full source was removed from the inventory before remapping.
        assert after.attrs["double_counting_clipped_values"] == 0
        assert str(round(SEC_PER_YR)) in after.attrs[
            "double_counting_removed_kg_y"
        ].replace(".0", "")


def test_point_sources_can_be_kept_at_their_own_resolution(
    cams_file: Path, tmp_path: Path
) -> None:
    """Merging keeps the total emissions while keeping the point geometry."""
    points = _point_inventory(
        longitudes=[-0.04],
        latitudes=[-0.04],
        sectors=["power"],
        emissions_kg_s=[1.0],
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
        cams_file, grid, tmp_path / "base.nc", variables=["sum"]
    )
    merged = import_and_export_cams_global(
        cams_file,
        grid,
        tmp_path / "merged.nc",
        variables=["sum"],
        point_sources=points,
        category_mapping={"power": "sum"},
        keep_point_sources=True,
    )

    # Removing then adding the same sources leaves the totals unchanged.
    with xr.open_dataset(baseline) as before, xr.open_dataset(merged) as after:
        total_before = (before["emi_CH4"] * before["cell_area"]).sum()
        total_after = (after["emi_CH4"] * after["cell_area"]).sum()
        np.testing.assert_allclose(total_after, total_before, rtol=REMAP_RTOL)


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
        np.testing.assert_allclose(
            dataset["emi_CH4"].isel(time=0), 7.0, rtol=REMAP_RTOL
        )
        assert dataset.attrs["input_file_count"] == 2
        assert "CAMS-GLOB-ANT_v6.2_methane_2025.nc" in dataset.attrs["input_files"]
        assert "CAMS-GLOB-ANT_v6.2_methane-part_2025.nc" in dataset.attrs["input_files"]


def test_load_multiple_files_subtracts_point_sources_after_summation(
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
    points = _point_inventory(
        longitudes=[-0.04],
        latitudes=[-0.04],
        sectors=["power"],
        emissions_kg_s=[requested_rate],
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
        point_sources=points,
        category_mapping={"power": "sum"},
    )

    with xr.open_dataset(output) as dataset:
        # The sources fit inside the summed CAMS emissions, so nothing is lost.
        assert dataset.attrs["double_counting_clipped_values"] == 0
        removed = dataset.attrs["double_counting_removed_kg_y"]
        assert (
            str(round(requested_rate * SEC_PER_YR))
            in removed.replace(
                str(requested_rate * SEC_PER_YR),
                str(round(requested_rate * SEC_PER_YR)),
            )
            or "sum/CH4" in removed
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
        np.testing.assert_allclose(
            dataset["emi_CH4"].isel(time=0), 46.0, rtol=REMAP_RTOL
        )
        assert dataset.sizes["time"] == 1
        assert pd.Timestamp(dataset["time"].values[0]) == requested_date
        assert pd.Timestamp(dataset["time_bnds"].values[0, 0]) == datetime(2025, 2, 10)
        assert pd.Timestamp(dataset["time_bnds"].values[0, 1]) == datetime(2025, 2, 11)


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
        np.testing.assert_allclose(
            dataset["emi_CH4"].isel(time=0), 5.0, rtol=REMAP_RTOL
        )


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
        np.testing.assert_allclose(emission.isel(time=0), 3.0, rtol=REMAP_RTOL)


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
        np.testing.assert_allclose(
            dataset["emi_VOCs"].isel(time=0), 7.0, rtol=REMAP_RTOL
        )
        details = dataset.attrs["input_details"]
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
        np.testing.assert_allclose(
            dataset["emi_CH4"].isel(time=0), 2.0, rtol=REMAP_RTOL
        )


def test_export_any_inventory_without_profiles(tmp_path: Path) -> None:
    """A plain inventory with no temporal profiles is exported as constant."""
    grid = MicroHHGrid(
        xsize=10_000, ysize=10_000, itot=2, jtot=2, lon=0, lat=0, crs="EPSG:3857"
    )
    points = _point_inventory(
        longitudes=[-0.01, 0.01],
        latitudes=[-0.01, 0.01],
        sectors=["power", "power"],
        emissions_kg_s=[1.0, 3.0],
    )

    output = export_to_microhh(points, grid, tmp_path / "points.nc")

    with xr.open_dataset(output) as ds:
        assert ds["emi_CH4"].dims == ("time", "category", "y", "x")
        assert ds.sizes["time"] == 1
        assert ds["category"].values.tolist() == ["power"]
        # All 4 kg/s land inside the domain and are conserved.
        total = (ds["emi_CH4"].isel(time=0) * ds["cell_area"]).sum()
        np.testing.assert_allclose(total, 4.0, rtol=REMAP_RTOL)


def test_export_corso_point_sources_to_microhh(tmp_path: Path) -> None:
    """A CORSO point source database can be exported to MicroHH directly."""
    from emiproc.inventories.corso import CorsoInventory

    def write_profile(filename: str, values: np.ndarray) -> None:
        columns = [f"value_{i}" for i in range(values.size)] + ["unused"]
        pd.DataFrame(
            [np.append(values, np.nan)],
            index=pd.Index([1], name="ID"),
            columns=columns,
        ).to_csv(tmp_path / filename, sep=";")

    write_profile("corso_ps_hourly_profiles_v2.0.csv", np.ones(24))
    write_profile("corso_ps_weekly_profiles_v2.0.csv", np.ones(7))
    write_profile("corso_ps_monthly_profiles_v2.0.csv", np.ones(12))
    write_profile("corso_ps_vertical_profiles_v2.0.csv", np.arange(16))
    pd.DataFrame(
        {
            "plant_name": ["a", "b"],
            "longitude": [-0.01, 0.01],
            "latitude": [-0.01, 0.01],
            "sector": ["power", "cement"],
            "ID_MonthFact": [1, 1],
            "ID_WeekFact": [1, 1],
            "ID_HourFact": [1, 1],
            "ID_VertProf": [1, 1],
            "co2_kty": SEC_PER_YR / 1_000_000 * np.array([1.0, 3.0]),
        },
        index=pd.Index([1, 2], name="ID"),
    ).to_csv(tmp_path / "corso_ps_catalogue_v2.0.csv", sep=";")

    inv = CorsoInventory(tmp_path, year=2025)
    grid = MicroHHGrid(
        xsize=10_000, ysize=10_000, itot=2, jtot=2, lon=0, lat=0, crs="EPSG:3857"
    )

    output = export_to_microhh(inv, grid, tmp_path / "corso.nc")

    with xr.open_dataset(output) as ds:
        assert ds["category"].values.tolist() == ["cement", "power"]
        assert ds.sizes["time"] == 12
        # Flat profiles, so every month carries the same 4 kg/s.
        total = (ds["emi_CO2"].isel(time=0) * ds["cell_area"]).sum()
        np.testing.assert_allclose(total, 4.0, rtol=REMAP_RTOL)


def test_point_source_remapping_uses_its_own_weights_file(
    cams_file: Path, tmp_path: Path
) -> None:
    """The two remappings must not share a weights file.

    ``get_weights_mapping`` caches by file name without checking that the
    shapes still match, so reusing one file would silently apply the CAMS to
    MicroHH weights to the point sources.
    """
    grid = MicroHHGrid(
        xsize=10_000, ysize=10_000, itot=2, jtot=2, lon=0, lat=0, crs="EPSG:3857"
    )
    weights = tmp_path / "weights.npz"

    import_and_export_cams_global(
        cams_file,
        grid,
        tmp_path / "out.nc",
        variables=["sum"],
        weights_file=weights,
        point_sources=_point_inventory(
            longitudes=[-0.04],
            latitudes=[-0.04],
            sectors=["power"],
            emissions_kg_s=[1.0],
        ),
        category_mapping={"power": "sum"},
    )

    written = sorted(p.name for p in tmp_path.glob("*.npz"))
    assert "weights.npz" in written
    assert any("point_sources" in name for name in written), written


def test_load_multiple_files_with_lumping_weights(tmp_path: Path) -> None:
    files = []
    for species, molecular_weight in [("propene", 42.0), ("toluene", 92.0)]:
        path = write_single_variable_cams_file(
            tmp_path / f"CAMS-GLOB-ANT_v6.2_{species}_2025.nc",
            value=molecular_weight,
            variable="sum",
            substance_label=species,
        )
        with xr.open_dataset(path, decode_times=False) as dataset:
            dataset = dataset.load()
        dataset["sum"].attrs["molecular_weight"] = molecular_weight
        dataset["sum"].attrs["molecular_weight_units"] = "g mole-1"
        dataset.to_netcdf(path)
        files.append(path)

    output = import_and_export_cams_global(
        files,
        _microhh_grid(),
        tmp_path / "lumped.nc",
        substance="VOCs",
        lumping_weights={"propene": 1.0, "toluene": 0.5},
    )

    with xr.open_dataset(output) as dataset:
        # 1 mol propene + 0.5 * 1 mol toluene, as kg of propene.
        np.testing.assert_allclose(
            dataset["emi_VOCs_total"].isel(time=0), 1.5 * 42.0, rtol=REMAP_RTOL
        )
        assert dataset["emi_VOCs"].attrs["molecular_weight"] == 42.0
        assert "lumping_weight=0.5" in dataset.attrs["input_details"]


def _microhh_grid(itot: int = 2) -> MicroHHGrid:
    return MicroHHGrid(
        xsize=10_000,
        ysize=10_000,
        itot=itot,
        jtot=itot,
        lon=0,
        lat=0,
        crs="EPSG:3857",
    )


def test_up_to_date_output_is_kept(cams_file: Path, tmp_path: Path) -> None:
    """A second identical call must not rebuild the file."""
    out = tmp_path / "cams.nc"
    first = import_and_export_cams_global(
        cams_file, _microhh_grid(), out, variables=["sum"]
    )
    stamp = first.stat().st_mtime_ns

    again = import_and_export_cams_global(
        cams_file, _microhh_grid(), out, variables=["sum"]
    )

    assert again == first
    assert again.stat().st_mtime_ns == stamp, "the file was rebuilt"


def test_overwrite_forces_a_rebuild(cams_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "cams.nc"
    first = import_and_export_cams_global(
        cams_file, _microhh_grid(), out, variables=["sum"]
    )
    stamp = first.stat().st_mtime_ns

    again = import_and_export_cams_global(
        cams_file, _microhh_grid(), out, variables=["sum"], overwrite=True
    )

    assert again.stat().st_mtime_ns != stamp


def test_changed_inputs_replace_the_output(cams_file: Path, tmp_path: Path) -> None:
    """A file built from other inputs is replaced, not reused."""
    out = tmp_path / "cams.nc"
    import_and_export_cams_global(cams_file, _microhh_grid(2), out, variables=["sum"])
    with xr.open_dataset(out) as ds:
        assert ds["emi_CH4"].shape[-1] == 2

    # A different grid must invalidate the cached file.
    import_and_export_cams_global(cams_file, _microhh_grid(4), out, variables=["sum"])
    with xr.open_dataset(out) as ds:
        assert ds["emi_CH4"].shape[-1] == 4

    # So must different point sources.
    import_and_export_cams_global(
        cams_file,
        _microhh_grid(4),
        out,
        variables=["sum"],
        point_sources=_point_inventory(
            longitudes=[-0.04],
            latitudes=[-0.04],
            sectors=["power"],
            emissions_kg_s=[1.0],
        ),
    )
    with xr.open_dataset(out) as ds:
        assert "double_counting_removed_kg_y" in ds.attrs


def test_output_can_be_rewritten_while_open(cams_file: Path, tmp_path: Path) -> None:
    """An open handle must not block the rewrite, see `write_netcdf`."""
    out = tmp_path / "cams.nc"
    import_and_export_cams_global(cams_file, _microhh_grid(), out, variables=["sum"])

    held = xr.open_dataset(out)
    _ = held["emi_CH4"].values
    try:
        import_and_export_cams_global(
            cams_file, _microhh_grid(), out, variables=["sum"], overwrite=True
        )
    finally:
        held.close()


def test_summed_cams_subtracts_every_point_source_category(
    cams_file: Path, tmp_path: Path
) -> None:
    """Summed CAMS files hold one 'sum' category to subtract from.

    `DEFAULT_CORSO_SECTOR_MAPPING` targets ene/ind/ref, which do not exist
    there, so the mapping must be redirected onto 'sum' instead of silently
    subtracting nothing.
    """
    from emiproc.inventories.cams_global import DEFAULT_CORSO_SECTOR_MAPPING

    points = _point_inventory(
        longitudes=[-0.04],
        latitudes=[-0.04],
        sectors=["power"],
        emissions_kg_s=[1.0],
    )
    out = import_and_export_cams_global(
        [cams_file],  # a sequence always takes the summation path
        _microhh_grid(),
        tmp_path / "summed.nc",
        point_sources=points,
        category_mapping=DEFAULT_CORSO_SECTOR_MAPPING,
    )

    with xr.open_dataset(out) as ds:
        assert ds["category"].values.tolist() == ["sum"]
        assert "sum/CH4" in ds.attrs["double_counting_removed_kg_y"]
