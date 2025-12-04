"""Tests for the NetcdfRaster inventory class."""

import warnings
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from emiproc.grids import RegularGrid
from emiproc.inventories import Inventory
from emiproc.inventories.netcdf_raster import (
    NetcdfRaster,
    get_unit_scaling_factor_to_kg_per_year_per_cell,
    get_year_from_attrs,
)
from emiproc.profiles.temporal.profiles import MounthsProfile, DayOfYearProfile
from emiproc.tests_utils import TEST_OUTPUTS_DIR
from emiproc.tests_utils.test_grids import regular_grid
from emiproc.tests_utils.test_inventories import inv_with_pnt_sources
from emiproc.utilities import DAY_PER_YR, SEC_PER_DAY


# Grid dimensions from test_utils regular_grid (nx=10, ny=15)
GRID_NX = regular_grid.nx
GRID_NY = regular_grid.ny
GRID_NCELLS = GRID_NX * GRID_NY  # 150 cells


# Suppress geopandas warnings about geographic CRS
@pytest.fixture(autouse=True)
def suppress_geopandas_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")
        yield


@pytest.fixture
def simple_netcdf_file(tmp_path):
    """Create a simple NetCDF file for testing using regular_grid from test_utils."""
    file_path = tmp_path / "simple_raster.nc"

    # Create emission data matching regular_grid dimensions (ny x nx)
    emission_data = np.random.rand(GRID_NY, GRID_NX).astype(np.float32)

    ds = xr.Dataset(
        data_vars={
            "CO2_industry": (
                ["lat", "lon"],
                emission_data,
                {
                    "units": "kg/year/cell",
                    "substance": "CO2",
                    "category": "industry",
                },
            ),
        },
        coords={
            "lon": regular_grid.lon_range,
            "lat": regular_grid.lat_range,
        },
        attrs={"year": 2020},
    )

    ds.to_netcdf(file_path)
    return file_path


@pytest.fixture
def multi_variable_netcdf_file(tmp_path):
    """Create a NetCDF file with multiple emission variables."""
    file_path = tmp_path / "multi_variable_raster.nc"

    ds = xr.Dataset(
        data_vars={
            "CO2_industry": (
                ["lat", "lon"],
                np.random.rand(GRID_NY, GRID_NX).astype(np.float32),
                {
                    "units": "kg/year/cell",
                    "substance": "CO2",
                    "category": "industry",
                },
            ),
            "CO2_transport": (
                ["lat", "lon"],
                np.random.rand(GRID_NY, GRID_NX).astype(np.float32),
                {
                    "units": "kg/year/cell",
                    "substance": "CO2",
                    "category": "transport",
                },
            ),
            "CH4_agriculture": (
                ["lat", "lon"],
                np.random.rand(GRID_NY, GRID_NX).astype(np.float32),
                {
                    "units": "kg/year/cell",
                    "substance": "CH4",
                    "category": "agriculture",
                },
            ),
        },
        coords={
            "lon": regular_grid.lon_range,
            "lat": regular_grid.lat_range,
        },
        attrs={"year": 2021},
    )

    ds.to_netcdf(file_path)
    return file_path


@pytest.fixture
def netcdf_file_kg_per_m2_per_s(tmp_path):
    """Create a NetCDF file with emissions in kg/m2/s."""
    file_path = tmp_path / "kg_m2_s_raster.nc"

    ds = xr.Dataset(
        data_vars={
            "CO2_power": (
                ["lat", "lon"],
                np.ones((GRID_NY, GRID_NX)).astype(np.float32),  # 1 kg/m2/s everywhere
                {
                    "units": "kg/m2/s",
                    "substance": "CO2",
                    "category": "power",
                },
            ),
        },
        coords={
            "lon": regular_grid.lon_range,
            "lat": regular_grid.lat_range,
        },
        attrs={"year": 2022},
    )

    ds.to_netcdf(file_path)
    return file_path


@pytest.fixture
def netcdf_file_with_time(tmp_path):
    """Create a NetCDF file with a single time dimension."""
    file_path = tmp_path / "time_raster.nc"

    time_values = pd.date_range("2023-06-01", periods=1)

    ds = xr.Dataset(
        data_vars={
            "CO2_heating": (
                ["time", "lat", "lon"],
                np.random.rand(1, GRID_NY, GRID_NX).astype(np.float32),
                {
                    "units": "kg/year/cell",
                    "substance": "CO2",
                    "category": "heating",
                },
            ),
        },
        coords={
            "lon": regular_grid.lon_range,
            "lat": regular_grid.lat_range,
            "time": time_values,
        },
    )

    ds.to_netcdf(file_path)
    return file_path


@pytest.fixture
def netcdf_file_with_monthly_profiles(tmp_path):
    """Create a NetCDF file with 12 monthly time steps."""
    file_path = tmp_path / "monthly_raster.nc"

    # Create time values for each month of a single year
    time_values = pd.to_datetime([f"2023-{m:02d}-15" for m in range(1, 13)])

    # Create monthly varying emissions (12 months)
    emission_data = np.random.rand(12, GRID_NY, GRID_NX).astype(np.float32)

    ds = xr.Dataset(
        data_vars={
            "CO2_seasonal": (
                ["time", "lat", "lon"],
                emission_data,
                {
                    "units": "kg/year/cell",
                    "substance": "CO2",
                    "category": "seasonal",
                },
            ),
        },
        coords={
            "lon": regular_grid.lon_range,
            "lat": regular_grid.lat_range,
            "time": time_values,
        },
    )

    ds.to_netcdf(file_path)
    return file_path


@pytest.fixture
def netcdf_file_no_unit_attr(tmp_path):
    """Create a NetCDF file without unit attribute."""
    file_path = tmp_path / "no_unit_raster.nc"

    ds = xr.Dataset(
        data_vars={
            "emissions": (
                ["lat", "lon"],
                np.random.rand(GRID_NY, GRID_NX).astype(np.float32),
                {
                    "substance": "CO2",
                    "category": "test",
                },
            ),
        },
        coords={
            "lon": regular_grid.lon_range,
            "lat": regular_grid.lat_range,
        },
    )

    ds.to_netcdf(file_path)
    return file_path


@pytest.fixture
def netcdf_file_no_catsub_attrs(tmp_path):
    """Create a NetCDF file without category/substance attributes."""
    file_path = tmp_path / "no_catsub_raster.nc"

    ds = xr.Dataset(
        data_vars={
            "emissions": (
                ["lat", "lon"],
                np.random.rand(GRID_NY, GRID_NX).astype(np.float32),
                {"units": "kg/year/cell"},
            ),
        },
        coords={
            "lon": regular_grid.lon_range,
            "lat": regular_grid.lat_range,
        },
    )

    ds.to_netcdf(file_path)
    return file_path


# Tests for utility functions


def test_unit_scaling_factor_kg_per_m2_per_s():
    """Test unit scaling factor for kg/m2/s."""
    factor, multiply_by_area = get_unit_scaling_factor_to_kg_per_year_per_cell(
        "kg/m2/s"
    )
    assert factor == DAY_PER_YR * SEC_PER_DAY
    assert multiply_by_area is True


def test_unit_scaling_factor_kg_per_year_per_cell():
    """Test unit scaling factor for kg/year/cell variants."""
    for unit in ["kg/y/cell", "kg y-1 cell-1", "kg/year/cell"]:
        factor, multiply_by_area = get_unit_scaling_factor_to_kg_per_year_per_cell(
            unit
        )
        assert factor == 1.0
        assert multiply_by_area is False


def test_unit_scaling_factor_unsupported_unit():
    """Test that unsupported unit raises error."""
    with pytest.raises(NotImplementedError):
        get_unit_scaling_factor_to_kg_per_year_per_cell("unsupported_unit")


def test_get_year_from_attrs_valid():
    """Test year extraction from valid attributes."""
    assert get_year_from_attrs({"year": 2020}) == 2020
    assert get_year_from_attrs({"year": "2021"}) == 2021


def test_get_year_from_attrs_missing():
    """Test year extraction when year is missing."""
    assert get_year_from_attrs({}) is None


def test_get_year_from_attrs_invalid():
    """Test year extraction with invalid values."""
    assert get_year_from_attrs({"year": "invalid"}) is None
    assert get_year_from_attrs({"year": None}) is None


# Tests for NetcdfRaster basic reading


def test_read_simple_netcdf(simple_netcdf_file):
    """Test reading a simple NetCDF file."""
    inv = NetcdfRaster(simple_netcdf_file)

    assert isinstance(inv, Inventory)
    assert inv.year == 2020
    assert "industry" in inv.categories
    assert "CO2" in inv.substances
    assert len(inv.gdf) == GRID_NCELLS


def test_read_multi_variable_netcdf(multi_variable_netcdf_file):
    """Test reading a NetCDF file with multiple variables."""
    inv = NetcdfRaster(multi_variable_netcdf_file)

    assert inv.year == 2021
    assert set(inv.categories) == {"industry", "transport", "agriculture"}
    assert set(inv.substances) == {"CO2", "CH4"}


def test_total_emissions_preserved(tmp_path):
    """Test that total emissions are preserved when reading."""
    file_path = tmp_path / "total_test.nc"

    # Create known emission values
    known_emissions = np.ones((GRID_NY, GRID_NX)) * 100.0  # 100 kg/year/cell everywhere

    ds = xr.Dataset(
        data_vars={
            "CO2_test": (
                ["lat", "lon"],
                known_emissions,
                {
                    "units": "kg/year/cell",
                    "substance": "CO2",
                    "category": "test",
                },
            ),
        },
        coords={
            "lon": regular_grid.lon_range,
            "lat": regular_grid.lat_range,
        },
    )
    ds.to_netcdf(file_path)

    inv = NetcdfRaster(file_path)
    total = inv.gdf[("test", "CO2")].sum()

    # Total should be GRID_NCELLS cells * 100 kg/year/cell
    np.testing.assert_almost_equal(total, GRID_NCELLS * 100.0)


# Tests for variable_to_catsub mapping


def test_custom_variable_mapping(tmp_path):
    """Test reading with custom variable to category/substance mapping."""
    file_path = tmp_path / "custom_mapping.nc"

    ds = xr.Dataset(
        data_vars={
            "my_emissions": (
                ["lat", "lon"],
                np.random.rand(GRID_NY, GRID_NX),
                {"units": "kg/year/cell"},
            ),
        },
        coords={
            "lon": regular_grid.lon_range,
            "lat": regular_grid.lat_range,
        },
    )
    ds.to_netcdf(file_path)

    inv = NetcdfRaster(
        file_path, variable_to_catsub={"my_emissions": ("custom_cat", "CO2")}
    )

    assert "custom_cat" in inv.categories
    assert "CO2" in inv.substances


def test_missing_variable_raises_error(simple_netcdf_file):
    """Test that referencing a missing variable raises an error."""
    with pytest.raises(KeyError):
        NetcdfRaster(
            simple_netcdf_file,
            variable_to_catsub={"nonexistent_var": ("cat", "sub")},
        )


# Tests for unit conversion


def test_unit_override(netcdf_file_no_unit_attr):
    """Test providing unit parameter when attribute is missing."""
    inv = NetcdfRaster(
        netcdf_file_no_unit_attr,
        variable_to_catsub={"emissions": ("test", "CO2")},
        unit="kg/year/cell",
    )

    assert "test" in inv.categories


def test_missing_unit_raises_error(netcdf_file_no_unit_attr):
    """Test that missing unit attribute without override raises an error."""
    with pytest.raises(ValueError, match="Unit for variable"):
        NetcdfRaster(
            netcdf_file_no_unit_attr,
            variable_to_catsub={"emissions": ("test", "CO2")},
        )


# Tests for time dimension handling


def test_single_time_step(netcdf_file_with_time):
    """Test reading file with a single time step."""
    inv = NetcdfRaster(netcdf_file_with_time)

    assert inv.year == 2023
    assert "heating" in inv.categories


def test_multiple_time_steps_requires_profile(netcdf_file_with_monthly_profiles):
    """Test that multiple time steps require a temporal profile."""
    with pytest.raises(ValueError, match="Temporal profile must be provided"):
        NetcdfRaster(netcdf_file_with_monthly_profiles)


def test_monthly_profiles(netcdf_file_with_monthly_profiles):
    """Test reading with monthly temporal profiles."""
    inv = NetcdfRaster(
        netcdf_file_with_monthly_profiles, temporal_profile=MounthsProfile
    )

    assert inv.year == 2023
    assert "seasonal" in inv.categories
    assert hasattr(inv, "t_profiles_groups")


def test_year_override(simple_netcdf_file):
    """Test overriding the year parameter."""
    inv = NetcdfRaster(simple_netcdf_file, year=2025)

    assert inv.year == 2025


# Tests for error handling


def test_no_catsub_no_mapping_raises_error(netcdf_file_no_catsub_attrs):
    """Test that missing category/substance attrs without mapping raises error."""
    with pytest.raises(ValueError, match="variable_to_catsub is None"):
        NetcdfRaster(netcdf_file_no_catsub_attrs)


# Tests for grid handling


def test_grid_created_correctly(simple_netcdf_file):
    """Test that the grid is created correctly from the NetCDF file."""
    inv = NetcdfRaster(simple_netcdf_file)

    assert hasattr(inv, "grid")
    assert isinstance(inv.grid, RegularGrid)
    assert len(inv.grid.gdf) == GRID_NCELLS


# Tests for round-trip (export then import)


def test_export_import_roundtrip():
    """Test exporting an inventory and reading it back."""
    from emiproc.exports.rasters import export_raster_netcdf

    # Prepare the inventory using test_utils objects
    original_inv = inv_with_pnt_sources.copy()
    original_inv.set_crs(regular_grid.crs)

    # Export
    output_file = TEST_OUTPUTS_DIR / "roundtrip_test.nc"
    export_raster_netcdf(original_inv, output_file, regular_grid)

    # Import
    imported_inv = NetcdfRaster(output_file)

    # Compare categories
    assert set(imported_inv.categories) == set(original_inv.categories)

    # Compare substances
    assert set(imported_inv.substances) == set(original_inv.substances)


# Tests for custom coordinate names


def test_custom_coordinate_names(tmp_path):
    """Test reading with custom lat/lon variable names."""
    file_path = tmp_path / "custom_coords.nc"

    ds = xr.Dataset(
        data_vars={
            "CO2_test": (
                ["latitude", "longitude"],
                np.random.rand(GRID_NY, GRID_NX),
                {
                    "units": "kg/year/cell",
                    "substance": "CO2",
                    "category": "test",
                },
            ),
        },
        coords={
            "longitude": regular_grid.lon_range,
            "latitude": regular_grid.lat_range,
        },
    )
    ds.to_netcdf(file_path)

    inv = NetcdfRaster(file_path, lat_name="latitude", lon_name="longitude")

    assert "test" in inv.categories
    assert len(inv.gdf) == GRID_NCELLS


# Tests for year selection with multi-year data


def test_select_specific_year_from_multiyear(tmp_path):
    """Test selecting a specific year from multi-year data."""
    file_path = tmp_path / "multi_year.nc"

    # Create data spanning two years (24 months)
    time_values = pd.date_range("2022-01-15", periods=24, freq="MS")

    emission_data = np.random.rand(24, GRID_NY, GRID_NX).astype(np.float32)

    ds = xr.Dataset(
        data_vars={
            "CO2_multi": (
                ["time", "lat", "lon"],
                emission_data,
                {
                    "units": "kg/year/cell",
                    "substance": "CO2",
                    "category": "multi",
                },
            ),
        },
        coords={
            "lon": regular_grid.lon_range,
            "lat": regular_grid.lat_range,
            "time": time_values,
        },
    )
    ds.to_netcdf(file_path)

    # Select year 2023 (12 months)
    inv = NetcdfRaster(file_path, year=2023, temporal_profile=MounthsProfile)

    assert inv.year == 2023
    assert "multi" in inv.categories


def test_multiple_years_without_selection_raises_error(tmp_path):
    """Test that multiple years without year selection raises an error."""
    file_path = tmp_path / "multi_year_no_select.nc"

    time_values = pd.date_range("2022-01-15", periods=24, freq="MS")

    ds = xr.Dataset(
        data_vars={
            "CO2_multi": (
                ["time", "lat", "lon"],
                np.random.rand(24, GRID_NY, GRID_NX),
                {
                    "units": "kg/year/cell",
                    "substance": "CO2",
                    "category": "multi",
                },
            ),
        },
        coords={
            "lon": regular_grid.lon_range,
            "lat": regular_grid.lat_range,
            "time": time_values,
        },
    )
    ds.to_netcdf(file_path)

    with pytest.raises(ValueError, match="Multiple years found"):
        NetcdfRaster(file_path, temporal_profile=MounthsProfile)


def test_nonexistent_year_raises_error(tmp_path):
    """Test that requesting a nonexistent year raises an error."""
    file_path = tmp_path / "specific_year.nc"

    time_values = pd.date_range("2022-01-15", periods=12, freq="MS")

    ds = xr.Dataset(
        data_vars={
            "CO2_test": (
                ["time", "lat", "lon"],
                np.random.rand(12, GRID_NY, GRID_NX),
                {
                    "units": "kg/year/cell",
                    "substance": "CO2",
                    "category": "test",
                },
            ),
        },
        coords={
            "lon": regular_grid.lon_range,
            "lat": regular_grid.lat_range,
            "time": time_values,
        },
    )
    ds.to_netcdf(file_path)

    with pytest.raises(ValueError, match="No data found for year"):
        NetcdfRaster(file_path, year=2025, temporal_profile=MounthsProfile)


# Tests for temporal profile size mismatches


def test_profile_size_mismatch_raises_error(tmp_path):
    """Test that mismatched profile size raises an error."""
    file_path = tmp_path / "size_mismatch.nc"

    # Create 6 time steps but use monthly profile (12 steps expected)
    time_values = pd.date_range("2023-01-15", periods=6, freq="MS")

    ds = xr.Dataset(
        data_vars={
            "CO2_test": (
                ["time", "lat", "lon"],
                np.random.rand(6, GRID_NY, GRID_NX),
                {
                    "units": "kg/year/cell",
                    "substance": "CO2",
                    "category": "test",
                },
            ),
        },
        coords={
            "lon": regular_grid.lon_range,
            "lat": regular_grid.lat_range,
            "time": time_values,
        },
    )
    ds.to_netcdf(file_path)

    with pytest.raises(ValueError, match="Temporal profile size"):
        NetcdfRaster(file_path, temporal_profile=MounthsProfile)
