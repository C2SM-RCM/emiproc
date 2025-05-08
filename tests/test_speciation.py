"""Test the speciation module."""

from os import PathLike
from pathlib import Path

import xarray as xr

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from emiproc import TESTS_DIR
from emiproc.inventories import Inventory
from emiproc.profiles.vertical_profiles import VerticalProfile, VerticalProfiles
from emiproc.speciation import (
    read_speciation_table,
    speciate,
    speciate_inventory,
    speciate_nox,
    merge_substances,
)
from emiproc.tests_utils.test_inventories import inv, inv_with_pnt_sources
from emiproc.tests_utils.african_case import (
    african_inv_emissions_only_land,
    african_inv,
)


@pytest.mark.parametrize(
    "table_path",
    [
        TESTS_DIR / "speciation" / "test_table_NOx.csv",
        TESTS_DIR / "speciation" / "table_test_inv_CO2.csv",
    ],
)
def test_read_speciation_table(table_path: PathLike):
    read_speciation_table(table_path)


def test_load_wrong_table():
    with pytest.raises(ValueError):
        read_speciation_table(TESTS_DIR / "speciation" / "wrong_ratio_table.csv")


def test_load_wrong_no_checks():
    read_speciation_table(
        TESTS_DIR / "speciation" / "wrong_ratio_table.csv", check_sum=False
    )


def test_speciation_wrong_ratios():
    da_speciation = read_speciation_table(
        TESTS_DIR / "speciation" / "wrong_ratio_table.csv", check_sum=False
    )
    inv_speciated = speciate(
        inv_with_pnt_sources, substance="CO2", speciation_ratios=da_speciation
    )

    # Check that the total emissions are the same
    # Ratios for adf in the file are {CO2_ANT: 0.5, CO2_BIO: 0.4}
    assert (
        inv_speciated.total_emissions.loc["CO2_ANT", "adf"]
        == 0.5 * inv_with_pnt_sources.total_emissions.loc["CO2", "adf"]
    )
    assert (
        inv_speciated.total_emissions.loc["CO2_BIO", "adf"]
        == 0.4 * inv_with_pnt_sources.total_emissions.loc["CO2", "adf"]
    )


def test_speciation_african_case():
    da_africa = read_speciation_table(
        TESTS_DIR / "speciation" / "table_africa_testcase.csv"
    )

    speciated = speciate(
        african_inv_emissions_only_land, substance="CO2", speciation_ratios=da_africa
    )

    # Test that the total emissions are the same
    pd.testing.assert_series_equal(
        speciated.total_emissions.loc[["CO2_ANT", "CO2_BIO"], :]
        .sum(axis="index")
        .sort_index(),
        african_inv_emissions_only_land.total_emissions.loc["CO2", :].sort_index(),
        check_names=False,
    )


def test_fail_on_emissions_on_no_country():
    """By default, the speciation should fail if there are emissions in cells with no country (eg on ocean)."""
    da_africa = read_speciation_table(
        TESTS_DIR / "speciation" / "table_africa_testcase.csv"
    )

    with pytest.raises(ValueError):
        speciate(
            african_inv,
            substance="CO2",
            speciation_ratios=da_africa,
        )


def test_use_default_when_country_missing():
    """Use a custom default value in the file when there are emissions in cells with no country (eg on ocean).

    The country is -99 in the file
    """
    da_africa = read_speciation_table(
        TESTS_DIR / "speciation" / "table_africa_testcase_with_missing.csv"
    )

    speciated = speciate(african_inv, substance="CO2", speciation_ratios=da_africa)

    # Test that the total emissions are the same
    pd.testing.assert_series_equal(
        speciated.total_emissions.loc[["CO2_ANT", "CO2_BIO"], :]
        .sum(axis="index")
        .sort_index(),
        african_inv.total_emissions.loc["CO2", :].sort_index(),
        check_names=False,
    )


def test_speciate_simple_inventory():
    da_speciation = read_speciation_table(
        TESTS_DIR / "speciation" / "table_test_inv_CO2.csv"
    )

    inv_speciated = speciate(
        inv_with_pnt_sources, substance="CO2", speciation_ratios=da_speciation
    )

    # Check that the total emissions are the same
    pd.testing.assert_series_equal(
        inv_speciated.total_emissions.loc[["CO2_ANT", "CO2_BIO"], :]
        .sum(axis="index")
        .sort_index(),
        inv_with_pnt_sources.total_emissions.loc["CO2", :].sort_index().fillna(0),
        check_names=False,
    )


def test_speciate_when_profiles_are_there():
    da_speciation = read_speciation_table(
        TESTS_DIR / "speciation" / "table_test_inv_CO2.csv"
    )
    inv = inv_with_pnt_sources.copy()

    inv.set_profile(
        VerticalProfile(ratios=np.array([1.0, 0.0]), height=np.array([1.0, 2.0])),
        category="liku",
        substance="CO2",
    )
    inv_speciated = speciate(inv, substance="CO2", speciation_ratios=da_speciation)
    new_subs = ["CO2_ANT", "CO2_BIO"]
    for sub in new_subs:
        assert sub in inv_speciated.substances

        # Check in the profiles are there
        assert sub in inv_speciated.v_profiles_indexes.coords["substance"].values


def test_speciate_inventory():
    # create a test speciation dictionary
    speciation_dict = {
        ("adf", "CH4"): {
            ("adf", "14CH4"): 0.9,
            ("adf", "12CH4"): 0.1,
        },
        ("liku", "CO2"): {
            ("liku", "14CO2"): 0.5,
            ("liku", "12CO2"): 0.2,
        },
    }
    sp_inv = speciate_inventory(inv_with_pnt_sources, speciation_dict)

    # check that the speciation worked
    assert np.allclose(
        sp_inv.gdf[("adf", "14CH4")], 0.9 * inv_with_pnt_sources.gdf[("adf", "CH4")]
    )
    assert np.allclose(
        sp_inv.gdf[("adf", "12CH4")], 0.1 * inv_with_pnt_sources.gdf[("adf", "CH4")]
    )
    assert np.allclose(
        sp_inv.gdf[("liku", "14CO2")], 0.5 * inv_with_pnt_sources.gdf[("liku", "CO2")]
    )
    assert np.allclose(
        sp_inv.gdf[("liku", "12CO2")], 0.2 * inv_with_pnt_sources.gdf[("liku", "CO2")]
    )
    # check that the original substance is dropped
    assert ("adf", "CH4") not in sp_inv.gdf.columns
    assert ("liku", "CO2") not in sp_inv.gdf.columns
    # check also the point sources
    assert np.allclose(
        sp_inv.gdfs["liku"]["14CO2"],
        0.5 * inv_with_pnt_sources.gdfs["liku"]["CO2"],
    )
    assert np.allclose(
        sp_inv.gdfs["liku"]["12CO2"],
        0.2 * inv_with_pnt_sources.gdfs["liku"]["CO2"],
    )


def test_speciate_inventory_with_profiles():
    # create a test speciation dictionary
    speciation_dict = {
        ("adf", "CH4"): {
            ("adf", "14CH4"): 0.9,
            ("adf", "12CH4"): 0.1,
        },
        ("liku", "CO2"): {
            ("liku", "14CO2"): 0.5,
            ("liku", "12CO2"): 0.2,
        },
    }

    inv = inv_with_pnt_sources.copy()
    inv.set_profile(
        VerticalProfile(ratios=np.array([1.0, 0.0]), height=np.array([1.0, 2.0])),
        category="liku",
        substance="CO2",
    )
    inv.set_profile(
        VerticalProfile(ratios=np.array([0.0, 1.0]), height=np.array([1.0, 2.0])),
        substance="NH3",
    )
    # This is what the indexes should be after setting the profile
    assert inv.v_profiles_indexes.sel(substance="CO2", category="liku").values != -1
    assert inv.v_profiles_indexes.sel(substance="CH4", category="adf").values == -1
    assert inv.v_profiles_indexes.sel(substance="CO2", category="adf").values == -1
    assert inv.v_profiles_indexes.sel(substance="CH4", category="liku").values == -1

    sp_inv = speciate_inventory(inv, speciation_dict)

    assert np.all(
        sp_inv.v_profiles[
            sp_inv.v_profiles_indexes.sel(substance="14CO2", category="liku")
        ].ratios
        == np.array([1.0, 0.0])
    )
    assert np.all(
        sp_inv.v_profiles[
            sp_inv.v_profiles_indexes.sel(substance="12CO2", category="liku")
        ].ratios
        == np.array([1.0, 0.0])
    )

    # Should not be found
    assert sp_inv.v_profiles_indexes.sel(substance="12CO2", category="adf").values == -1

    # NH3 should still be there
    assert sp_inv.v_profiles_indexes.sel(substance="NH3", category="test").values != -1


def test_speciate_nox():
    # create a test inventory containg NOx and 2 test categories
    gdf = gpd.GeoDataFrame(
        {
            ("cat1", "NOx"): np.array([1, 2]),
            ("cat2", "NOx"): np.array([3, 4]),
        },
        geometry=[Point(0, 0), Point(1, 1)],
    )

    inv = Inventory.from_gdf(gdf)
    s_inv = speciate_nox(inv)
    # check that the speciation worked
    assert np.allclose(s_inv.gdf[("cat1", "NO2")], 0.18 * gdf[("cat1", "NOx")])
    # Is a more complex than just a ratio
    assert np.all(s_inv.gdf[("cat1", "NO")] != 0.82 * gdf[("cat1", "NOx")])

    # Test with a dict
    s_inv = speciate_nox(inv, NOX_TO_NO2={"cat1": 0.5, "cat2": 0.2})
    assert np.allclose(s_inv.gdf[("cat1", "NO2")], 0.5 * gdf[("cat1", "NOx")])
    assert np.allclose(s_inv.gdf[("cat2", "NO2")], 0.2 * gdf[("cat2", "NOx")])


def test_simple():

    categories = ["adf", "blek", "liku"]
    da_speciation = xr.DataArray(
        np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]),
        coords={
            "substance": ["CO2_ANT", "CO2_BIO"],
            "speciation": range(len(categories)),
            "category": ("speciation", categories),
        },
        dims=["substance", "speciation"],
    )

    inv_speciated = speciate(
        inv_with_pnt_sources, substance="CO2", speciation_ratios=da_speciation
    )


def test_no_speciation_in_ratios():

    da_speciation = xr.DataArray(
        np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]),
        coords=[["CO2_ANT", "CO2_BIO"], ["A", "B", "C"]],
        dims=["substance", "category"],
    )
    with pytest.raises(ValueError):
        inv_speciated = speciate(
            inv_with_pnt_sources, substance="CO2", speciation_ratios=da_speciation
        )


def test_merge_substances():

    merged = merge_substances(inv_with_pnt_sources, substances={"GHG": ["CO2", "CH4"]})

    assert "GHG" in merged.substances
    assert "CO2" not in merged.substances
    assert "CH4" not in merged.substances

    # Check that the total emissions are the same
    merged_emissions = merged.total_emissions.loc["GHG"].fillna(0)
    previous_emissions = inv_with_pnt_sources.total_emissions.loc[["CO2", "CH4"]].sum(
        axis="index"
    )
    pd.testing.assert_series_equal(
        merged_emissions.sort_index(),
        previous_emissions.sort_index(),
        check_names=False,
    )


def test_merge_substances_no_drop():

    merged = merge_substances(
        inv_with_pnt_sources, substances={"GHG": ["CO2", "CH4"]}, drop=False
    )

    assert "GHG" in merged.substances
    # Substances should not be dropped
    assert "CO2" in merged.substances
    assert "CH4" in merged.substances


def test_merge_substances_use_as_rename():

    merged = merge_substances(inv_with_pnt_sources, substances={"co2": ["CO2"]})

    # Check that the total emissions are the same
    merged_emissions = merged.total_emissions.loc["co2"]
    previous_emissions = inv_with_pnt_sources.total_emissions.loc["CO2"]
    pd.testing.assert_series_equal(
        merged_emissions.sort_index(),
        previous_emissions.sort_index(),
        check_names=False,
    )


def test_cannot_merge_using_new_substances():

    pytest.raises(
        KeyError,
        merge_substances,
        inv_with_pnt_sources,
        substances={"GHG": ["CO2", "CH4"], "GHG2": ["GHG"]},
    )

    pytest.raises(
        ValueError,
        merge_substances,
        inv_with_pnt_sources,
        substances={"CO2": ["CO2", "CH4"], "GHG2": ["CO2"]},
    )
