"""Test the speciation module."""
from shapely.geometry import Point
from emiproc.inventories import Inventory
from emiproc.speciation import speciate_nox, speciate_inventory
from emiproc.tests_utils.test_inventories import inv, inv_with_pnt_sources
import numpy as np
import geopandas as gpd

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
    assert np.allclose(s_inv.gdf[("cat1", "NO2")],  0.18 * gdf[("cat1", "NOx")])
    # Is a more complex than just a ratio
    assert np.all(s_inv.gdf[("cat1", "NO")] !=  0.82 * gdf[("cat1", "NOx")])

    # Test with a dict
    s_inv = speciate_nox(inv, NOX_TO_NO2={"cat1": 0.5, "cat2": 0.2})
    assert np.allclose(s_inv.gdf[("cat1", "NO2")],  0.5 * gdf[("cat1", "NOx")])
    assert np.allclose(s_inv.gdf[("cat2", "NO2")],  0.2 * gdf[("cat2", "NOx")])


