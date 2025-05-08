"""Test scaling of the emissions."""

from emiproc.tests_utils.test_inventories import inv, inv_with_pnt_sources


from emiproc.inventories.utils import scale_inventory


scaled_inv = scale_inventory(inv, {"NH3": {"test": 2.42}})


def test_total_emissions_values():

    assert all(scaled_inv.gdf[("test", "NH3")] == 2.42 * inv.gdf[("test", "NH3")])
    assert all(scaled_inv.gdf[("liku", "CO2")] == inv.gdf[("liku", "CO2")])


scaled_inv_pntsrc = scale_inventory(
    inv_with_pnt_sources, {"CO2": {"liku": 1.42}, "NH3": {"test": 2.42}}
)


def test_total_emissions_values_pntsrc():

    assert all(
        scaled_inv_pntsrc.gdf[("test", "NH3")]
        == 2.42 * inv_with_pnt_sources.gdf[("test", "NH3")]
    )
    assert all(
        scaled_inv_pntsrc.gdf[("liku", "CO2")]
        == 1.42 * inv_with_pnt_sources.gdf[("liku", "CO2")]
    )
    assert all(
        scaled_inv_pntsrc.gdf[("adf", "CO2")]
        == inv_with_pnt_sources.gdf[("adf", "CO2")]
    )
    assert all(
        scaled_inv_pntsrc.gdfs["liku"]["CO2"]
        == 1.42 * inv_with_pnt_sources.gdfs["liku"]["CO2"]
    )


def test_scale_with_float():

    scaling_factor = 2.42
    scaled_inv = scale_inventory(inv, scaling_factor)

    assert scaled_inv.total_emissions.equals(inv.total_emissions * scaling_factor)
