"""Test the speciation module."""

from emiproc.inventories.utils import drop
from emiproc.tests_utils.test_inventories import inv, inv_with_pnt_sources


def test_inv_with_gdfs():

    categories = ["adf", "blek", "liku"]


    inv_dropped = drop(
        inv_with_pnt_sources, substances=["CO2"], categories=categories
    )


    # Check the the categories are dropped
    assert all([cat not in inv_dropped.categories for cat in categories])
    # Was kept
    assert "other" in inv_dropped.categories

    assert "CO2" not in inv_dropped.substances
    assert "NH3" in inv_dropped.substances
    assert "AITS" in inv_dropped.substances


def test_inv_drop_cat():

    categories = ["adf"]

    inv_dropped = drop(
        inv, categories=categories
    )

    assert "adf" not in inv_dropped.categories
    assert "liku" in inv_dropped.categories
    # Check we did not change the original inventory
    assert "adf" in inv.categories


def test_inv_drop_sub():

    subtances = ["CO2"]

    inv_dropped = drop(
        inv, substances=subtances
    )

    assert "CO2" not in inv_dropped.substances
    assert "NH3" in inv_dropped.substances
    # Check we did not change the original inventory
    assert "CO2" in inv.substances


def test_keep_instead_of_drop():
    subtances = ["CO2"]

    inv_dropped = drop(
        inv, substances=subtances, keep_instead_of_drop=True
    )

    assert "CO2" in inv_dropped.substances
    assert "NH3" not in inv_dropped.substances
    # Check we did not change the original inventory
    assert "CO2" in inv.substances
