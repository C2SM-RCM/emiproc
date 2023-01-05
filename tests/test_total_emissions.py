"""Test the calucation of total emissions."""
from emiproc.tests_utils.test_inventories import inv, inv_with_pnt_sources


from emiproc.inventories.utils import get_total_emissions


total_emissions = get_total_emissions(inv)


def test_total_emissions_values():

    assert total_emissions["NH3"]["test"] == 15
    assert total_emissions["NH3"]["__total__"] == 15


def test_categories_with_no_emissions_are_not_in_dict():
    assert "adf" not in total_emissions["NH3"]


total_emissions2 = get_total_emissions(inv_with_pnt_sources)


def test_total_emissions_values():

    assert total_emissions2["AITS"]["other"] == 3
    assert total_emissions2["NH3"]["test"] == 15
    assert total_emissions2["NH3"]["__total__"] == 15
    assert total_emissions2["CO2"]["blek"] == 6
    assert total_emissions2["CO2"]["__total__"] == 6 + 10 + 3 + 10


def test_categories_with_no_emissions_are_not_in_dict():
    assert "adf" not in total_emissions2["NH3"]
