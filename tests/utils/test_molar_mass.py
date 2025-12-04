import pytest
from emiproc.utils.constants import get_molar_mass


@pytest.mark.parametrize(
    "substance,expected_mass",
    [
        ("CH4", 16.04),
        ("CO2", 44.009),
        ("N2O", 44.013),
        ("C", 12.01),
    ],
)
def test_molar_mass(substance, expected_mass):
    mm = get_molar_mass(substance)
    assert mm == expected_mass


def test_no_molar_mass_raises():
    with pytest.raises(ValueError):
        get_molar_mass("SOMETHING UNKNOWN")
