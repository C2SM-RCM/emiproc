import pytest
from emiproc.utils.constants import get_molar_mass


def test_molar_mass():

    mm_ch4 = get_molar_mass("CH4")

    assert mm_ch4 == 16.04


def test_no_molar_mass_raises():

    with pytest.raises(ValueError):
        get_molar_mass("SOMETHING UNKNOWN")
