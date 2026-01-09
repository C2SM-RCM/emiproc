import pytest
from emiproc.utils.units import get_scaling_factor_to_emiproc, get_molar_mass

from emiproc.utilities import DAY_PER_YR, SEC_PER_DAY


def test_unit_scaling_factor_kg_per_m2_per_s():
    """Test unit scaling factor for kg/m2/s."""
    factor, multiply_by_area = get_scaling_factor_to_emiproc("kg/m2/s")
    assert factor == DAY_PER_YR * SEC_PER_DAY
    assert multiply_by_area is True


def test_unit_scaling_factor_kg_per_year_per_cell():
    """Test unit scaling factor for kg/year/cell variants."""
    for unit in ["kg/y/cell", "kg y-1 cell-1", "kg/year/cell"]:
        factor, multiply_by_area = get_scaling_factor_to_emiproc(unit)
        assert factor == 1.0
        assert multiply_by_area is False


def test_carbon_unit_only_for_co2():
    """Test that PgC/yr unit only works for CO2 substance."""
    with pytest.raises(ValueError):
        get_scaling_factor_to_emiproc("PgC/yr", substance="CH4")
    factor, multiply_by_area = get_scaling_factor_to_emiproc("PgC/yr", substance="CO2")
    assert factor == 1e12 * (get_molar_mass("CO2") / get_molar_mass("C"))
    assert multiply_by_area is False


def test_micromol_per_m2_per_s():
    """Test unit scaling factor for micromol/m2/s."""
    factor, multiply_by_area = get_scaling_factor_to_emiproc(
        "micromol/m2/s", substance="CH4"
    )
    molar_mass = get_molar_mass("CH4")  # g/mol for CH4
    expected_factor = 1e-3 * molar_mass * 1e-6 * SEC_PER_DAY * DAY_PER_YR
    assert factor == expected_factor
    assert multiply_by_area is True


def test_unit_scaling_factor_unsupported_unit():
    """Test that unsupported unit raises error."""
    with pytest.raises(NotImplementedError):
        get_scaling_factor_to_emiproc("unsupported_unit")
