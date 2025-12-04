"""Units in emiproc are always kg/year/cell unless otherwise specified.

Cell or shape depending on gridded emissions or shapefile based emissions.
"""

from emiproc.utils.constants import get_molar_mass, DAY_PER_YR, SEC_PER_DAY


def get_scaling_factor_to_emiproc(
    unit: str, substance: str | None = None
) -> tuple[float, bool]:
    """Get the scaling factor to convert from the given unit to kg/year/cell.

    Supported units:
    - "kg/m2/s"

    :param unit: Unit string.

    :return: Scaling factor. and a boolean indicating that we need to scale (multiply) with the cell area.
    """
    if unit == "kg/m2/s":
        # kg/m2/s * day/year * s/day * m2/cell = kg/year/cell
        return DAY_PER_YR * SEC_PER_DAY, True  # seconds to year
    elif unit == "kg/year/m2":
        # kg/year/m2 * m2/cell = kg/year/cell
        return 1.0, True
    elif unit in ["kg/y/cell", "kg y-1 cell-1", "kg/year/cell"]:
        return 1.0, False
    elif unit == "PgC/yr":
        # Carbon to CO2 conversion
        if substance != "CO2":
            raise ValueError("PgC/yr unit can only be used for CO2 substance.")
        return 1e12 * (get_molar_mass("CO2") / get_molar_mass("C")), False
    elif unit == "micromol/m2/s":
        molar_mass = get_molar_mass(substance)  # g/mol
        # micromol/m2/s * kg/g * g/mol * mol/micromol * s/year * m2/cell
        return 1e-3 * molar_mass * 1e-6 * SEC_PER_DAY * DAY_PER_YR, True
    else:
        raise NotImplementedError(
            f"Unit {unit} not supported. "
            "Please implement in "
            "emiproc.utils.units.get_scaling_factor_to_emiproc."
        )
