"""Units in emiproc are always kg/year/cell unless otherwise specified.

Cell or shape depending on gridded emissions or shapefile based emissions.
"""

from __future__ import annotations

from emiproc.utils.constants import HOUR_PER_YR, get_molar_mass, DAY_PER_YR, SEC_PER_DAY

_REPLACE = {
    "kgr": "kg",
    "year": "y",
    "yr": "y",
    "m-2": "/m2",
    "y-1": "/y",
    "h-1": "/h",
    "s-1": "/s",
    "cell-1": "/cell",
    " ": "",
}


def get_scaling_factor_to_emiproc(
    unit: str, substance: str | None = None
) -> tuple[float, bool]:
    """Get the scaling factor to convert from the given unit to kg/year/cell.

    Supported units:
    - "kg/m2/s"

    :param unit: Unit string.

    :return: Scaling factor. and a boolean indicating that we need to scale (multiply) with the cell area.
    """
    unit_original = unit
    for old, new in _REPLACE.items():
        unit = unit.replace(old, new)
    if unit == "kg/m2/s":
        # kg/m2/s * day/year * s/day * m2/cell = kg/year/cell
        return DAY_PER_YR * SEC_PER_DAY, True  # seconds to year
    elif unit == "kg/m2/h":
        # kg/m2/h * h/year * m2/cell = kg/year/cell
        return HOUR_PER_YR, True  # hours to year
    elif unit == "kg/y/m2":
        # kg/year/m2 * m2/cell = kg/year/cell
        return 1.0, True
    elif unit == "kg/y/cell":
        return 1.0, False
    elif unit == "PgC/y":
        # Carbon to CO2 conversion
        if substance != "CO2":
            raise ValueError("PgC/y unit can only be used for CO2 substance.")
        return 1e12 * (get_molar_mass("CO2") / get_molar_mass("C")), False
    elif unit == "micromol/m2/s":
        molar_mass = get_molar_mass(substance)  # g/mol
        # micromol/m2/s * kg/g * g/mol * mol/micromol * s/year * m2/cell
        return 1e-3 * molar_mass * 1e-6 * SEC_PER_DAY * DAY_PER_YR, True
    else:
        raise NotImplementedError(
            f"Unit `{unit_original}`, `{unit}` after cleaning not supported. "
            "Please implement in "
            "emiproc.utils.units.get_scaling_factor_to_emiproc."
        )
