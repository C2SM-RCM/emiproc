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
    "kgch4": "kg",
}


def get_scaling_factor_to_emiproc(
    unit: str, substance: str | None = None
) -> tuple[float, bool]:
    """Get the scaling factor to convert from the given unit to kg/year/cell.

    Supported units are of the form: weight / time [/ area]

    They can be specified in different ways, e.g. "kg/m2/s", "kg cell-2 year-1". 
    
    :param unit: Unit string.
    :param substance: Substance string, e.g. "CO2", "CH4".
        Required for units based on moles.

    :return: A tuple containing
    
        * Scaling factor
        * A boolean indicating that scaling with the cell area is needed

    """
    unit_original = unit

    unit = unit.lower().replace(" ", "")
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
    elif unit == "pgc/y":
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
