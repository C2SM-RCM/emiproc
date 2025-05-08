# Molar mass in g / mol
MOLAR_MASSES_ = {
    "CH4": 16.04,
    "CO2": 44.01,
    # This is a test value
    "test": 1.0,
    "test2": 2.0,
}


def get_molar_mass(substance: str) -> float:
    """Get the molar mass of a substance in g/mol."""
    if substance not in MOLAR_MASSES_:
        raise ValueError(
            f"Unknown molar mass for substance `{substance}`."
            f"Please add it to the MOLAR_MASSES_ dictionary in {__file__}."
        )
    return MOLAR_MASSES_[substance]
