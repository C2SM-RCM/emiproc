import xarray as xr


def get_desired_profile_index(
    profiles_indexes: xr.DataArray,
    cell: int | None = None,
    cat: str | None = None,
    sub: str | None = None,
) -> int:
    """Return the index of the desired profile.

    Smart function allowing to select based on desired attributes.
    It will check that the profile can be extracted.
    """

    # First check that the user did not omit a required dimension
    dims = profiles_indexes.dims
    if cell is None and "cell" in dims:
        raise ValueError("cell must be specified, as each cell has a specific profile.")
    if cat is None and "category" in dims:
        raise ValueError(
            "category must be specified, as each category has a specific profile."
        )
    if sub is None and "substance" in dims:
        raise ValueError(
            "substance must be specified, as each substance has a specific profile."
        )
    
    access_dict = {}

    # Add to the access the dimension specified, 
    # If a dimension is specified but not in the dims, it means 
    # we don't care becausse it is the same for all the dimension cooridnates
    if cell is not None and "cell" in dims:
        if cell not in profiles_indexes.coords["cell"]:
            raise ValueError(
                f"cell {cell} is not in the profiles indexes, "
                f"got {profiles_indexes.coords['cell']}"
            )
        access_dict["cell"] = cell
    if cat is not None and "category" in dims:
        if cat not in profiles_indexes.coords["category"]:
            raise ValueError(
                f"category {cat} is not in the profiles indexes, "
                f"got {profiles_indexes.coords['category']}"
            )
        access_dict["category"] = cat
    if sub is not None and "substance" in dims:
        if sub not in profiles_indexes.coords["substance"]:
            raise ValueError(
                f"substance {sub} is not in the profiles indexes, "
                f"got {profiles_indexes.coords['substance']}"
            )
        access_dict["substance"] = sub
    
    # Access the xarray 
    desired_index = profiles_indexes.sel(**access_dict)

    # Check the the seleciton is just a single value
    if desired_index.size != 1:
        raise ValueError(
            f"More than one profile matches the selection: {desired_index}, got {desired_index.size =}"
        )

    # Return the index as int
    return int(desired_index.values)
