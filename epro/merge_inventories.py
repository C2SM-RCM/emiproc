#!/usr/bin/env python
# coding: utf-8
"""
Merge netCDF files.

The variables to be combined need to have identical dimensions. All
dimensions and attributes are copied from the base inventory.
"""
import os

import numpy as np

from netCDF4 import Dataset

from country_code import country_codes
from nc_operations import copy_dataset, VariableCreator


def merge_inventories(base_inv, nested_invs, output_path):
    """Merge nested inventories into a base inventory.

    Copy the base inventory to output_path.
    Then, for every variable in nested_invs key, overwrite the cells matching
    the country code (value of nested_invs) with values from nested_invs.

    Variables missing in base_inv are created with zeros everywhere and then
    overwritten where the country code matches.

    Dimensions and country_ids have to be identical for all inventories.

    Parameters
    ----------
    base_inv : str
        Path to the base inventory
    nested_invs : dict(str: str)
        Dictionary matching paths to nested inventories (key) to EMEP country
        codes (values). Overwrite all gridcells in the base inventory matching
        this country code with values from the nested inventory.
    output_path : str
        Path to the created file. If necessary create the containing directory.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with Dataset(output_path, "w") as out_dataset:
        with Dataset(base) as base_dataset:
            copy_dataset(base_dataset, out_dataset)

        for nest_path, nest_code in nested_invs.items():
            mask = out_dataset["country_ids"][:] == country_codes[nest_code]

            with Dataset(nest_path) as nest_dataset:
                assert np.array_equal(
                    nest_dataset["country_ids"][:],
                    out_dataset["country_ids"][:],
                ), "Country code maps have to match"
                for variable_name in nest_dataset.variables:
                    if len(nest_dataset[variable_name].dimensions) == 2:
                        if variable_name not in out_dataset.variables:
                            VariableCreator.from_existing_var(
                                src_variable=nest_dataset[variable_name],
                                var_vals=0,
                            ).apply_to(out_dataset)

                        # "reference" to variables array so that the values
                        # written by copyto are persistent
                        out_vals = out_dataset[variable_name][:]
                        np.copyto(
                            out_vals, nest_dataset[variable_name][:], where=mask
                        )
                        out_dataset[variable_name][:] = out_vals


if __name__ == "__main__":
    merge_inventories(base, nest, output)
