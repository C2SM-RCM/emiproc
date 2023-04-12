"""Function for doing speciation.

The function speciate_inventory() is the main function.
Other functions are hardcoded for several substances.
"""
from __future__ import annotations
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from emiproc.inventories import Inventory, CatSub, Category


def speciate_inventory(
    inv: Inventory,
    speciation_dict: dict[CatSub, dict[CatSub, float]],
    drop: bool = True,
) -> Inventory:
    """Speciate an inventory.

    Speciation is splitting one substance into several substances.
    For example NOx can be split into NO and NO2.
    Replaces the current category/substance with the new one.

    :arg inv: The inventory to speciate.
    :arg speciation_dict: A dict with the speciation rules.
        The keys are the category/substance to speciate.
        The values is a dict mapping new category/substance and the fraction
        of the orignal substance to be specified.
        Ratio is the weight fraction conversion from the speciated substance.
        Note that the ratio don't need to sum to 1, depending on the
        chemical parameters.
    :arg drop: Whether to drop the speciated category/substance.

    :returns: The speciated inventory.
    """
    new_inv = inv.copy()

    for cat_sub, new_species in speciation_dict.items():
        cat, sub = cat_sub
        # Check the there is a substance to speciate
        if (
            cat_sub not in inv._gdf_columns
            and cat not in inv.gdfs
            and sub not in inv.gdfs[cat]
        ):
            raise KeyError(f"Cannot speciate: {cat_sub} not in {inv}")

        # Speciate the gdf
        if cat_sub in inv._gdf_columns:
            for new_cat_sub, speciation_ratio in new_species.items():
                # if the new cat/sub is already in the gdf raise an error
                if new_cat_sub in inv._gdf_columns:
                    raise KeyError(f"{new_cat_sub} already in the gdf of {inv}")
                new_inv.gdf[new_cat_sub] = inv.gdf[cat_sub] * speciation_ratio
            if drop:
                new_inv.gdf.drop(columns=cat_sub, inplace=True)
        # Speciate the gdfs
        if cat in inv.gdfs and sub in inv.gdfs[cat].columns:
            for new_cat_sub, speciation_ratio in new_species.items():
                new_cat, new_sub = new_cat_sub
                # if the new cat/sub is already in the gdf raise an error
                if new_cat in inv.gdfs and new_sub in inv.gdfs[new_cat].columns:
                    raise KeyError(
                        f"Cannot speciate: {new_cat_sub} already in the gdfs of {inv}"
                    )
                new_inv.gdfs[new_cat][new_sub] = inv.gdfs[cat][sub] * speciation_ratio
            if drop:
                new_inv.gdfs[cat].drop(columns=sub, inplace=True)

    new_inv.history.append(f"Speciated with {speciation_dict}.")

    return new_inv


def speciate_nox(
    inv: Inventory,
    NOX_TO_NO2: float | dict[Category, float] = 0.18,
    drop: bool = True,
) -> Inventory:
    """Speciate NOx into NO and NO2.

    :arg inv: The inventory to speciate.
    :arg drop: Whether to drop the speciated category/substance.
    :arg NOX_TO_NO2: The fraction of NOx that is speciated to NO2.
        It is possible to use a dict with a different fraction for each category.

        .. note:: 

            Depending on the sector, this value can vary.
           
            For most emission sources, the fraction of NO is closer to 95%,
            only for traffic a fraction of 82% may be applied.
            The reason is that oxidation catalysts in diesel engines partly
            oxidize NO to NO2 before it is emitted through the tailpipe.
            In the first decade of 2000,
            the fraction of NO gradually decreased from 95% to about 80%.

            See more: 

            * https://www.empa.ch/documents/56101/246436/Trend+NO2+Immissionen+Stadt+2022/ddba8b88-c599-4ed4-8b94-cc24670be683
            * https://www.zh.ch/de/umwelt-tiere/luft-strahlung/luftschadstoffquellen/emissionen-verkehr/abgasmessungen-rsd.html



    :returns: The speciated inventory.

    """
    MOLAR_MASS_NO2 = 46.0
    MOLAR_MASS_NO = 30.0
    MM_RATIO = MOLAR_MASS_NO / MOLAR_MASS_NO2

    if isinstance(NOX_TO_NO2, dict):
        speciation_dict = {
            (cat, "NOx"): {
                (cat, "NO"): (1.0 - ratio) * MM_RATIO,
                (cat, "NO2"): ratio,
            }
            for cat, ratio in NOX_TO_NO2.items()
        }
    elif isinstance(NOX_TO_NO2, float):
        # Make sure the fraction is between 0 and 1
        if NOX_TO_NO2 < 0 or NOX_TO_NO2 > 1:
            raise ValueError(f"NOX_TO_NO2 must be between 0 and 1, not {NOX_TO_NO2}.")
        # Aplly the same fraction to all categories
        speciation_dict = {
            (cat, "NOx"): {
                (cat, "NO"): (1.0 - NOX_TO_NO2) * MM_RATIO,
                (cat, "NO2"): NOX_TO_NO2,
            }
            for cat in inv.categories
            if (cat, "NOx") in inv._gdf_columns
            or (cat in inv.gdfs and "NOx" in inv.gdfs[cat].columns)
        }
    else:
        raise TypeError(f"NOX_TO_NO2 must be a float or dict, not {type(NOX_TO_NO2)}.")

    return speciate_inventory(inv, speciation_dict, drop=drop)
