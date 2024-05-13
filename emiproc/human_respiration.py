"""Emiproc provides tools to help generate human respiration emissions.



"""
from __future__ import annotations

from os import PathLike
import geopandas as gpd
from emiproc.grids import RegularGrid
from emiproc.inventories import Inventory, Category, Substance, CatSub
from emiproc.utilities import DAY_PER_YR


class EmissionFactor:
    """Emissions factors are how much of a compound is produced per individual per day.

    Units are kg/day (kg of the compounds)
    """

    # Very simple estimation for aproximate results
    ROUGH_ESTIMATON = 1.0
    CO2_ROUGH_ESTIMATON = 1.0

    # 2.1 g / year / person 
    # https://doi.org/10.1016/S0048-9697(97)00267-2
    N2O_MITSUI_ET_ALL = 2.1e-3 / DAY_PER_YR

    # Breath: 2.3 mmol / day / person 
    # Flatus: 7.0 mmol / day / person 
    # https://doi.org/10.1016/j.atmosenv.2019.116823
    # Multiply by the molar mass of CH4 and correct units
    CH4_POLAG_KEPPLER = (2.3 + 7.0) * 1e-3 * 16e-3 


def load_data_from_quartieranalyse(
    file: PathLike,
    grid: RegularGrid | None = None,
) -> gpd.GeoDataFrame:
    """Load the data required from the quartieranalyse file from zurich.

    :arg file: The Path to the Quartier data file.
    :arg grid: Optionally the grid of the simulation. This helps to
        load the data.

    :return: A geodataframe containing columns:
        geometry: The shapes of the different zones
        people_working: The number of people who work in these zones
        people_living: The number of people who live in these zones

    """
    kwargs = {}
    if grid:
        kwargs["bbox"] = grid.bounds

    gdf = gpd.read_file(file, **kwargs)
    gdf_interest = gdf[["U_EINW", "U_BESCH", "geometry"]]

    # Replace missing values
    gdf_interest = gdf_interest.replace(-999.0, 0)

    gdf_interest = gdf_interest.rename(
        columns={
            "U_BESCH": "people_working",
            "U_EINW": "people_living",
        }
    )

    return gdf_interest


def people_to_emissions(
    people_gdf: gpd.GeoDataFrame,
    time_ratios: dict[Category, float],
    emission_factor:  dict[CatSub, float] | float = EmissionFactor.ROUGH_ESTIMATON,
    output_gdfs: bool = False,
    name: str = "human_respiration",
    substance: Substance = "CO2",
) -> Inventory:
    """Get human respiration emissions.

    Convert the number of people living in different areas
    to annual emission of CO2 from human respiration.

    Different categories can be provided in the input. (ex. working, living, etc.)

    The formula used is quite simple:

    .. math::
        \\text{emissions} [kg/y/shape] =
        \\text{emission factor} [kg/p/d]
        * \\text{people} [p/shape]
        * \\text{time ratio} [-]
        * \\text{days per year} [d/y]

    :arg people_gdf: A geodataframe containing the number of people for
        each geometry/shape/row. Each column must be named after the category of people.
    :arg time_ratios: The ratio of time that people spend for each category.
        The sum of all the ratios must be 1.
        Ex: `{'working': 0.3, 'living': 0.7}`.
    :arg emission_factor: The emission factor to use for each of the activities.
        Can be a single value or a dict with the same keys as the categories.
    :arg output_gdfs: Whether the output inventory should contain the emission
        data in gdfs instead of in the gdf(default).
    :arg name: The name of the inventory.
    :arg substance: The substance name of the emissions. 

    :return: The Inventory containing human emissions.
    """

    categories = list(time_ratios.keys())
    if isinstance(emission_factor, (int, float)):
        # Apply the emission to all categories
        emission_factor = {(cat, substance): emission_factor for cat in categories}
    else:
        # Check that the keys are correct
        for key in emission_factor.keys():
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError(f"Invalid key {key = } in emission_factor. Must be a tuple of (category, substance).")
            cat, sub = key
            if cat not in categories:
                raise ValueError(f"Invalid category {cat = } in emission_factor. Must be one of {categories = }")
    if not sum(time_ratios.values()) == 1:
        raise ValueError(
            f"The time ratios must sum up to 1, got {time_ratios.values() = }"
        )

    yearly_emissions = {
        # Calculate the yearly emissions for each of the categories
        # (kg/p/day) * (p/shape) * (-) * (day/y) = (kg/y/shape)
        catsub: factor * people_gdf[catsub[0]] * time_ratios[catsub[0]] * DAY_PER_YR
        for catsub, factor in emission_factor.items()
    }

    inv_kwargs = {}
    if output_gdfs:
        inv_kwargs["gdfs"] = {
            catsub[0]: gpd.GeoDataFrame(
                {catsub[1]: yearly_emissions[catsub]}, geometry=people_gdf.geometry
            )
            for catsub in emission_factor.keys()
        }
    else:
        # Use the main gdf
        inv_kwargs["gdf"] = gpd.GeoDataFrame(
            {catsub: yearly_emissions[catsub] for catsub in emission_factor.keys()},
            geometry=people_gdf.geometry,
        )

    return Inventory.from_gdf(**inv_kwargs, name=name)
