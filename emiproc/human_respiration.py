"""Emiproc provides tools to help generate human respiration emissions.



"""
from __future__ import annotations

from enum import Enum
from os import PathLike
import geopandas as gpd
from emiproc.grids import RegularGrid
from emiproc.inventories import Inventory
from emiproc.utilities import DAY_PER_YR

class EmissionFactor(Enum):
    """Emissions factors are how much CO2 is produce per individual per day

    Units are kg/day (kg of CO2)
    """

    # Very simple estimation for aproximate results
    ROUGH_ESTIMATON = 1.0


def load_data_from_quartieranalyse(
    file: PathLike,
    grid: RegularGrid | None,
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
    time_ratios: dict[str, float],
    emission_factor: dict[str, float] | float = EmissionFactor.ROUGH_ESTIMATON.value,
    output_gdfs: bool = False,
    name: str = "human_respiration"
) -> Inventory:
    """Convert people living there to emissions.

    People produce only CO2

    :arg people_gdf: A geodataframe containing the number of people from each emission category.
    :arg time_ratios: The ratio of time that people spend in each of the 
        activities. 
    :arg time_profiles: Instead of just ratio we can provied time profiles for 
        each of the activities.
    :arg emission_factor: The emission factor to use for each of the activities.
    :arg output_gdfs: Whether the output inventory should contain the emission
        data in gdfs instead of in the gdf(default).

    :return: The Inventory of human emissions.

    """

    categories = list(time_ratios.keys())
    if isinstance(emission_factor, (int, float)):
        emission_factor = {cat: emission_factor for cat in categories}

    if not sum(time_ratios.values()) == 1:
        raise ValueError(f"The time ratios must sum up to 1, got {time_ratios.values() = }")

    yearly_emissions = {
        # Calculate the yearly emissions for each of the categories
        # (kg/p/day) * (p/shape) * (-) * (day/y) = (kg/y/shape)
        cat: emission_factor[cat] * people_gdf[cat] * time_ratios[cat] * DAY_PER_YR  
        for cat in categories
    }
    
    inv_kwargs = {}
    if output_gdfs:
        inv_kwargs['gdfs'] = {
            cat: gpd.GeoDataFrame({'CO2': yearly_emissions[cat]},geometry=people_gdf.geometry)
            for cat in categories
        }
    else:
        # Use the main gdf 
        inv_kwargs['gdf'] = gpd.GeoDataFrame(
            {
                (cat, 'CO2'): yearly_emissions[cat] for cat in categories
            },
            geometry=people_gdf.geometry
        )
    
    return Inventory.from_gdf(**inv_kwargs, name=name)



