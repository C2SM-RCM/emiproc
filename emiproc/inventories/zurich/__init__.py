"""Zurich inventory."""
from os import PathLike
from pathlib import Path
from emiproc.inventories import Inventory, Substance, Category
from emiproc.inventories.utils import list_categories, process_emission_category
from shapely.geometry import Point


class MapLuftZurich(Inventory):
    """Inventory of Zurich based on the mapluft.gbd file.

    It contains only shaped emissions for each categories.
    """

    mapluft_gdb: Path

    def __init__(
        self,
        mapluft_gdb: PathLike,
        # Hardcoded the name of the substances for simplicity
        substances: list[Substance] = [
            "CO2",
            "CO",
            "PM10ex",
            "PM10non",
            "PM25ex",
            "PM25non",
            "SO2",
            "NOx",
            "N2O",
            "NH3",
            "CH4",
            "BC",
            "VOC",
            "benzene",
        ],
        categories: list[Category] = [],
        remove_josefstrasse_khkw: bool = False,
        convert_lines_to_polygons: bool = True,
    ) -> None:
        """Load the mapluft inventory.

        :arg mapluft_gdb: The Mapluft file
        :arg substances: A list of substances to load.
            (by default all of them).
            Categories not contianing any of the substances are not loaded.
        :arg categories: A list of categories to load (if one is interested
            in only a subset).
            If not specified, all categories are loaded.
        :arg remove_josefstrasse_khkw: Whether the incineration plant
            at josefstrasse should be removed from the inventory.
            Emission for category 'c2301_KHKWKehricht_Emissionen_Kanton' at 
            the Josefstrasse location will be removed from the inventory.
            It should be planned to be removed in March 2021.
            The other Josefstrasse category will still be present, as they
            account for some kinds of energy production.
            In case  remove_josefstrasse_khkw, the emissions are not set to any
            other location in the inventory.
        :arg convert_lines_to_polygons: Whether this should convert line emissions 
            to polygons. Only few models can handle line emissions.
            The default width of the line is 10m for all categories.
            This is not currently not changeable.
        """
        super().__init__()
        self.mapluft_gdb = Path(mapluft_gdb)

        if not self.mapluft_gdb.is_dir():
            raise FileNotFoundError(f"{self.mapluft_gdb=} is not a existing directory.")

        if not categories:
            # Load all the categories
            categories = list_categories(mapluft_gdb)
        else:
            self.history.append("Only a subset of categories was loaded.")

        # Maps substances name from the mapluft files to the names in emiproc
        emission_names = {
            (f"Emission_{sub}" if sub != "benzene" else "Emission_Benzol"): sub
            for sub in substances
        }

        # Mapluft has no grid
        self.gdf = None

        self.gdfs = {}

        for category in categories:
            gdf = process_emission_category(
                mapluft_gdb, category, convert_lines_to_polygons=convert_lines_to_polygons
            )
            # Select the columns with the emissions values
            names_in_gdf = [
                name for name in emission_names.keys() if name in gdf.columns
            ]
            if not names_in_gdf:
                # Ignore categories not containing any substance of interest
                continue
            gdf = gdf.loc[:, list(names_in_gdf) + ["geometry"]]

            
            if remove_josefstrasse_khkw and category == "c2301_KHKWKehricht_Emissionen_Kanton":
                # Check point sources at the josefstrasse location
                mask_josefstrasse = gdf.geometry == Point(2681839.000, 1248988.000)
                if any(mask_josefstrasse):
                    self.history.append(
                        f"Removed {sum(mask_josefstrasse)} point source"
                        f"from josefstrasse of {category=}."
                    )
                gdf = gdf.loc[~mask_josefstrasse]
            # Only keep the substances
            self.gdfs[category] = gdf.rename(columns=emission_names, errors="ignore")


if __name__ == "__main__":

    from emiproc.inventories.zurich.categories_info import ZURICH_SOURCES
    mapluft_file = Path("/store/empa/em05/mapluft/mapLuft_2020_v2021.gdb")
    inv = MapLuftZurich(mapluft_file)
    inv.emission_infos = ZURICH_SOURCES
    print(inv.emission_infos)
