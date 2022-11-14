# %%
from os import PathLike
from pathlib import Path
from emiproc.inventories import Inventory, Substance
from emiproc.inventories.utils import list_categories, process_emission_category
from shapely.geometry import Point


class MapLuftZurich(Inventory):
    """Inventory of Zurich."""

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
            "VOC",
            "Benzol",
        ],
        remove_josefstrasse_khkw: bool = True,
    ) -> None:
        """Load the mapluft inventory.

        :arg mapluft_gdb: The Mapluft file
        :arg substances: A list of substances to load.
            (by default all of them)
        :arg remove_josefstrasse_khkw: Whether the incineration plant
            at josefstrasse should be removed from the inventory.
            It should be planned to be removed in March 2021.
            In case  remove_josefstrasse_khkw, the emissions are not set to any
            other location in the inventory.
        """
        super().__init__()
        self.mapluft_gdb = Path(mapluft_gdb)

        categories = list_categories(mapluft_gdb)

        emission_names = {f"Emission_{sub}": sub for sub in substances}

        # Mapluft has no grid
        self.gdf = None

        self.gdfs = {}
        for category in categories:
            gdf = process_emission_category(mapluft_gdb, category)
            # Select the columns with the emissions values
            names_in_gdf = [
                name for name in emission_names.keys() if name in gdf.columns
            ]
            gdf = gdf.loc[:, list(names_in_gdf) + ["geometry"]]
            if remove_josefstrasse_khkw:
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


# %%

if __name__ == "__main__":
    mapluft_file = Path(
        r"H:\ZurichEmissions\Data\mapLuft_2020_v2021\mapLuft_2020_v2021.gdb"
    )
    inv = MapLuftZurich(mapluft_file)
