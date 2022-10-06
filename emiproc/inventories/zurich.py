# %%
from os import PathLike
from pathlib import Path
from emiproc.inventories import Inventory, Substance
from emiproc.inventories.utils import list_categories, process_emission_category


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
    ) -> None:
        """Load the mapluft inventory.

        :arg mapluft_gdb: The Mapluft file
        :arg substances: A list of substances to load.
            (by default all of them)
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
            # Only keep the substances
            self.gdfs[category] = gdf.rename(columns=emission_names, errors="ignore")


# %%

if __name__ == "__main__":
    inv = MapLuftZurich(
        Path(r"H:\ZurichEmissions\Data\mapLuft_2020_v2021\mapLuft_2020_v2021.gdb")
    )


# %%
