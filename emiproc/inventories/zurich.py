"""Zurich inventory."""
from os import PathLike
from pathlib import Path
from emiproc.inventories import EmissionInfo, Inventory, Substance, Category
from emiproc.inventories.utils import list_categories, process_emission_category
from shapely.geometry import Point

# Information about the categories
LARGE_ROAD_TRANSPORT = EmissionInfo(height=0.3, width=7.0)
ZURICH_SOURCES = {
    "c1101_Linienschiffe_Emissionen_Kanton": EmissionInfo(height=1.5, width=5.0),
    "c1102_PrivaterBootsverkehr_Emissionen_Kanton": EmissionInfo(height=0.5),
    "c1201_BahnPersonenverkehr_Emissionen_Kanton": EmissionInfo(height=0.3, width=3.0),
    "c1202_BahnGueterverkehr_Emissionen_Kanton": EmissionInfo(height=0.3, width=3.0),
    "c1203_Tramverkehr_Emissionen_Kanton": EmissionInfo(height=0.3, width=2.0),
    "c1204_Kleinbahnen_Emissionen_Kanton": EmissionInfo(height=0.3, width=2.0),
    "c1301_Personenwagen_Emissionen_Kanton": LARGE_ROAD_TRANSPORT,
    "c1302_Lastwagen_Emissionen_Kanton": LARGE_ROAD_TRANSPORT,
    "c1303_Motorraeder_Emissionen_Kanton": LARGE_ROAD_TRANSPORT,
    "c1304_Linienbusse_Emissionen_Kanton": LARGE_ROAD_TRANSPORT,
    "c1305_Trolleybusse_Emissionen_Kanton": LARGE_ROAD_TRANSPORT,
    "c1306_StartStopTankatmung_Emissionen_Kanton": EmissionInfo(height=0.3),
    "c1307_Lieferwagen_Emissionen_Kanton": LARGE_ROAD_TRANSPORT,
    "c1308_Reisebusse_Emissionen_Kanton": LARGE_ROAD_TRANSPORT,
    "c2101_Oelheizungen_Emissionen_Kanton": EmissionInfo(height=3.0),
    "c2102_Gasheizungen_Emissionen_Kanton": EmissionInfo(height=3.0),
    "c2103_HolzheizungenLokalisiert_Emissionen_Kanton": EmissionInfo(height=3.0),
    "c2104_HolzheizungenDispers_Emissionen_Kanton": EmissionInfo(height=3.0),
    "c2105_Warmwassererzeuger_Emissionen_Kanton": EmissionInfo(height=3.0),
    "c2201_BHKW_Emissionen_Kanton": EmissionInfo(height=3.0),
    "c2301_KHKWKehricht_Emissionen_Kanton": EmissionInfo(),
    "c2302_KHKWErdgas_Emissionen_Kanton": EmissionInfo(),
    "c2303_KHKWHeizoel_Emissionen_Kanton": EmissionInfo(),
    "c2401_Klaerschlammverwertung_Emissionen_Kanton": EmissionInfo(
        comment="new junk Cat, group with other category?",
    ),
    "c3101_MaschinenHochbau_Emissionen_Kanton": EmissionInfo(
        comment="new Cat for construction related"
    ),
    "c3102_Bitumen_Emissionen_Kanton": EmissionInfo(
        comment="new Cat for construction related"
    ),
    "c3103_FarbenBaustelle_Emissionen_Kanton": EmissionInfo(
        comment="new Cat for construction related"
    ),
    "c3104_MaschinenTiefbau_Emissionen_Kanton": EmissionInfo(
        comment="new Cat for construction related"
    ),
    "c3105_Strassenbelag_Emissionen_Kanton": EmissionInfo(
        comment="new Cat for construction related"
    ),
    "c3201_Notstromanlagen_Emissionen_Kanton": EmissionInfo(height=3.0),
    "c3301_Prozessenergie_Emissionen_Kanton": EmissionInfo(height=3.0),
    "c3401_Metallreinigung_Emissionen_Kanton": EmissionInfo(),
    "c3402_Holzbearbeitung_Emissionen_Kanton": EmissionInfo(),
    "c3403_Malereien_Emissionen_Kanton": EmissionInfo(),
    "c3404_Textilreinigung_Emissionen_Kanton": EmissionInfo(),
    "c3405_Karosserien_Emissionen_Kanton": EmissionInfo(),
    "c3406_Raeuchereien_Emissionen_Kanton": EmissionInfo(height=3.0),
    "c3407_Roestereien_Emissionen_Kanton": EmissionInfo(height=3.0),
    "c3408_Druckereien_Emissionen_Kanton": EmissionInfo(),
    "c3409_Laboratorien_Emissionen_Kanton": EmissionInfo(),
    "c3410_Bierbrauereien_Emissionen_Kanton": EmissionInfo(),
    "c3411_Brotproduktion_Emissionen_Kanton": EmissionInfo(),
    "c3412_MedizinischePraxen_Emissionen_Kanton": EmissionInfo(),
    "c3413_Gesundheitswesen_Emissionen_Kanton": EmissionInfo(),
    "c3414_Krematorium_Emissionen_Kanton": EmissionInfo(),
    "c3415_Kompostierung_Emissionen_Kanton": EmissionInfo(),
    "c3416_Tankstellen_Emissionen_Kanton": EmissionInfo(),
    "c3417_LoesemittelIG_Emissionen_Kanton": EmissionInfo(
        comment="new Cat for solvents"
    ),
    "c3418_Vergaerwerk_Emissionen_Kanton": EmissionInfo(
        comment="new Cat, group with other category?"
    ),
    "c3419_IndustrielleFZ_Emissionen_Kanton": EmissionInfo(
        comment="added to Agri/Forest vehicel emission",
    ),
    "c4101_ForstwirtschaftlicheFZ_Emissionen_Kanton": EmissionInfo(width=7.0),
    "c4201_LandwirtschaftlicheFZ_Emissionen_Kanton": EmissionInfo(),
    "c4301_Nutzflaechen_Emissionen_Kanton": EmissionInfo(),
    "c4401_Nutztierhaltung_Emissionen_Kanton": EmissionInfo(),
    "c5101_LoesemittelHH_Emissionen_Kanton": EmissionInfo(
        comment="new Cat for solvents"
    ),
    "c5201_Gruenabfallverbrennung_Emissionen_Kanton": EmissionInfo(),
    "c5301_HolzoefenKleingarten_Emissionen_Kanton": EmissionInfo(),
    "c5401_AbfallverbrennungHaus_Emissionen_Kanton": EmissionInfo(),
    "c5501_HausZooZirkustiere_Emissionen_Kanton": EmissionInfo(
        comment="new junk Cat, group with other category?",
    ),
    "c5601_Feuerwerke_Emissionen_Kanton": EmissionInfo(
        comment="new junk Cat, group with other category?",
    ),
    "c5701_Tabakwaren_Emissionen_Kanton": EmissionInfo(),
    "c5801_BrandFeuerschaeden_Emissionen_Kanton": EmissionInfo(
        comment="new junk Cat, group with other category?",
    ),
    "c6101_Waelder_Emissionen_Kanton": EmissionInfo(),
    "c6201_Grasflaechen_Emissionen_Kanton": EmissionInfo(),
    "c6301_Gewaesser_Emissionen_Kanton": EmissionInfo(),
    "c6401_Blitze_Emissionen_Kanton": EmissionInfo(
        comment="new junk Cat, group with other category?",
    ),
}


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
            "VOC",
            "benzene",
        ],
        categories: list[Category] = [],
        remove_josefstrasse_khkw: bool = True,
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
            It should be planned to be removed in March 2021.
            In case  remove_josefstrasse_khkw, the emissions are not set to any
            other location in the inventory.
        :arg convert_lines_to_polygons: Whether this should convert line emissions 
            to polygons. Only few models can handle line emissions.
            The default width of the line is 10m for all categories.
            This is not currently not changeable.
        """
        super().__init__()
        self.mapluft_gdb = Path(mapluft_gdb)

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


if __name__ == "__main__":
    mapluft_file = Path("/store/empa/em05/mapluft/mapLuft_2020_v2021.gdb")
    inv = MapLuftZurich(mapluft_file)
    inv.emission_infos = ZURICH_SOURCES
    print(inv.emission_infos)
