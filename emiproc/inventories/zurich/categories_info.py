"""Information about the categories."""
from emiproc.inventories import EmissionInfo


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

