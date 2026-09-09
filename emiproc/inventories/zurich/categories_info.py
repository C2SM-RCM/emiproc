"""Information about the categories."""

from emiproc.inventories import EmissionInfo

# default values are 
# height = 0.0
# height_over_buildings = True
# with = 0.5
# vertical_extension = 3.0
# temperature = 353.0
# speed = 5.0
# source_group = 0 # to be changed later

LARGE_ROAD_TRANSPORT = EmissionInfo(height=0.3, width=7.0)
ZURICH_SOURCES = {
    "c1101_Linienschiffe_Emissionen": EmissionInfo(height=1.5, width=5.0),
    "c1102_PrivaterBootsverkehr_Emissionen": EmissionInfo(height=0.5),
    "c1201_BahnPersonenverkehr_Emissionen": EmissionInfo(height=0.3, width=3.0),
    "c1202_BahnGueterverkehr_Emissionen": EmissionInfo(height=0.3, width=3.0),
    "c1203_Tramverkehr_Emissionen": EmissionInfo(height=0.3, width=2.0),
    "c1204_Kleinbahnen_Emissionen": EmissionInfo(height=0.3, width=2.0),
    "c1301_Personenwagen_Emissionen": LARGE_ROAD_TRANSPORT,
    "c1302_Lastwagen_Emissionen": LARGE_ROAD_TRANSPORT,
    "c1303_Motorraeder_Emissionen": LARGE_ROAD_TRANSPORT,
    "c1304_Linienbusse_Emissionen": LARGE_ROAD_TRANSPORT,
    "c1305_Trolleybusse_Emissionen": LARGE_ROAD_TRANSPORT,
    "c1306_StartStopTankatmung_Emissionen": EmissionInfo(height=0.3),
    "c1307_Lieferwagen_Emissionen": LARGE_ROAD_TRANSPORT,
    "c1308_Reisebusse_Emissionen": LARGE_ROAD_TRANSPORT,
    "c2101_Oelheizungen_Emissionen": EmissionInfo(height=3.0),
    "c2102_Gasheizungen_Emissionen": EmissionInfo(height=3.0),
    "c2103_HolzheizungenLokalisiert_Emissionen": EmissionInfo(height=3.0),
    "c2104_HolzheizungenDispers_Emissionen": EmissionInfo(height=3.0),
    "c2105_Warmwassererzeuger_Emissionen": EmissionInfo(height=3.0),
    "c2201_BHKW_Emissionen": EmissionInfo(height=3.0),
    "c2301_KHKWKehricht_Emissionen": EmissionInfo(),
    "c2302_KHKWErdgas_Emissionen": EmissionInfo(),
    "c2303_KHKWHeizoel_Emissionen": EmissionInfo(),
    "c2401_Klaerschlammverwertung_Emissionen": EmissionInfo(
        comment="new junk Cat, group with other category?",
    ),
    "c3101_MaschinenHochbau_Emissionen": EmissionInfo(
        comment="new Cat for construction related"
    ),
    "c3102_Bitumen_Emissionen": EmissionInfo(
        comment="new Cat for construction related"
    ),
    "c3103_FarbenBaustelle_Emissionen": EmissionInfo(
        comment="new Cat for construction related"
    ),
    "c3104_MaschinenTiefbau_Emissionen": EmissionInfo(
        comment="new Cat for construction related"
    ),
    "c3105_Strassenbelag_Emissionen": EmissionInfo(
        comment="new Cat for construction related"
    ),
    "c3201_Notstromanlagen_Emissionen": EmissionInfo(height=3.0),
    "c3301_Prozessenergie_Emissionen": EmissionInfo(height=3.0),
    "c3401_Metallreinigung_Emissionen": EmissionInfo(),
    "c3402_Holzbearbeitung_Emissionen": EmissionInfo(),
    "c3403_Malereien_Emissionen": EmissionInfo(),
    "c3404_Textilreinigung_Emissionen": EmissionInfo(),
    "c3405_Karosserien_Emissionen": EmissionInfo(),
    "c3406_Raeuchereien_Emissionen": EmissionInfo(height=3.0),
    "c3407_Roestereien_Emissionen": EmissionInfo(height=3.0),
    "c3408_Druckereien_Emissionen": EmissionInfo(),
    "c3409_Laboratorien_Emissionen": EmissionInfo(),
    "c3410_Bierbrauereien_Emissionen": EmissionInfo(),
    "c3411_Brotproduktion_Emissionen": EmissionInfo(),
    "c3412_MedizinischePraxen_Emissionen": EmissionInfo(),
    "c3413_Gesundheitswesen_Emissionen": EmissionInfo(),
    "c3414_Krematorium_Emissionen": EmissionInfo(),
    "c3415_Kompostierung_Emissionen": EmissionInfo(),
    "c3416_Tankstellen_Emissionen": EmissionInfo(),
    "c3417_LoesemittelIG_Emissionen": EmissionInfo(
        comment="new Cat for solvents"
    ),
    "c3418_Vergaerwerk_Emissionen": EmissionInfo(
        comment="new Cat, group with other category?"
    ),
    "c3419_IndustrielleFZ_Emissionen": EmissionInfo(
        comment="added to Agri/Forest vehicel emission",
    ),
    "c4101_ForstwirtschaftlicheFZ_Emissionen": EmissionInfo(width=7.0),
    "c4201_LandwirtschaftlicheFZ_Emissionen": EmissionInfo(),
    "c4301_Nutzflaechen_Emissionen": EmissionInfo(),
    "c4401_Nutztierhaltung_Emissionen": EmissionInfo(),
    "c5101_LoesemittelHH_Emissionen": EmissionInfo(
        comment="new Cat for solvents"
    ),
    "c5201_Gruenabfallverbrennung_Emissionen": EmissionInfo(),
    "c5301_HolzoefenKleingarten_Emissionen": EmissionInfo(),
    "c5401_AbfallverbrennungHaus_Emissionen": EmissionInfo(),
    "c5501_HausZooZirkustiere_Emissionen": EmissionInfo(
        comment="new junk Cat, group with other category?",
    ),
    "c5601_Feuerwerke_Emissionen": EmissionInfo(
        comment="new junk Cat, group with other category?",
    ),
    "c5701_Tabakwaren_Emissionen": EmissionInfo(),
    "c5801_BrandFeuerschaeden_Emissionen": EmissionInfo(
        comment="new junk Cat, group with other category?",
    ),
    "c6101_Waelder_Emissionen": EmissionInfo(),
    "c6201_Grasflaechen_Emissionen": EmissionInfo(),
    "c6301_Gewaesser_Emissionen": EmissionInfo(),
    "c6401_Blitze_Emissionen": EmissionInfo(
        comment="new junk Cat, group with other category?",
    ),
}

ZURICH_SOURCES_Kanton = {
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

ZURICH_DUCKDB_SOURCES = {
    "linienschiff": EmissionInfo(height=1.5, width=5.0),
    "privater_bootsverkehr": EmissionInfo(height=0.5),
    "bahnpersonen": EmissionInfo(height=0.3, width=3.0),
    "bahngüter": EmissionInfo(height=0.3, width=3.0),
    "tram": EmissionInfo(height=0.3, width=2.0),
#    "kleinbahnen": EmissionInfo(height=0.3, width=2.0),
    "personenwagen": LARGE_ROAD_TRANSPORT,
    "lastwagen": LARGE_ROAD_TRANSPORT,
    "motorräder": LARGE_ROAD_TRANSPORT,
    "linienbus": LARGE_ROAD_TRANSPORT,
    "trolleybus": LARGE_ROAD_TRANSPORT,
    "startstoptankatmung": EmissionInfo(height=0.3),
    "lieferwagen": LARGE_ROAD_TRANSPORT,
    "reisebus": LARGE_ROAD_TRANSPORT,
    "ölheizung": EmissionInfo(height=3.0),
    "gasheizung": EmissionInfo(height=3.0),
    "holzheizung_lokalisiert": EmissionInfo(height=3.0),
    "holzheizung_dispers": EmissionInfo(height=3.0),
    "warmwassererzeuger": EmissionInfo(height=3.0),
    "bhkw": EmissionInfo(height=3.0),
    "khkw": EmissionInfo(),
#    "KHKWErdgas": EmissionInfo(),
#    "KHKWHeizoel": EmissionInfo(),
    "klärschlammverwertung": EmissionInfo(),
    "abwasserreinigung": EmissionInfo(),
    "hochbaustelle": EmissionInfo(),
    "bitumen": EmissionInfo(),
    "farben_baustelle": EmissionInfo(),
    "tiefbaustelle": EmissionInfo(),
    "strassenbelagsarbeiten": EmissionInfo(),
    "notstromanlage": EmissionInfo(height=3.0),
    "prozessenergie": EmissionInfo(height=3.0),
    "gewerbebetriebe": EmissionInfo(),
#    "metallreinigung": EmissionInfo(),
#    "holzbearbeitung": EmissionInfo(),
#    "malereien": EmissionInfo(),
#    "textilreinigung": EmissionInfo(),
#    "karosserien": EmissionInfo(),
#    "raeuchereien": EmissionInfo(height=3.0),
#    "roestereien": EmissionInfo(height=3.0),
#    "druckereien": EmissionInfo(),
#    "laboratorien": EmissionInfo(),
#    "bierbrauereien": EmissionInfo(),
#    "brotproduktion": EmissionInfo(),
#    "medizinischepraxen": EmissionInfo(),
#    "gesundheitswesen": EmissionInfo(),
    "krematorium": EmissionInfo(),
#    "kompostierung": EmissionInfo(),
    "tankstelle": EmissionInfo(),
    "lösemittel_IG": EmissionInfo(),
    "vergärwerk": EmissionInfo(),
    "industrie_fahrzeug": EmissionInfo(),
    "forst_fahrzeug": EmissionInfo(width=7.0),
    "landwirtschaft_fahrzeug": EmissionInfo(),
    "nutzfläche": EmissionInfo(),
    "nutztierhaltung": EmissionInfo(),
    "lösemittel_HH": EmissionInfo(),
    "grünabfallverbrennung": EmissionInfo(),
    "holzofen_kleingarten": EmissionInfo(),
    "abfallverbrennung_haus": EmissionInfo(),
    "haus_zoo_zirkustiere": EmissionInfo(),
    "feuerwerk": EmissionInfo(),
    "tabakwaren": EmissionInfo(),
    "brandfeuerschaden": EmissionInfo(),
    "wald": EmissionInfo(),
    "gras": EmissionInfo(),
    "gewässer": EmissionInfo(),
    "blitz": EmissionInfo(),
}

# CH emissions are area sources except for eipro and eipkv
CH_EMISSIONS = EmissionInfo(height=0.3)
ZURICH_CH_SOURCES = {
    'evsfa': CH_EMISSIONS,
    'evsee': CH_EMISSIONS,
    'ehhaf': CH_EMISSIONS,
    'evfgva': CH_EMISSIONS,
    'evsch': CH_EMISSIONS,
    'ehfho': CH_EMISSIONS,
    'evsrh': CH_EMISSIONS,
    'eibau': CH_EMISSIONS,
    'eilpf': CH_EMISSIONS,
    'elfer': CH_EMISSIONS,
    'eifrz': CH_EMISSIONS,
    'evzon': CH_EMISSIONS,
    'ehhab': CH_EMISSIONS,
    'eilmi': CH_EMISSIONS,
    'evfzrh': CH_EMISSIONS,
    'eidep': CH_EMISSIONS,
    'ehfoe': CH_EMISSIONS,
    'elfeu': CH_EMISSIONS,
    'eipkv': EmissionInfo(height=3.0),
    'eikmp': CH_EMISSIONS,
    'elabf': CH_EMISSIONS,
    'eipdh': CH_EMISSIONS,
    'evsra': CH_EMISSIONS,
    'eipro': EmissionInfo(height=3.0),
    'evstrf1': CH_EMISSIONS,
    'evstrf4': CH_EMISSIONS,
    'ehgws': CH_EMISSIONS,
    'evstrf2': CH_EMISSIONS,
    'eipis': CH_EMISSIONS,
    'elver': CH_EMISSIONS,
    'evstrf3': CH_EMISSIONS,
    'ehare': CH_EMISSIONS,
    'eikla': CH_EMISSIONS,
    'ellwm': CH_EMISSIONS,
    'eiprd': CH_EMISSIONS,
    'elfwm': CH_EMISSIONS,
    'ehhan': CH_EMISSIONS,
    'ehmgh': CH_EMISSIONS,
    'eivgn': CH_EMISSIONS,
}
