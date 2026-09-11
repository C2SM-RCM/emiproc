# Cateogies taken from the Bericht CO2_GRAMM_GRAL_Bericht_UGZ_20230207
# Any change is written there as comment
# Categories not having any CO2 are then not included
# There are 4 categories that have been ignored
# 'c5801_BrandFeuerschaeden_Emissionen'
# 'c3418_Vergaerwerk_Emissionen'
# 'c5601_Feuerwerke_Emissionen'
# 'c5701_Tabakwaren_Emissionen'
ZH_CO2_GROUPS = {
    # Shipping
    "Schiffahrt": [
        "c1101_Linienschiffe_Emissionen",
        "c1102_PrivaterBootsverkehr_Emissionen",
    ],
    # Industry
    "Industrie": [
        "c3201_Notstromanlagen_Emissionen",
        "c3301_Prozessenergie_Emissionen",
        # "c3401_Metallreinigung_Emissionen", # not in bericht
        # "c3402_Holzbearbeitung_Emissionen", # not in bericht
        # "c3403_Malereien_Emissionen", # not in bericht
        # "c3404_Textilreinigung_Emissionen", # not in bericht
        # "c3405_Karosserien_Emissionen", # not in bericht
        # "c3406_Raeuchereien_Emissionen", # not in bericht
        # "c3407_Roestereien_Emissionen", # not in bericht
        # "c3408_Druckereien_Emissionen", # not in bericht
        # "c3409_Laboratorien_Emissionen", # not in bericht
        "c3410_Bierbrauereien_Emissionen",
        "c3411_Brotproduktion_Emissionen",
        # "c3412_MedizinischePraxen_Emissionen", # not in bericht
        # "c3413_Gesundheitswesen_Emissionen", # not in bericht
        "c3414_Krematorium_Emissionen",
        "c3416_Tankstellen_Emissionen",
    ],
    # Fossil heating
    "FeuerungenFossil": [
        "c2101_Oelheizungen_Emissionen",
        "c2102_Gasheizungen_Emissionen",
        "c2105_Warmwassererzeuger_Emissionen",
        "c2201_BHKW_Emissionen",
    ],
    # Heating
    "Feuerungen": [
        "c2103_HolzheizungenLokalisiert_Emissionen",
        "c2104_HolzheizungenDispers_Emissionen",
        "c2401_Klaerschlammverwertung_Emissionen",
    ],
    # Heting Power Plants
    "KehrichtheizkraftwerkeKHKW": [
        "c2301_KHKWKehricht_Emissionen",
        "c2302_KHKWErdgas_Emissionen",
        "c2303_KHKWHeizoel_Emissionen",
    ],
    # # Solvents and product use
    # "GNFR_E": [
    #     "c3417_LoesemittelIG_Emissionen",
    #     "c5101_LoesemittelHH_Emissionen",
    # ],
    # Road transport
    "Strassenverkehr": [
        "c1301_Personenwagen_Emissionen",
        "c1303_Motorraeder_Emissionen",
        "c1306_StartStopTankatmung_Emissionen",
        "c1307_Lieferwagen_Emissionen",
        "c1308_Reisebusse_Emissionen",
    ],
    # Heavy transport
    "Schwerverkehr": [
        "c1302_Lastwagen_Emissionen",
    ],
    # Public transport
    "OffentlicherVerkehr ": [
        "c1304_Linienbusse_Emissionen",
        # "c1305_Trolleybusse_Emissionen", # This is assumed from the bericht
    ],
    # Offroad mobility
    # "GNFR_I": [
    #    "c1201_BahnPersonenverkehr_Emissionen",
    #    "c1202_BahnGueterverkehr_Emissionen",
    #    "c1203_Tramverkehr_Emissionen",
    #    "c1204_Kleinbahnen_Emissionen",
    #    # c31xx are construction stuff
    #    "c3102_Bitumen_Emissionen",
    #    "c3103_FarbenBaustelle_Emissionen",
    #    "c3105_Strassenbelag_Emissionen",
    # ],
    "FahrzeugeMaschinen": [
        "c3101_MaschinenHochbau_Emissionen",
        "c3104_MaschinenTiefbau_Emissionen",
        "c3419_IndustrielleFZ_Emissionen",
        "c4101_ForstwirtschaftlicheFZ_Emissionen",
        "c4201_LandwirtschaftlicheFZ_Emissionen",
    ],
    # Waste
    "Umschwung": [
        "c5201_Gruenabfallverbrennung_Emissionen",
        "c5301_HolzoefenKleingarten_Emissionen",
        "c5401_AbfallverbrennungHaus_Emissionen",
        # "c3418_Vergaerwerk_Emissionen",
    ],
    # # AgriLivestock
    # "GNFR_K": [
    #     "c4401_Nutztierhaltung_Emissionen",
    # ],
    # # AgriOther
    # "GNFR_L": [
    #     "c4301_Nutzflaechen_Emissionen",
    # ],
    # # Others
    # "GNFR_R": [
    #     "c5501_HausZooZirkustiere_Emissionen",
    #     "c5601_Feuerwerke_Emissionen",
    #     "c5701_Tabakwaren_Emissionen",
    #     "c5801_BrandFeuerschaeden_Emissionen",
    #     "c6101_Waelder_Emissionen",
    #     "c6201_Grasflaechen_Emissionen",
    #     "c6301_Gewaesser_Emissionen",
    #     "c6401_Blitze_Emissionen",
    # ],
}

ZH_CO2_GROUPS_KANTON = {
    group_name: [cat + "_Kanton" for cat in cats]
    for group_name, cats in ZH_CO2_GROUPS.items()
}

# Swiss invenotry is also used to be remapped on the the boundaries around zuirch
# Note missing co2 categories: many have been removeved becaue not in the zh area
# eipwp # Industrial point sources
# eipzm # Zement Werke
# evsfa # Shiffarts lilinen should be verkeher
# evsrh # Rehin shiffarts, not in zh but still
# eipis # pisten fahrzeuge
# evsra # Schienenverkehr Rangieren
# elfer # landwirtschaftliche Nutzflächen
# ehhab # Hausalte Brande
# ehhaf # Haushalte andere Feuerwerk etc
CH_2_GNFR = {
    # Road transport
    "Verkehr": [
        # "evstr_ch4",
        # "evstr_co",
        # "evstr_co2",
        # "evstr_n2o",
        # "evstr_nh3",
        # "evstr_nmvoc",
        # "evstr_nox",
        # "evstr_so2",
        "evstr",
        "evzon",
        "evsee",
    ],
    "FeuerungenFossil": [
        "ehfoe",
        "ehgws",
        "ehare",
    ],
    "Feuerungen": [
        "ehfho",
    ],
    "IndustrieUndRest": [
        "ehhan",
        "ehmgh",
        "eibau",
        "eifrz",
        "eilmi",  # Lösungsmittel Industrie
        "eilpf",
        "eiprd",
        "eipkv",  # Punktquellen KVA (Kehrichtverbrennungsanlagen ) == Waste incinerators
        "evsch",
        "evfgva",  # GVA airport, should not be in the area but put there as zh is there
        "evfzhr",  # ZH airport
        "elfwm",
        "ellwm",
        "eivgn",
        "eipro",
    ],
    # # Industry
    # "GNFR_B": [
    #     "eipwp",  # this is the weitere punktquelle (additional point sources)
    #     "eipzm",
    # ],
    # # Other stationary combustion (services, residential, agriculture)
    # "GNFR_C": [
    #     "eipdh",
    #     "elfeu",
    # ],
    # # Fugitives
    # "GNFR_D": [
    #     "eilgk",
    #     "evklm",
    #     "evtrk",
    # ],
    # # Solvents and product use
    # "GNFR_E": [
    #     "ehlmk",  # Lösungsmittel Konsumprodukte
    # ],
    # # Shipping
    # "GNFR_G": [
    #     "evsfa",
    #     "evsrh",
    # ],
    # # Aviation
    # "GNFR_H": [],
    # # Offroad mobility
    # "GNFR_I": [
    #     "eipis",
    #     "evsra",
    # ],
    # # Waste
    # "GNFR_J": [
    #     "eidep",
    #     "eikla",
    #     "eikmp",
    #     "elabf",
    #     "elver",
    # ],
    # # AgriLivestock
    # "GNFR_K": [
    #     "elapp",
    #     "elsto",
    # ],
    # #  AgriOther
    # "GNFR_L": [
    #     "elfer",
    # ],
    # # Others
    # "GNFR_R": [
    #     "ehhab",
    #     "ehhaf",
    #     "enwal",
    # ],
}


ZH_CO2_DUCK_GROUPS = {
    # Waste heating power plant (Hagenholz)
    "KHKW": [
        "khkw",
    ],
    # Industry
    "Industrie": [
        "notstromanlage",
        "prozessenergie",
        "gewerbebetriebe",
        # "tankstelle",
    ],
    # Fossil Heating
    "FeuerungenFossil": [
        "ölheizung",
        "gasheizung",
        "warmwassererzeuger",
        "bhkw",
    ],
    # Non-fossil Heating
    "FeuerungenBio": [
        "holzheizung_lokalisiert",
        "holzheizung_dispers",
    ],
    #    "GNFR_D": [
    # "netzverluste",
    #    ],
    #    "Loesemittel": [
    # "lösemittel_IG",
    # "lösemittel_HH",
    #    ],
    # Road traffic (car, light-duty, motorcycles)
    "Strassenverkehr": [
        "personenwagen",
        "motorräder",
        "startstoptankatmung",
        "lieferwagen",
    ],
    # Heavy duty traffic
    "Schwerverkehr": [
        "lastwagen",
        "reisebus",
    ],
    "OeffentlicherVerkehr": [
        "linienbus",
        # "trolleybus",
    ],
    "Schiffahrt": [
        "linienschiff",
        "privater_bootsverkehr",
    ],
    # Offroad mobility
    # "GNFR_I": [
    #    "bahnpersonen",
    #    "bahngüter",
    #    "tram",
    #    "bitumen",
    #    "farben_baustelle",
    #    "strassenbelagsarbeiten",
    # ],
    "FahrzeugeMaschinen": [
        "hochbaustelle",
        "tiefbaustelle",
        "industrie_fahrzeug",
        "forst_fahrzeug",
        "landwirtschaft_fahrzeug",
    ],
    # Waste
    "Umschwung": [
        "abwasserreinigung",
        "klärschlammverwertung",
        "vergärwerk",
        "krematorium",
        "grünabfallverbrennung",
        "holzofen_kleingarten",
        "abfallverbrennung_haus",
    ],
    #   # AgriLivestock
    #    "GNFR_K": [
    # "nutztierhaltung",
    #    ],
    #    # AgriOther
    #    "GNFR_L": [
    # "nutzfläche",
    #    ],
    #    # Others
    #    "GNFR_R": [
    # "haus_zoo_zirkustiere",
    # "feuerwerk",
    # "tabakwaren",
    # "brandfeuerschaden",
    # "wald",
    # "gras",
    # "gewässer",
    # "blitz",
    #    ],
}


# Swiss invenotry is also used to be remapped on the the boundaries around Zurich
# Note missing CO2 categories: many have been removeved becaue not in the ZH area
# eipwp # Industrial point sources
# eipzm # Zement Werke
# evsrh # Rhein ships, not in zh but still
# eipis # pisten fahrzeuge
# elfer # landwirtschaftliche Nutzflächen
# ehhab # Hausalte Brande
# ehhaf # Haushalte andere Feuerwerk etc
CH_CO2_GROUPS = {
    # Feuerungen/Haushalte Fossil
    "FeuerungenFossil_CH": [
        "ehare",
        "ehfoe",
        "ehgws",
    ],
    # Other stationary combustion (services, residential, agriculture)
    "FeuerungenBio_CH": [
        "ehfho",
    ],
    # Road transport
    "Verkehr_CH": [
        "evstrf1",  # Passenger cars
        "evstrf2",  # light duty
        "evstrf3",  # Heavy duty
        "evstrf4",  # motorcycles
        "evzon",  # zone traffic, cold start/evaporation
        "evsee",  # shipping lakes
        "evsfa",  # Schiffahrtslininen
        # "evsrh",    # shipping rivers
    ],
    # Industry and rest
    "Rest_CH": [
        "ehhan",  # Haushalte andere private
        "ehmgh",  # Maschinen Garten und Hobby
        "eibau",  # Baumaschinen
        "eifrz",  # Industriefahrzeuge
        "eilmi",  # Lösungsmittel Industrie
        "eilpf",  # Dienstleistungen Landschaftspflege
        "eiprd",  # Dienstleistungen Öl und Gas
        "eipkv",  # Waste energy/heat plants (point sources)
        "evsch",  # Schienenverkehr Bau-/Dienstzüge
        "evfzrh",  # Flughafen Zurich
        "ellwm",  # landwirtschaftliche Maschinen
        "elfwm",  # forstwirtschaftliche Maschinen
        "eivgn",  # Verluste Gasnetz
        "eipro",  # Flächenquellen Industrie
        "evsra",  # Schienenverkehr Rangieren
        "ehhab",  # Haushalte andere Brände
        "eikmp",  # Kompostierung
        "eipdh",  # Diensleistungen Holz und Kohle
        "ehhaf",  # Haushalte andere feuerwerk
        "elver",  # Vergärung
        "eidep",  # Deponien
        "eipis",  # Pistenfahrzeuge
        "evsrh",  # Rhein ships
        "eikla",  # Kläranlagen
        "elabf",  # Abfallverbrennung Land- und Forstwirtschaft
        "evfgva",  # Geneva airport
        "elfer",
        "elfeu",
        # "eipwp",  # this is the weitere punktquelle (additional point sources)
        # "eipzm",  # Punktquellen Zementwerke (point sources cement plants)
        # "elapp",
        # "elsto",
        # "enwal",  # Emissionen aus Waldern (emissions from forests)
    ],
}
