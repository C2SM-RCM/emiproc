# Cateogies taken from the Bericht CO2_GRAMM_GRAL_Bericht_UGZ_20230207
# Any change is written there as comment
# Categories not having any CO2 are then not included
# There are 4 categories that have been ignored
# 'c5801_BrandFeuerschaeden_Emissionen_Kanton'
# 'c3418_Vergaerwerk_Emissionen_Kanton'
# 'c5601_Feuerwerke_Emissionen_Kanton'
# 'c5701_Tabakwaren_Emissionen_Kanton'
ZH_CO2_Groups = {
    # Shipping
    "Schiffahrt": [
        "c1101_Linienschiffe_Emissionen_Kanton",
        "c1102_PrivaterBootsverkehr_Emissionen_Kanton",
    ],
    # Industry
    "Industrie": [
        "c3201_Notstromanlagen_Emissionen_Kanton",
        "c3301_Prozessenergie_Emissionen_Kanton",
        # "c3401_Metallreinigung_Emissionen_Kanton", # not in bericht
        # "c3402_Holzbearbeitung_Emissionen_Kanton", # not in bericht
        # "c3403_Malereien_Emissionen_Kanton", # not in bericht
        # "c3404_Textilreinigung_Emissionen_Kanton", # not in bericht
        # "c3405_Karosserien_Emissionen_Kanton", # not in bericht
        # "c3406_Raeuchereien_Emissionen_Kanton", # not in bericht
        # "c3407_Roestereien_Emissionen_Kanton", # not in bericht
        # "c3408_Druckereien_Emissionen_Kanton", # not in bericht
        # "c3409_Laboratorien_Emissionen_Kanton", # not in bericht
        "c3410_Bierbrauereien_Emissionen_Kanton",
        "c3411_Brotproduktion_Emissionen_Kanton",
        # "c3412_MedizinischePraxen_Emissionen_Kanton", # not in bericht
        # "c3413_Gesundheitswesen_Emissionen_Kanton", # not in bericht
        "c3414_Krematorium_Emissionen_Kanton",
        "c3416_Tankstellen_Emissionen_Kanton",
    ],
    # Fossil heating
    "FeuerungenFossil": [
        "c2101_Oelheizungen_Emissionen_Kanton",
        "c2102_Gasheizungen_Emissionen_Kanton",
        "c2105_Warmwassererzeuger_Emissionen_Kanton",
        "c2201_BHKW_Emissionen_Kanton",
    ],
    # Heating
    "Feuerungen": [
        "c2103_HolzheizungenLokalisiert_Emissionen_Kanton",
        "c2104_HolzheizungenDispers_Emissionen_Kanton",
        "c2401_Klaerschlammverwertung_Emissionen_Kanton",
    ],
    # Heting Power Plants
    "KehrichtheizkraftwerkeKHKW": [
        "c2301_KHKWKehricht_Emissionen_Kanton",
        "c2302_KHKWErdgas_Emissionen_Kanton",
        "c2303_KHKWHeizoel_Emissionen_Kanton",
    ],
    # # Solvents and product use
    # "GNFR_E": [
    #     "c3417_LoesemittelIG_Emissionen_Kanton",
    #     "c5101_LoesemittelHH_Emissionen_Kanton",
    # ],
    # Road transport
    "Strassenverkehr": [
        "c1301_Personenwagen_Emissionen_Kanton",
        "c1303_Motorraeder_Emissionen_Kanton",
        "c1306_StartStopTankatmung_Emissionen_Kanton",
        "c1307_Lieferwagen_Emissionen_Kanton",
        "c1308_Reisebusse_Emissionen_Kanton",
    ],
    # Heavy transport
    "Schwerverkehr": [
        "c1302_Lastwagen_Emissionen_Kanton",
    ],
    # Public transport
    "OffentlicherVerkehr ": [
        "c1304_Linienbusse_Emissionen_Kanton",
        # "c1305_Trolleybusse_Emissionen_Kanton", # This is assumed from the bericht
    ],
    # Offroad mobility
    # "GNFR_I": [
    #    "c1201_BahnPersonenverkehr_Emissionen_Kanton",
    #    "c1202_BahnGueterverkehr_Emissionen_Kanton",
    #    "c1203_Tramverkehr_Emissionen_Kanton",
    #    "c1204_Kleinbahnen_Emissionen_Kanton",
    #    # c31xx are construction stuff
    #    "c3102_Bitumen_Emissionen_Kanton",
    #    "c3103_FarbenBaustelle_Emissionen_Kanton",
    #    "c3105_Strassenbelag_Emissionen_Kanton",
    # ],
    "FahrzeugeMaschinen": [
        "c3101_MaschinenHochbau_Emissionen_Kanton",
        "c3104_MaschinenTiefbau_Emissionen_Kanton",
        "c3419_IndustrielleFZ_Emissionen_Kanton",
        "c4101_ForstwirtschaftlicheFZ_Emissionen_Kanton",
        "c4201_LandwirtschaftlicheFZ_Emissionen_Kanton",
    ],
    # Waste
    "Umschwung": [
        "c5201_Gruenabfallverbrennung_Emissionen_Kanton",
        "c5301_HolzoefenKleingarten_Emissionen_Kanton",
        "c5401_AbfallverbrennungHaus_Emissionen_Kanton",
        # "c3418_Vergaerwerk_Emissionen_Kanton",
    ],
    # # AgriLivestock
    # "GNFR_K": [
    #     "c4401_Nutztierhaltung_Emissionen_Kanton",
    # ],
    # # AgriOther
    # "GNFR_L": [
    #     "c4301_Nutzflaechen_Emissionen_Kanton",
    # ],
    # # Others
    # "GNFR_R": [
    #     "c5501_HausZooZirkustiere_Emissionen_Kanton",
    #     "c5601_Feuerwerke_Emissionen_Kanton",
    #     "c5701_Tabakwaren_Emissionen_Kanton",
    #     "c5801_BrandFeuerschaeden_Emissionen_Kanton",
    #     "c6101_Waelder_Emissionen_Kanton",
    #     "c6201_Grasflaechen_Emissionen_Kanton",
    #     "c6301_Gewaesser_Emissionen_Kanton",
    #     "c6401_Blitze_Emissionen_Kanton",
    # ],
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
