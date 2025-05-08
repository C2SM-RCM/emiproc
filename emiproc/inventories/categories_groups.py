"""Mappings to regroup categories together."""

CH_2_GNFR = {
    # PublicPower
    "GNFR_A": [
        # Waste incinerator we agreed on nov. 22 to put them in the public power category
        "eipkv",  # Punktquellen KVA (Kehrichtverbrennungsanlagen ) == Waste incinerators
    ],
    # Industry
    "GNFR_B": [
        "eipro",  # Flächenquellen Industrie
        "eipwp",  # this is the weitere punktquelle (additional point sources)
        "eipzm",  # Punktquellen Zementwerke (point sources cement plants)
    ],
    # Other stationary combustion (services, residential, agriculture)
    "GNFR_C": [
        "ehare",
        "ehfho",
        "ehfoe",
        "ehgws",
        "eipdh",  # Diensleistungen Holz und Kohle
        "eiprd",  # Dienstleistungen Ol und Gas
        "elfeu",
    ],
    # Fugitives
    "GNFR_D": [
        "eilgk",  # F-Gase: Läden, Gebäude mit Klimaanlagen Refrigeration (F-gases: shops, buildings with air conditioning)
        "eivgn",  # Verluste Gasnetz (losses gas network)
        "evklm",  # F-Gase: Klimaanlagen Motorfahrzeuge (F-gases: air conditioning motor vehicles)
        "evtrk",  # F-Gase: Transporte mit Kühlung  (F-gases: transports with cooling)
    ],
    # Solvents and product use
    "GNFR_E": [
        "eilmi",  # Lösungsmittel Industrie
        "ehlmk",  # Lösungsmittel Konsumprodukte
    ],
    # Road transport
    "GNFR_F": [
        # "evstr_ch4",
        # "evstr_co",
        # "evstr_co2",
        # "evstr_n2o",
        # "evstr_nh3",
        # "evstr_nmvoc",
        # "evstr_nox",
        # "evstr_so2",
        "evstr",  # Strassenverkehr (road transport)
        "evzon",  # Zonenverkehr Kaltstart/Verdampfung (zone traffic, cold start/evaporation)
    ],
    # Shipping
    "GNFR_G": [
        "evsee",
        "evsfa",
        "evsrh",
    ],
    # Aviation
    "GNFR_H": [
        "evfgva",
        "evfzrh",
    ],
    # Offroad mobility
    "GNFR_I": [
        "ehmgh",
        "eibau",
        "eifrz",
        "eilpf",
        "eipis",
        "elfwm",
        "ellwm",
        "evsch",
        "evsra",
    ],
    # Waste
    "GNFR_J": [
        "eidep",  # Deponien
        "eikla",  # Kläranlagen
        "eikmp",  # Kompostierung
        "elabf",  # Abfallverbrennung Land- und Forstwirtschaft
        "elver",  # Vergärung
    ],
    # AgriLivestock
    "GNFR_K": [
        "elapp",
        "elsto",
    ],
    #  AgriOther
    "GNFR_L": [
        "elfer",
    ],
    # Others
    "GNFR_R": [
        "ehhab",  # Haushalte andere brande
        "ehhaf",  # Haushalte andere feuerwerk
        "ehhan",  # Haushalte andere private
        "enwal",  # Emissionen aus Waldern (emissions from forests)
    ],
}


TNO_2_GNFR = {
    # PublicPower
    "GNFR_A": [
        "A",
    ],
    # Industry
    "GNFR_B": [
        "B",
    ],
    # Other stationary combustion (services, residential, agriculture)
    "GNFR_C": [
        "C",
    ],
    # Fugitives
    "GNFR_D": [
        "D",
    ],
    # Solvents and product use
    "GNFR_E": [
        "E",
    ],
    # Road transport
    "GNFR_F": [
        "F1",
        "F2",
        "F3",
        "F4",
    ],
    # Shipping
    "GNFR_G": [
        "G",
    ],
    # Aviation
    "GNFR_H": [
        "H",
    ],
    # Offroad mobility
    "GNFR_I": [
        "I",
    ],
    # Waste
    "GNFR_J": [
        "J",
    ],
    # AgriLivestock
    "GNFR_K": [
        "K",
    ],
    # AgriOther
    "GNFR_L": [
        "L",
    ],
    # Others
    "GNFR_R": [],
}
