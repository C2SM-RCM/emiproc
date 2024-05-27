"""Mappings to regroup categories together."""
CH_2_GNFR_v2 = {
 # PublicPower
    "A": [
        # Waste incinerator we agreed on nov. 22 to put them in the public power category
        "eipkv",  # Punktquellen KVA (Kehrichtverbrennungsanlagen ) == Waste incinerators
    ],
    # Industry
    "B": [
        "eipro",
        "eipwp",  # this is the weitere punktquelle (additional point sources)
        "eipzm",
    ],
    # Other stationary combustion (services, residential, agriculture)
    "C": [
        "ehare",
        "ehfho",
        "ehfoe",
        "ehgws",
        "eipdh",
        "eiprd",
        "elfeu",
    ],
    # Fugitives
    "D": [
        "eilgk",
        "eivgn",
        "evklm",
        "evtrk",
    ],
    # Solvents and product use
    "E": [
        "eilmi",  # Lösungsmittel Industrie
        "ehlmk", # Lösungsmittel Konsumprodukte
    ],
    # Road transport
    #"GNFR_F": [
    #    # "evstr_ch4",
    #    # "evstr_co",
    #    # "evstr_co2",
    #    # "evstr_n2o",
    #    # "evstr_nh3",
    #    # "evstr_nmvoc",
    #    # "evstr_nox",
    #    # "evstr_so2",
    #    "evstr",
    #    "evzon",
    #],

    "F1": [
        "evstrf1",
        "evzonf1"
    ],

    "F2": [
        "evstrf2",
        "evzonf2"
    ],

    "F3": [
        "evstrf3",
        "evzonf3"
    ],

    "F4": [
        "evstrf4",
        "evzonf4"
    ],

    # Shipping
    "G": [
        "evsee",
        "evsfa",
        "evsrh",
    ],
    # Aviation
    "H": [
        "evfgva",
        "evfzrh",
    ],
    # Offroad mobility
    "I": [
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
    "J": [
        "eidep",
        "eikla",
        "eikmp",
        "elabf",
        "elver",
    ],
    # AgriLivestock
    "K": [
        "elapp",
        "elsto",
    ],
    #  AgriOther
    "L": [
        "elfer",
    ],
    # Others
    "R": [
        "ehhab",
        "ehhaf",
        "ehhan",
        "enwal",
    ],
}

CH_2_GNFR = {
    # PublicPower
    "GNFR_A": [
        # Waste incinerator we agreed on nov. 22 to put them in the public power category
        "eipkv",  # Punktquellen KVA (Kehrichtverbrennungsanlagen ) == Waste incinerators
    ],
    # Industry
    "GNFR_B": [
        "eipro",
        "eipwp",  # this is the weitere punktquelle (additional point sources)
        "eipzm",
    ],
    # Other stationary combustion (services, residential, agriculture)
    "GNFR_C": [
        "ehare",
        "ehfho",
        "ehfoe",
        "ehgws",
        "eipdh",
        "eiprd",
        "elfeu",
    ],
    # Fugitives
    "GNFR_D": [
        "eilgk",
        "eivgn",
        "evklm",
        "evtrk",
    ],
    # Solvents and product use
    "GNFR_E": [
        "eilmi",  # Lösungsmittel Industrie
        "ehlmk", # Lösungsmittel Konsumprodukte
    ],
    # Road transport
    #"GNFR_F": [
    #    # "evstr_ch4",
    #    # "evstr_co",
    #    # "evstr_co2",
    #    # "evstr_n2o",
    #    # "evstr_nh3",
    #    # "evstr_nmvoc",
    #    # "evstr_nox",
    #    # "evstr_so2",
    #    "evstr",
    #    "evzon",
    #],

    "GNFR_F1": [
        "evstrf1",
        "evzonf1"
    ],

    "GNFR_F2": [
        "evstrf2",
        "evzonf2"
    ],

    "GNFR_F3": [
        "evstrf3",
        "evzonf3"
    ],

    "GNFR_F4": [
        "evstrf4",
        "evzonf4"
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
        "eidep",
        "eikla",
        "eikmp",
        "elabf",
        "elver",
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
        "ehhab",
        "ehhaf",
        "ehhan",
        "enwal",
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

TNO_2_GNFR_v2 = {
    # PublicPower
    "A": [
        "A",
    ],
    # Industry
    "B": [
        "B",
    ],
    # Other stationary combustion (services, residential, agriculture)
    "C": [
        "C",
    ],
    # Fugitives
    "D": [
        "D",
    ],
    # Solvents and product use
    "E": [
        "E",
    ],
    # Road transport
    "F1": [
        "F1",
    ],

    "F2": [
        "F2",
    ],
    
    "F3": [
        "F3",
    ],

    "F4": [
        "F4",
    ],
    # Shipping
    "G": [
        "G",
    ],
    # Aviation
    "H": [
        "H",
    ],
    # Offroad mobility
    "I": [
        "I",
    ],
    # Waste
    "J": [
        "J",
    ],
    # AgriLivestock
    "K": [
        "K",
    ],
    # AgriOther
    "L": [
        "L",
    ],
    # Others
    "R": [],
}

