"""Mappings to regroup categories together."""

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
    "GNRF_D": [
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
    "GNFR_F": [
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
        "evfzhr",
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
    "GNFR_R": [
        "E",  # E is not in the zh or swiss to we set e to that
    ],
}
