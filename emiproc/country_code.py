# TODO: check if we need to use another source of data
# constants
country_codes = {
    "ALB": 1,
    "AUT": 2,
    "BEL": 3,
    "BGR": 4,
    "DNK": 6,
    "FIN": 7,
    "FRA": 8,
    "SMR": 8, #HJM
    "FGD": 9,
    "FFR": 10,
    "GRC": 11,
    "HUN": 12,
    "IRL": 14,
    "ITA": 15,
    "LUX": 16,
    "NLD": 17,
    "NOR": 18,
    "POL": 19,
    "PRT": 20,
    "ROM": 21,
    "ROU": 21,
    "ESP": 22,
    "AND": 22, # Andora -> ESP
    "SWE": 23,
    "CHE": 24,
    "TUR": 25,
    "GBR": 27,
    "IMN": 27, # Isle of Man -> GBR
    "GGY": 27, # Guernsey -> GBR
    "JEY": 27, # Jursey -> GBR
    "BLR": 39,
    "UKR": 40,
    "MKD": 41,
    "MDA": 42,
    "EST": 43,
    "LVA": 44,
    "LTU": 45,
    "CZE": 46,
    "SVK": 47,
    "SVN": 48,
    "HRV": 49,
    "BIH": 50,
    "YUG": 51,
    "GEO": 54,
    "MLT": 57,
    "DEU": 60,
    "RUS": 61,
    "ARM": 56,
    "AZE": 58,
    "CYP": 55,
    "ISL": 18,  # added for AQMEII, assumed to be same as Norway
    # ISO 2 (from EMEP site)
    "AL": 1,  # Albania
    "AT": 2,  # Austria
    "BE": 3,  # Belgium
    "BG": 4,  # Bulgaria
    "FCS": 5,  # Former Czechoslovakia
    "DK": 6,  # Denmark
    "FI": 7,  # Finland
    "ALD": 7,  # Aland, part of Finland
    "FR": 8,  # France
    "FGD": 9,  # Former German Democratic Republic
    "FFR": 10,  # Former Federal Republic of Germany
    "GR": 11,  # Greece
    "WSB": 11,  # Akrotiri/Santorini (Assigned to Greece)
    "HU": 12,  # Hungary
    "IS": 13,  # Iceland
    "IE": 14,  # Ireland
    "IT": 15,  # Italy
    "LU": 16,  # Luxembourg
    "NL": 17,  # Netherlands
    "NO": 18,  # Norway
    "PL": 19,  # Poland
    "PT": 20,  # Portugal
    "RO": 21,  # Romania
    "ES": 22,  # Spain
    "SMR": 22,  # San Marino (assigned to Spain)
    "AD": 22,  # Andorra (assigned to Spain)
    "GIB": 22,  # Gibraltar (assigned to Spain)
    "SE": 23,  # Sweden
    "CH": 24,  # Switzerland
    "TR": 25,  # Turkey
    "FSU": 26,  # Former USSR
    "GB": 27,  # United Kingdom
    "FRO": 27,  # Faroe Island (assigned to United Kingdom)
    "VOL": 28,  # Volcanic emissions
    "REM": 29,  # Remaining land Areas
    "BAS": 30,  # Baltic Sea
    "NOS": 31,  # North Sea
    "ATL": 32,  # Remaining North-East Atlantic Ocean
    "MED": 33,  # Mediterranean Sea
    "BLS": 34,  # Black Sea
    "NAT": 35,  # Natural marine emissions
    "RUO": 36,  # Kola & Karelia
    "RUP": 37,  # St.Petersburg & Novgorod-Pskov
    "RUA": 38,  # Kaliningrad
    "BY": 39,  # Belarus
    "UA": 40,  # Ukraine
    "MD": 41,  # Republic of Moldova
    "RUR": 42,  # Rest of the Russian Federation
    "EE": 43,  # Estonia
    "LV": 44,  # Latvia
    "LT": 45,  # Lithuania
    "CZ": 46,  # Czech Republic
    "SK": 47,  # Slovakia
    "SI": 48,  # Slovenia
    "HR": 49,  # Croatia
    "BA": 50,  # Bosnia and Herzegovina
    "CS": 51,  # Serbia and Montenegro
    "MK": 52,  # The former Yugoslav Republic of Macedonia
    "KZ": 53,  # Kazakhstan in the former official EMEP domain
    "GE": 54,  # Georgia
    "CY": 55,  # Cyprus
    "CYN": 55,  # Cyprus (alternate code)
    "CNM": 55,  # Cyprus (alternate code)
    "ESB": 55,  # Dhekelia Cantonment (assigned to Cyprus)
    "AM": 56,  # Armenia
    "MT": 57,  # Malta
    "ASI": 58,  # Remaining Asian areas
    "LI": 59,  # Liechtenstein,
    "LIE": 59,  # Liechtenstein (alternate code)
    "DE": 60,  # Germany
    "RU": 61,  # Russian Federation in the former official EMEP domain
    "MC": 62,  # Monaco
    "MCO": 62,  # Monaco
    "NOA": 63,  # North Africa
    "MAR": 63,  # Maroko (assigned to NOA)
    "TUN": 63,  # Tunisia (assigned to NOA)
    "DZA": 63,  # Algeria (assigned to NOA)
    "SYR": 63,  # Syria (assigned to NOA)
    "EU": 64,  # European Community
    "US": 65,  # United States
    "CA": 66,  # Canada
    "BIC": 67,  # Boundary and Initial Conditions
    "KG": 68,  # Kyrgyzstan
    "AZ": 69,  # Azerbaijan
    "ATX": 70,  # EMEP-external Remaining North-East Atlantic Ocean
    "RUX": 71,  # EMEP-external part of Russian Federation
    "RS": 72,  # Serbia
    "SRB": 72,  # Serbia (alternate code)
    "KOS": 72,  # Kosovo (assigned to SRB)
    "ME": 73,  # Montenegro
    "MNE": 73,  # Montenegro (alternate code)
    "RFE": 74,  # Rest of Russian Federation in the extended EMEP domain
    "KZE": 75,  # Rest of Kazakhstan in the extended EMEP domain
    "UZO": 76,  # Uzbekistan in the former official EMEP domain
    "TMO": 77,  # Turkmenistan in the former official EMEP domain
    "UZE": 78,  # Rest of Uzbekistan in the extended EMEP domain
    "TME": 79,  # Rest of Turkmenistan in the extended EMEP domain
    "CAS": 80,  # Caspian Sea
    "TJ": 81,  # Tajikistan
    "ARO": 82,  # Aral Lake in the former official EMEP domain
    "ARE": 83,  # Rest of Aral Lake in the extended EMEP domain
    "ASM": 84,  # Modified Remaining Asian Areas in the former official EMEP domain
    "ASE": 85,  # Remaining Asian Areas in the extended EMEP domain
    "AOE": 86,  # Arctic Ocean in the extended EMEP domain
    "RFX": 87,  # Extended EMEP External Part of Russian Federation
    "ASX": 88,  # Extended EMEP External Part of Asia
    "PAX": 89,  # Extended EMEP External Part of Pacific Ocean
    "AOX": 90,  # Extended EMEP External Part of Arctic Ocean
    "NAX": 91,  # Extended EMEP External Part of North Africa
    "KZT": 92,  # Kazakhstan
    "RUE": 93,  # Russian Federation in the extended EMEP domain (RU+RFE+RUX)
    "UZ": 94,  # Uzbekistan
    "TM": 95,  # Turkmenistan
    "AST": 96,  # Asian areas in the extended EMEP domain (ASM+ASE+ARO+ARE+CAS)
    "FYU": 99,  # Former Yugoslavia
    # Added for emiproc 
    "ISR": 200, # Israel
    "LBN": 201, # Lebanon
    "LBY": 202, # Libya
    "VAT": 203, # Vatican City
    "ETH": 204, # Ethiopia
    "SOM": 205, # Somalia
    "SOL": 206, # ???
    "UZB": 207, # Uzbekistan
    "KAZ": 208, # Kazakhstan
    "TKM": 209, # Turkmenistan
    "KGZ": 210, # Kyrgyzstan
    "TJK": 211, # Tajikistan
    "IRN": 212, # Iran
    "IRQ": 213, # Iraq
    # End of added for emiproc
    "BEF": 301,  # Belgium (Flanders)
    "BA2": 302,  # Baltic Sea EU Cargo o12m
    "BA3": 303,  # Baltic Sea ROW Cargo o12m
    "BA4": 304,  # Baltic Sea EU Cargo i12m
    "BA5": 305,  # Baltic Sea ROW Cargo i12m
    "BA6": 306,  # Baltic Sea EU Ferry o12m
    "BA7": 307,  # Baltic Sea ROW Ferry o12m
    "BA8": 308,  # Baltic Sea EU Ferry i12m
    "BA9": 309,  # Baltic Sea ROW Ferry i12m
    "NO2": 312,  # North Sea EU Cargo o12m
    "NO3": 313,  # North Sea ROW Cargo o12m
    "NO4": 314,  # North Sea EU Cargo i12m
    "NO5": 315,  # North Sea ROW Cargo i12m
    "NO6": 316,  # North Sea EU Ferry o12m
    "NO7": 317,  # North Sea ROW Ferry o12m
    "NO8": 318,  # North Sea EU Ferry i12m
    "NO9": 319,  # North Sea ROW Ferry i12m
    "AT2": 322,  # Remaining North-East Atlantic Ocean EU Cargo     'o1': 2 # m
    "AT3": 323,  # Remaining North-East Atlantic Ocean ROW Cargo     'o1': 2 # m
    "AT4": 324,  # Remaining North-East Atlantic Ocean EU Cargo     'i1': 2 # m
    "AT5": 325,  # Remaining North-East Atlantic Ocean ROW Cargo     'i1': 2 # m
    "AT6": 326,  # Remaining North-East Atlantic Ocean EU Ferry     'o1': 2 # m
    "AT7": 327,  # Remaining North-East Atlantic Ocean ROW Ferry     'o1': 2 # m
    "AT8": 328,  # Remaining North-East Atlantic Ocean EU Ferry     'i1': 2 # m
    "AT9": 329,  # Remaining North-East Atlantic Ocean ROW Ferry     'i1': 2 # m
    "ME2": 332,  # Mediterranean Sea EU Cargo o12m
    "ME3": 333,  # Mediterranean Sea ROW Cargo o12m
    "ME4": 334,  # Mediterranean Sea EU Cargo i12m
    "ME5": 335,  # Mediterranean Sea ROW Cargo i12m
    "ME6": 336,  # Mediterranean Sea EU Ferry o12m
    "ME7": 337,  # Mediterranean Sea ROW Ferry o12m
    "ME8": 338,  # Mediterranean Sea EU Ferry i12m
    "ME9": 339,  # Mediterranean Sea ROW Ferry i12m
    "BL2": 342,  # Black Sea EU Cargo o12m
    "BL3": 343,  # Black Sea ROW Cargo o12m
    "BL4": 344,  # Black Sea EU Cargo i12m
    "BL5": 345,  # Black Sea ROW Cargo i12m
    "BL6": 346,  # Black Sea EU Ferry o12m
    "BL7": 347,  # Black Sea ROW Ferry o12m
    "BL8": 348,  # Black Sea EU Ferry i12m
    "BL9": 349,  # Black Sea ROW Ferry i12m
    "GL": 601,  # Greenland
}

# Mapping from code to iso3
code_2_iso3 = {
    code: iso3 for iso3, code in country_codes.items() if len(iso3) == 3
}
