"""Provides default values for how the data columns can be named."""


from datetime import datetime


accepted_category_colnames = [
    "Category",
    "GNFR_Category",
    "TNO GNFR sectors Sept 2018",
]
substances_colnames = [
    "Substance",
]
accepted_country_colnames = [
    "Country",
]

metadata_colnames = [
    "GNFR_Category_Name",
]
type_colnames = [
    "Type",
]

time_colnames = [
    "Year",
    "Month",
    "Day",
    "Hour",
]

cell_colnames = ["Cell"]

attributes_accepted_colnames = {
    "category": accepted_category_colnames,
    "substance": substances_colnames,
    "cell": cell_colnames,
    "time": time_colnames,
    "country": accepted_country_colnames,
    "type": type_colnames,
}

# Add lowercase versions of the accepted colnames
for key, value in attributes_accepted_colnames.items():
    attributes_accepted_colnames[key].extend([v.lower() for v in value])

type_of_dim = {
    "category": str,
    "substance": str,
    "cell": int,
    "time": datetime,
    "country": str,
    "type": str,
}

all_reserved_colnames = sum(
    [
        accepted_category_colnames,
        substances_colnames,
        accepted_country_colnames,
        metadata_colnames,
        type_colnames,
    ],
    [],
)
