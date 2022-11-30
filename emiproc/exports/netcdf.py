from datetime import datetime


def nc_cf_attributes(
    author: str,
    contact: str,
    title: str,
    source: str,
    comment: str,
    institution: str = "Empa, Swiss Federal Laboratories for Materials Science and Technology",
    history: str = "",
    references: str = "Produced by emiproc.",
    additional_attributes: dict[str, str] = {},
):
    """Create attributes for a nc file based on cf conventions.


    Most of the following instructions at
    https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#_attributes

    Some fields (author, contact) were also added for simplifying data sharing.
    :attr author: The author of the file.
    :attr contact: The contact address (email) of a person that can be contacted
        if one has questions about the file.
    :attr title:
        A succinct description of what is in the dataset.
    :attr institution:
        Specifies where the original data was produced.
    :attr source:
        The method of production of the original data. If it was model-generated, source should name the model and its version, as specifically as could be useful. If it is observational, source should characterize it (e.g., "surface observation" or "radiosonde").
    :attr history:
        Provides an audit trail for modifications to the original data.
        Well-behaved generic netCDF filters will automatically append their name and the parameters with which they were invoked to the global history attribute of an input netCDF file
        We recommend that each line begin with a timestamp indicating the date and time of day that the program was executed.
    :attr references:
        Published or web-based references that describe the data or methods used to produce it.
    :attr comment:
        Miscellaneous information about the data or methods used to produce it.
    :attr additional_attributes:
        Any attribute you want to add the thing.

    """
    return {
        "Conventions": "CF-1.10",
        "title": title,
        "comment": comment,
        "source": source,
        "history": history,
        "references": references,
        "institution": institution,
        "author": author,
        "contact": contact,
        "creation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **additional_attributes,
    }
