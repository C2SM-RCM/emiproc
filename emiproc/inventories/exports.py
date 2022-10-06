"""Export functions for inventories."""


from datetime import datetime
from os import PathLike
from pathlib import Path
import xarray as xr
import numpy as np
from emiproc.inventories import Inventory
from emiproc.utilities import SEC_PER_YR

from emiproc.country_code import country_codes


def export_icon_oem(
    inv: Inventory,
    icon_grid_file: Path,
    output_file: Path,
    group_dict: dict[str, list[str]] = {},
):
    """Export to a netcdf file for OEM.

    The inventory should have already been remapped to the icon_grid.

    Values will be convergted from kg/year to kg/m2/s
    """

    # Load the output xarray

    ds_out: xr.Dataset = xr.load_dataset(icon_grid_file)

    for (categorie, sub) in inv._gdf_columns:
        name = f"{categorie}-{sub}"

        # Convert from kg/year to kg/m2/s
        emissions = (
            inv.gdf[(categorie, sub)].to_numpy() / ds_out["cell_area"] / SEC_PER_YR
        )

        attributes = {
            "units": "kg/m2/s",
            "standard_name": name,
            "long_name": f"Emission of {sub} from {categorie}",
            "created_by_emiproc": f"{datetime.now()}",
        }
        if group_dict:
            attributes["group_made_from"] = f"{group_dict[categorie]}"

        emission_with_metadata = emissions.assign_attrs(attributes)

        ds_out = ds_out.assign({name: emission_with_metadata})

    

    # Add the country ids variable for oem
    ds_out = ds_out.assign(
        {
            "country_ids": (
                ("cell"),
                # Dummy value for switzerland TODO: use proper contry codes
                np.full(len(inv.gdf), country_codes["CH"]),
                {
                    "standard_name": "country_ids",
                    "long_name": f"Id of the country",
                    "history": "Created such as country_ids==cell ",
                },
            )
        }
    )

    ds_out.to_netcdf(output_file)


def export_netcdf(inv: Inventory, path: PathLike):
    """Export to a netcdf file.

    # TODO: add the grid
    """
    n_cells = len(inv.gdf)
    ds = xr.Dataset(
        data_vars={
            "emissions": (
                ("substance", "category", "ncells"),
                [
                    [
                        inv.gdf[(cat, sub)]
                        if (cat, sub) in inv.gdf
                        else np.full(n_cells, np.nan)
                        for cat in inv.categories
                    ]
                    for sub in inv.substances
                ],
                {
                    "name": "emissions",
                    "unit": "kg/y",
                    "history": str(inv.history),
                },
            )
        },
        coords={
            "substance": inv.substances,
            "category": inv.categories,
        },
    )
    ds.to_netcdf(Path(path).with_suffix(".nc"))
