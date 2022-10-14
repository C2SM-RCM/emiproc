"""Export functions for inventories."""


from datetime import datetime
from os import PathLike
from pathlib import Path
import xarray as xr
import numpy as np
from emiproc.grids import ICONGrid
from emiproc.inventories import Inventory
from emiproc.utilities import SEC_PER_YR, compute_country_mask, get_country_mask

from emiproc.country_code import country_codes


def export_icon_oem(
    inv: Inventory,
    icon_grid_file: PathLike,
    output_file: PathLike,
    group_dict: dict[str, list[str]] = {},
    country_resolution: str = "10m",
):
    """Export to a netcdf file for ICON OEM.

    The inventory should have already been remapped to the
    :py:class:`emiproc.grids.IconGrid` .

    Values will be convergted from kg/y to kg/m2/s .

    :arg group_dict: If you groupped some categories, you can optionally
        add the groupping in the metadata.
    :arg country_resolution: The resolution
        can be either '10m', '50m' or '110m'

    .. warning::

        Country codes are not yet implemented

    """
    icon_grid_file = Path(icon_grid_file)
    output_file = Path(output_file)
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

    # Find the proper contry codes
    mask_file = (
        output_file.parent
        / f".emiproc_country_mask_{country_resolution}_{icon_grid_file.stem}"
    ).with_suffix(".npy")
    if mask_file.is_file():
        country_mask = np.load(mask_file)
    else:
        icon_grid = ICONGrid(icon_grid_file)
        country_mask = compute_country_mask(icon_grid, country_resolution, 1)
        np.save(mask_file, country_mask)

    # Add the country ids variable for oem
    ds_out = ds_out.assign(
        {
            "country_ids": (
                ("cell"),
                country_mask.reshape(-1),
                {
                    "standard_name": "country_ids",
                    "long_name": "EMEP_country_code",
                    "history": f"Added by emiproc",
                    "country_resolution": f"country_resolution",
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
