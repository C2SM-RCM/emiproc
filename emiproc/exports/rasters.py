

from os import PathLike
from pathlib import Path
import xarray as xr
import numpy as np
from emiproc.inventories import Inventory





def export_netcdf(inv: Inventory, path: PathLike, ):
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
