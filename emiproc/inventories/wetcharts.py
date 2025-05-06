from os import PathLike

import geopandas as gpd
import pandas as pd
import xarray as xr

from emiproc.grids import RegularGrid
from emiproc.inventories import Inventory
from emiproc.profiles.temporal.composite import CompositeTemporalProfiles
from emiproc.profiles.temporal.profiles import MounthsProfile
from emiproc.profiles.utils import ratios_dataarray_to_profiles
from emiproc.utilities import DAY_PER_YR


class WetCHARTs(Inventory):
    """WetCHARTs inventory.

    This class is used to read the WetCHARTs file into an inventory.

    More information about the WetCHARTs is available at:
    https://daac.ornl.gov/CMS/guides/MonthlyWetland_CH4_WetCHARTsV2.html


    :param wetcharts_file: Path to the WetCHARTs file.
        It can be downloaded from the link above, after creating an account.
    :param model: Model number to select from the dataset.
        Few keys are available. To see the available models, please check
        the data specification in the link above.
        If None, a mean of all models is used.
    :param category: Category name to use in the inventory.
    :param substance: Substance name to use in the inventory.

    """

    def __init__(
        self,
        wetcharts_file: PathLike,
        model: int | None = None,
        category: str = "wetcharts",
        substance: str = "CH4",
    ) -> None:
        super().__init__()

        with xr.open_dataset(wetcharts_file) as ds:
            if model is None:
                ds = ds.mean(dim="model", keep_attrs=True)
            elif isinstance(model, int):
                if model not in ds.model.values:
                    raise ValueError(
                        f"Model {model} not found in the dataset. Available models: {ds.model.values}"
                    )
                ds = ds.sel(model=model)
            else:
                raise TypeError(f"Model {model} is not an integer or None.")

            self.grid = RegularGrid.from_centers(
                x_centers=ds.lon.values,
                y_centers=ds.lat.values,
                name="WetCHARTs_grid",
            )

            assert len(ds.time.values) == 12, "The dataset should have 12 months."

            self.year = pd.to_datetime(ds.time.values[0]).year

            var = "wetland_CH4_emissions"
            da_stacked = (
                ds[var]
                .stack(cell=("lon", "lat"))
                .drop_vars(["lon", "lat"])
                .expand_dims(dim=dict(category=[category], substance=[substance]))
                .fillna(0.0)
            )

        cell_areas = self.grid.cell_areas
        cell_areas = xr.DataArray(
            cell_areas.values, dims=["cell"], coords={"cell": da_stacked.cell.values}
        )

        # Unit conversion
        assert ds[var].units == "mg m-2 d-1"

        da_ratios = da_stacked / da_stacked.sum(dim="time")

        da_ratios = da_ratios.rename(time="ratio")
        ratios, indices = ratios_dataarray_to_profiles(da_ratios)

        # kg / year / cell = mg m-2 / day * m2 /cell  *  kg / mg * days / year
        da_stacked = da_stacked * cell_areas * 1e-6 * DAY_PER_YR

        # Convert to pandas
        df = (
            da_stacked.stack(catsub=("category", "substance"))
            .mean(dim="time")
            .drop_vars(["cell"])
            .to_pandas()
        )
        self.gdf = gpd.GeoDataFrame(df, geometry=self.grid.gdf.geometry)
        self.gdfs = {}

        self.set_profiles(
            profiles=CompositeTemporalProfiles.from_ratios(
                ratios=ratios, types=[MounthsProfile]
            ),
            indexes=indices,
        )
