import logging
from os import PathLike
from pathlib import Path
from urllib.error import HTTPError
import xarray as xr
import numpy as np
import re
import geopandas as gpd
import pyogrio

import urllib.request
import zipfile

from emiproc.grids import WGS84, EDGARGrid, RegularGrid
from emiproc.inventories import Inventory
from emiproc.utilities import SEC_PER_YR


def download_edgar_files(
    data_dir: PathLike,
    year: int,
    categories: list[str] = [
        "ENE",
        "REF_TRF",
        "IND",
        "TNR_Aviation_CDS",
        "TNR_Aviation_CRS",
        "TNR_Aviation_LTO",
        "TNR_Aviation_SPS",
        "TRO",
        "TNR_Other",
        "TNR_Ship",
        "RCO",
        "PRO_FFF",
        "NMM",  # Non-metallic minerals production
        "CHE",  # Chemical processes
        "IRO",  # Iron and steel production
        "NFE",  # Non-ferrous metals production
        "NEU",  # Non-energy use of fuels
        "PRU_SOL",  # Solvents and products use
        # Agriculture
        "ENF",  # Enteric fermentation
        "MNM",  # Manure management
        "AWB",  # Agricultural waste burning
        "AGS",  # Agricultural soils
        "N2O",  # Indirect N2O emissions from agriculture
        "SWD_LDF",  # Solid waste landfills
        "SWD_INC",  # Solid waste incineration
        "WWT",  # Waste water handling
        "IDE",  # Indirect emissions from NOx and NH3
    ],
    substances: list[str] = [
        "CO2",
        "CH4",
        "N2O",
        "CO2bio",
        "GWP_100_AR5_GHG",
    ],
    link_template: str = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EDGAR/datasets/v80_FT2022_GHG/{substance}/{category}/emi_nc/v8.0_FT2022_GHG_{substance}_{year}_{category}_emi_nc.zip",
):
    download_links = [
        link_template.format(substance=substance, category=category, year=year)
        for substance in substances
        for category in categories
    ]

    downloaded = []

    # Download the files
    for link in download_links:
        filename = link.split("/")[-1]
        filepath = data_dir / filename
        try:
            urllib.request.urlretrieve(link, filepath)
        except HTTPError as e:
            # Link does not exist
            continue

        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(data_dir)

        downloaded.append(filepath)
        filepath.unlink()

    if not downloaded:
        raise ValueError(
            f"No files were downloaded for {substances=} {year=} {categories=}."
            "Check the links."
        )

    print(f"Downloaded {len(downloaded)} files.")


class EDGARv8(Inventory):
    """The EDGAR inventory.

    `Emissions Database for Global Atmospheric Research <https://edgar.jrc.ec.europa.eu/>`_


    The files are freely available.
    There are different versions. The one we use here is the
    `Annual sector-specific gridmaps` and we scale them over the months using the
    `Monthly time series` files.

    Download the files you want to include and provide the path to the directory
    containing the files.

    EDGAR has only grid cell sources.
    """

    grid: EDGARGrid

    def __init__(
        self,
        nc_file_pattern_or_dir: PathLike,
        year: int | None = None,
        use_short_category_names: bool = False,
    ) -> None:
        """Create a EDGAR_Inventory.

        :arg nc_file_pattern_or_dir: Pattern or directory of files.
        :arg year: Year of the inventory.
        :arg use_short_category_names: Use short category names.

        """
        super().__init__()

        nc_file_pattern = Path(nc_file_pattern_or_dir)
        logger = logging.getLogger(__name__)

        if nc_file_pattern.is_dir():
            nc_file_pattern = nc_file_pattern / "*.nc"

        columns = {}

        file_for_grid = None

        for filepath in nc_file_pattern.parent.glob(nc_file_pattern.name):
            with xr.open_dataset(filepath) as ds:
                if "emissions" not in ds:
                    logger.warning(
                        f"File {filepath} does not contain 'emissions' variable. Skipping."
                    )
                    continue
                da = ds["emissions"]
                substance = da.attrs["substance"]
                category = da.attrs["long_name"]
                if use_short_category_names:
                    # Read from the path string
                    category = "_".join(filepath.stem.split("_")[5:-1])

                file_year = int(da.attrs["year"])
                if year is None:
                    year = file_year
                if file_year != year:
                    logger.warning(
                        f"File {filepath} has year {file_year}, expected {year}. Skipping."
                    )
                    continue
                units = da.attrs["units"]
                assert units == "Tonnes", f"Units are {units}, expected `Tonnes`."

                columns[(category, substance)] = da.data.T.reshape(-1)

            if file_for_grid is None:
                file_for_grid = filepath

        with xr.open_dataset(file_for_grid) as ds:
            self.grid = RegularGrid.from_centers(
                ds["lon"].data, ds["lat"].data, name="EDGARv8_grid"
            )

        # Swtich longitude from 0/360 to -180/180
        self.gdf = gpd.GeoDataFrame(
            data={
                # Convert from tonnes/yr to kg/yr
                col: value * 1e3
                for col, value in columns.items()
            },
            geometry=self.grid.gdf.geometry,
            crs=WGS84,
        )

        # Empty shaped sources
        self.gdfs = {}


class EDGAR_Inventory(Inventory):
    """The EDGAR inventory.

    `Emissions Database for Global Atmospheric Research <https://edgar.jrc.ec.europa.eu/>`_

    EDGAR has only grid cell sources.
    """

    grid: EDGARGrid

    def __init__(
        self, nc_file_pattern: PathLike, grid_shapefile: PathLike | None = None
    ) -> None:
        """Create a EDGAR_Inventory.

        The EDGAR directory that contains the original datasets must be structured as ./substance/categories/dataset,
        e.g., ./SF6/NFE/v7.0_FT2021_SF6_*_NFE.0.1x0.1.nc and ./SF6/PRU/v7.0_FT2021_SF6_*_PRU.0.1x0.1.nc

        :arg nc_file_pattern: Pattern of files, e.g "EDGAR/SF6/PRU/v7.0_FT2021_SF6_*_PRU.0.1x0.1.nc"

        """
        super().__init__()

        nc_file_pattern = Path(nc_file_pattern)

        self.name = nc_file_pattern.stem

        list_filepaths = list(nc_file_pattern.parent.glob(nc_file_pattern.name))
        list_ds = [xr.open_dataset(f) for f in list_filepaths]

        substance = list_filepaths[0].parent.parent.stem

        substances_mapping = {}
        categories = [
            re.search(r"{}_(.+?)_(.+?)\.".format(substance), filepath.name)[2]
            for filepath in list_filepaths
        ]

        for i, ds in enumerate(list_ds):
            old_varname = list(ds.data_vars.keys())[0]
            new_varname = substance.upper()
            list_ds[i] = ds.rename_vars({old_varname: new_varname})
            substances_mapping[new_varname] = new_varname.upper()

        ds = xr.merge(list_ds)

        # Swtich longitude from 0/360 to -180/180
        attributes = ds.lon.attrs
        ds = ds.assign(lon=(ds.lon + 180) % 360 - 180).sortby("lon")
        ds.lon.attrs = attributes

        self.grid = EDGARGrid(list_filepaths[0])

        if grid_shapefile is None:
            polys = self.grid.cells_as_polylist
        else:
            gdf_geometry = gpd.read_file(grid_shapefile, engine="pyogrio")
            gdf_geometry = gdf_geometry.to_crs(WGS84)
            polys = gdf_geometry.geometry.values

        # Index in the polygon list (from the gdf) (index start at 1)
        poly_ind = np.arange(self.grid.nx * self.grid.ny)

        self.gdfs = {}
        mapping = {}
        for cat_idx, cat_name in enumerate(categories):

            for sub_in_nc, sub_emiproc in substances_mapping.items():
                tuple_idx = (cat_name, sub_emiproc)
                if tuple_idx not in mapping:
                    mapping[tuple_idx] = np.zeros(len(polys))

                # Add all the area sources corresponding to that category
                np.add.at(mapping[tuple_idx], poly_ind, ds[sub_in_nc].data.T.flatten())

        self.gdf = gpd.GeoDataFrame(
            mapping,
            geometry=polys,
            crs=WGS84,
        )

        self.cell_areas = self.grid.cell_areas

        # -- Convert to kg/yr
        self.gdf[list(mapping)] *= SEC_PER_YR * self.cell_areas[:, np.newaxis]
