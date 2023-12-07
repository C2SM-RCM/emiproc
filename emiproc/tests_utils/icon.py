import logging
from pathlib import Path
from urllib.request import urlretrieve

import xarray as xr

from emiproc import FILES_DIR
from emiproc.grids import ICONGrid

ICON_GRID_DIR = FILES_DIR / "test" / "icon_grid"

ICON_GRID_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


ICON_GRID_REPO = "http://icon-downloads.mpimet.mpg.de/grids/public/edzw/"

SIMPLE_ICON_GRID_PATH = ICON_GRID_DIR / "icon_grid_0055_R02B05_N.nc"


def download_test_grid(grid_filename: str):
    """Download a test grid from the ICON grid repository."""

    url = ICON_GRID_REPO + grid_filename
    logger.info("Downloading test grid from %s", url)
    urlretrieve(url, ICON_GRID_DIR / Path(url).name)


def get_test_grid(
    grid_path: Path = SIMPLE_ICON_GRID_PATH,
) -> ICONGrid:
    """Return the path of the test grid."""

    if not grid_path.exists():
        download_test_grid(grid_path.name)
    return ICONGrid(grid_path)


if __name__ == "__main__":
    print(get_test_grid())
