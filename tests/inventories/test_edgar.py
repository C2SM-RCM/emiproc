from emiproc import TESTS_DIR
from emiproc.inventories.edgar import download_edgar_files, EDGARv8
import pytest


edgar_test_dir = TESTS_DIR / "inventories" / "edgar_test"
edgar_test_dir.mkdir(exist_ok=True, parents=True)


def test_donwload_and_read():
    download_edgar_files(
        data_dir=edgar_test_dir,
        year=2020,
        categories=["ENE"],
        substances=["CO2"],
    )

    inv = EDGARv8(edgar_test_dir / "*.nc")
