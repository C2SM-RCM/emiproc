from emiproc import TESTS_DIR
from emiproc.inventories.edgar import download_edgar_files, EDGARv8
import pytest
from emiproc.inventories.edgar.temporal import read_edgar_auxilary_profiles


edgar_test_dir = TESTS_DIR / "inventories" / "edgar_test"
edgar_test_dir.mkdir(exist_ok=True, parents=True)


@pytest.fixture(scope="module")
def download():
    download_edgar_files(
        data_dir=edgar_test_dir,
        year=2020,
        categories=["ENE"],
        substances=["CO2"],
    )


def test_read(download):

    inv = EDGARv8(edgar_test_dir / "*.nc")


def test_edgar_auxiliary_profiles(download):
    inv = EDGARv8(edgar_test_dir / "*.nc", use_short_category_names=True)

    profiles, indexes = read_edgar_auxilary_profiles(
        auxiliary_filesdir=edgar_test_dir / "auxiliary_profiles",
        inventory=inv,
    )

    assert "ENE" in indexes["category"].values

    assert "SDN" in indexes["country"].values

    assert len(profiles) == max(indexes) + 1
