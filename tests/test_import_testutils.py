from pathlib import Path

import pytest

import emiproc

# get the path of the emiproc module
emiproc_path = Path(*emiproc.__path__)
test_utils_path = emiproc_path / "tests_utils"

# List all the python files in the tests_utils directory
modules = [f.stem for f in test_utils_path.glob("*.py")]


@pytest.mark.parametrize(
    "module",
    [(m) for m in modules],
    ids=modules,
)
def test_import_test_module(module):
    # Call import on the module
    exec(f"import emiproc.tests_utils.{module}")
