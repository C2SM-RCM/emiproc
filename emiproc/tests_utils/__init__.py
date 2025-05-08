"""Utilities for testing emiproc."""
import emiproc
from pathlib import Path

TEST_DIR = Path(*emiproc.__path__) / ".." / "tests"
TEST_DIR.mkdir(exist_ok=True)

TESTFILES_DIR = emiproc.TESTS_DIR

WEIGHTS_DIR = TEST_DIR / ".weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

TEST_OUTPUTS_DIR = TEST_DIR / ".outputs"
TEST_OUTPUTS_DIR.mkdir(exist_ok=True)


if __name__ == "__main__":
    print(WEIGHTS_DIR)
    print(WEIGHTS_DIR.is_dir())
