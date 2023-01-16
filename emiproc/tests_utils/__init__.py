"""Utilities for testing emiproc."""
import emiproc
from pathlib import Path


WEIGHTS_DIR = Path(*emiproc.__path__) / ".." / "tests" / ".weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

TEST_OUTPUTS_DIR = Path(*emiproc.__path__) / ".." / "tests" / ".outputs"
TEST_OUTPUTS_DIR.mkdir(exist_ok=True)


if __name__ == "__main__":
    print(WEIGHTS_DIR)
    print(WEIGHTS_DIR.is_dir())