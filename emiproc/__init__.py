"""Emission processing package."""

import logging
from pathlib import Path

# directory where the data files are stored
FILES_DIR = Path(__file__).parent.parent / "files"
TESTS_DIR = FILES_DIR / "test"

logger = logging.getLogger("emiproc")

# Create a dedicated logging level for processes
PROCESS = 25  # between INFO (20) and WARNING (30)
logging.addLevelName(PROCESS, "PROCESS")

logger.setLevel(PROCESS)


def deprecated(func):
    """Decorator to mark functions as deprecated."""

    def wrapper(*args, **kwargs):
        logger.warning(
            "Call to deprecated function {}.".format(func.__name__),
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return wrapper
