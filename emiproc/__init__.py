"""Emission processing package."""
from __future__ import annotations

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


def deprecated(msg: str | None = None):
    """Decorator to mark functions as deprecated."""

    def deprecated_decorator(func, msg=msg):
        def wrapper(*args, msg=msg, **kwargs):
            msg_default = "Call to deprecated function {}.".format(func.__name__)
            if msg is None:
                msg = msg_default
            else:
                msg = msg_default + " " + msg
            logger.warning(msg, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return deprecated_decorator
