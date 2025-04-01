"""Temporal profiles.

This is an old deprecated module.
All functions are now moved to the temporal subfolder.
"""

import logging
import emiproc
from emiproc.profiles.temporal.profiles import *
from emiproc.profiles.temporal.composite import *
from emiproc.profiles.temporal.constants import *
from emiproc.profiles.temporal.io import *
from emiproc.profiles.temporal.operators import *
from emiproc.profiles.temporal.specific_days import *
from emiproc.profiles.temporal.utils import *


logger = logging.getLogger(__name__)


@emiproc.deprecated(
    "`emiproc.profiles.temporal_profiles` is deprecated. "
    "functions are now available at `emiproc.profiles.temporal`"
)
def show_deprecation():
    pass


show_deprecation()
