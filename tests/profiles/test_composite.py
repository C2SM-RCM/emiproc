"""Test composite profiles.

.. note:: Some of the composite test are also in other files.

"""

from __future__ import annotations

import numpy as np
from emiproc.profiles.temporal.profiles import TemporalProfile
from emiproc.profiles.temporal.composite import CompositeTemporalProfiles
from emiproc.tests_utils.temporal_profiles import TestProfile2, TestProfile3


def test_from_ratios():

    profile = CompositeTemporalProfiles.from_ratios(
        ratios=np.array([[0.1, 0.9, 0.3, 0.5, 0.2]]),
        types=[TestProfile2, TestProfile3],
    )


def test_from_rescale():

    profile = CompositeTemporalProfiles.from_ratios(
        ratios=np.array([[0.1, 0.2, 0.3, 0.5, 0.2]]),
        types=[TestProfile2, TestProfile3],
        rescale=True,
    )


def test_with_zero():
    """Test when one of the profile is zero."""
    profile = CompositeTemporalProfiles.from_ratios(
        ratios=np.array([[0.0, 0.0, 0.3, 0.5, 0.2]]),
        types=[TestProfile2, TestProfile3],
        rescale=True,
    )

    # Will be replaced by a constant profile
    # Test both, because it might be reversed
    assert (
        np.array_equal(profile.ratios, np.array([[0.5, 0.5, 0.3, 0.5, 0.2]]))
        and profile.types == [TestProfile2, TestProfile3]
    ) or (
        np.array_equal(profile.ratios, np.array([[0.3, 0.5, 0.2, 0.5, 0.5]]))
        and profile.types == [TestProfile3, TestProfile2]
    )
