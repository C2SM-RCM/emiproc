
import numpy as np

from emiproc.profiles.vertical_profiles import VerticalProfiles, VerticalProfile

def weighted_combination(profiles: VerticalProfiles, weights: np.ndarray) -> VerticalProfile:
    """Combine the different profiles according to the specified weights."""
    
    return VerticalProfile(
        np.average(profiles.ratios, axis=0, weights=weights),
        profiles.height.copy()
    )
    