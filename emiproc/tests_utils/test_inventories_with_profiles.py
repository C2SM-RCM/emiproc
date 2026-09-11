"""Test inventories with temporal profiles."""

from emiproc.profiles.temporal.composite import CompositeTemporalProfiles
from emiproc.tests_utils.temporal_profiles import indexes_inv_catsub, three_profiles
from emiproc.tests_utils.test_inventories import inv

weekly_and_monthly_profiles = CompositeTemporalProfiles(
    three_profiles + [three_profiles[0]]
)
monthly_profiles = CompositeTemporalProfiles(
    [[profiles[1]] for profiles in three_profiles] + [[three_profiles[0][1]]]
)

inv_with_weekly_and_monthly_profiles = inv.copy()
inv_with_weekly_and_monthly_profiles.set_profiles(
    weekly_and_monthly_profiles, indexes=indexes_inv_catsub
)

inv_with_monthly_profiles = inv.copy()
inv_with_monthly_profiles.set_profiles(monthly_profiles, indexes=indexes_inv_catsub)

# Set a year
for inv_ in [inv_with_weekly_and_monthly_profiles, inv_with_monthly_profiles]:
    inv_.year = 2018

# Exposed inventories for testing
inv_with_weekly_and_monthly_profiles
inv_with_monthly_profiles
