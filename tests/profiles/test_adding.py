import numpy as np
import xarray as xr

from emiproc.profiles.operators import add_profiles
from emiproc.profiles.temporal.composite import CompositeTemporalProfiles
from emiproc.profiles.temporal.profiles import WeeklyProfile
from emiproc.tests_utils import test_inventories


def _weekly_profiles_from_ratios(ratios: np.ndarray) -> CompositeTemporalProfiles:
    return CompositeTemporalProfiles.from_ratios(
        ratios=ratios,
        types=[WeeklyProfile],
        rescale=True,
    )


def test_add_profiles_concatenates_disjoint_categories():
    inv1 = test_inventories.inv.copy()
    inv2 = test_inventories.inv.copy()

    profiles1 = _weekly_profiles_from_ratios(
        np.array(
            [
                [1, 1, 1, 1, 1, 1, 1],
                [1, 2, 3, 4, 5, 6, 7],
            ],
            dtype=float,
        )
    )
    profiles2 = _weekly_profiles_from_ratios(
        np.array(
            [
                [7, 6, 5, 4, 3, 2, 1],
                [2, 2, 2, 2, 2, 2, 2],
            ],
            dtype=float,
        )
    )

    indexes1 = xr.DataArray(
        data=np.array([[0]]),
        dims=["category", "substance"],
        coords={"category": ["adf"], "substance": ["CO2"]},
    )
    indexes2 = xr.DataArray(
        data=np.array([[1]]),
        dims=["category", "substance"],
        coords={"category": ["liku"], "substance": ["CO2"]},
    )

    inv1.set_profiles(profiles1, indexes=indexes1)
    inv2.set_profiles(profiles2, indexes=indexes2)

    new_profiles, new_indexes = add_profiles(inv1, inv2)

    expected_ratios = np.vstack([profiles1.ratios, profiles2.ratios])

    np.testing.assert_allclose(new_profiles.ratios, expected_ratios)
    np.testing.assert_array_equal(
        new_indexes.coords["category"], np.array(["adf", "liku"])
    )
    assert new_indexes.sel(category="adf", substance="CO2").item() == 0
    assert new_indexes.sel(category="liku", substance="CO2").item() == 3


def test_add_profiles_combines_overlapping_categories_with_emission_weights():
    inv1 = test_inventories.inv.copy()
    inv2 = test_inventories.inv.copy()

    # Give inventory 2 a larger emission total so weighted combination is asymmetric.
    inv2.gdf[("adf", "CH4")] *= 3

    profile_1 = np.array([[1, 1, 1, 1, 1, 1, 1]], dtype=float)
    profile_2 = np.array([[1, 2, 3, 4, 5, 6, 7]], dtype=float)

    profiles1 = _weekly_profiles_from_ratios(profile_1)
    profiles2 = _weekly_profiles_from_ratios(profile_2)

    indexes = xr.DataArray(
        data=np.array([[0]]),
        dims=["category", "substance"],
        coords={"category": ["adf"], "substance": ["CH4"]},
    )

    inv1.set_profiles(profiles1, indexes=indexes)
    inv2.set_profiles(profiles2, indexes=indexes)

    new_profiles, new_indexes = add_profiles(inv1, inv2)

    w1 = inv1.gdf[("adf", "CH4")].sum()
    w2 = inv2.gdf[("adf", "CH4")].sum()
    expected = (profiles1.ratios[0] * w1 + profiles2.ratios[0] * w2) / (w1 + w2)

    assert new_indexes.sel(category="adf", substance="CH4").item() == 0
    np.testing.assert_allclose(new_profiles.ratios[0], expected)


def test_adding_with_profiles_on_cells():
    """Test that adding two inventories with profiles on cells works correctly."""

    inv1 = test_inventories.inv.copy()
    inv2 = test_inventories.inv.copy()

    # Give inventory 2 a larger emission total so weighted combination is asymmetric.
    inv2.gdf[("adf", "CH4")] *= 3

    profile_1 = np.array([[1, 1, 1, 1, 1, 1, 1]], dtype=float)
    profile_2 = np.array([[1, 2, 3, 4, 5, 6, 7]], dtype=float)

    profiles1 = _weekly_profiles_from_ratios(profile_1)
    profiles2 = _weekly_profiles_from_ratios(profile_2)

    indexes = xr.DataArray(
        data=np.array([[[0] * 5]]),
        dims=["category", "substance", "cell"],
        coords={
            "category": ["adf"],
            "substance": ["CH4"],
            "cell": np.arange(5),
        },
    )

    inv1.set_profiles(profiles1, indexes=indexes)
    inv2.set_profiles(profiles2, indexes=indexes)

    new_profiles, new_indexes = add_profiles(inv1, inv2)

    w1 = inv1.gdf[("adf", "CH4")].to_numpy()
    w2 = inv2.gdf[("adf", "CH4")].to_numpy()
    # shape is (cell x week)
    expected = (
        profiles1.ratios * w1.reshape(-1, 1) + profiles2.ratios * w2.reshape(-1, 1)
    ) / (w1 + w2).reshape(-1, 1)

    output = new_profiles.ratios[
        new_indexes.sel(category="adf", substance="CH4").drop_vars(
            ["category", "substance"]
        )
    ]

    np.testing.assert_allclose(output, expected)


def test_adding_with_different_profiles():
    """Test that adding two inventories with profiles on cells works correctly."""

    inv1 = test_inventories.inv.copy()
    inv2 = test_inventories.inv.copy()

    # Give inventory 2 a larger emission total so weighted combination is asymmetric.
    inv2.gdf[("adf", "CH4")] *= 3

    profile_1 = np.array([[1, 1, 1, 1, 1, 1, 1]], dtype=float)
    profile_2 = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7],
            [1] * 7,
            [1, 2, -1, 1, 1, 1, 1],
        ],
        dtype=float,
    )

    profiles1 = _weekly_profiles_from_ratios(profile_1)
    profiles2 = _weekly_profiles_from_ratios(profile_2)

    indexes1 = xr.DataArray(
        data=np.array([[[0] * 5]]),
        dims=["category", "substance", "cell"],
        coords={
            "category": ["adf"],
            "substance": ["CH4"],
            "cell": np.arange(5),
        },
    )
    indexes2 = xr.DataArray(
        data=np.array([[[0, 1, 2, -1, -1]]]),
        dims=["category", "substance", "cell"],
        coords={
            "category": ["adf"],
            "substance": ["CH4"],
            "cell": np.arange(5),
        },
    )

    inv1.set_profiles(profiles1, indexes=indexes1)
    inv2.set_profiles(profiles2, indexes=indexes2)

    new_profiles, new_indexes = add_profiles(inv1, inv2)

    w1 = inv1.gdf[("adf", "CH4")].to_numpy()
    w2 = inv2.gdf[("adf", "CH4")].to_numpy()
    # shape is (cell x week)
    expected_ratios2 = np.concatenate(
        # Constant profiles where it is missing
        [profiles2.ratios, np.full((2, 7), 1.0 / 7.0)],
        axis=0,
    )
    expected = (
        profiles1.ratios * w1.reshape(-1, 1) + expected_ratios2 * w2.reshape(-1, 1)
    ) / (w1 + w2).reshape(-1, 1)

    output = new_profiles.ratios[
        new_indexes.sel(category="adf", substance="CH4").drop_vars(
            ["category", "substance"]
        )
    ]

    np.testing.assert_allclose(output, expected)


def test_adding_with_profiles_with_negative_values():
    """Test that adding two inventories with profiles on cells works correctly."""

    inv1 = test_inventories.inv.copy()
    inv2 = test_inventories.inv.copy()

    # Give inventory 2 negative total emission
    inv2.gdf[("adf", "CH4")] *= -3

    profile_1 = np.array([[1, 1, 1, 1, 1, 1, 1]], dtype=float)
    profile_2 = np.array([[1, -2, 3, 4, 5, -6, 7]], dtype=float)

    profiles1 = _weekly_profiles_from_ratios(profile_1)
    profiles2 = _weekly_profiles_from_ratios(profile_2)

    indexes = xr.DataArray(
        data=np.array([[[0] * 5]]),
        dims=["category", "substance", "cell"],
        coords={
            "category": ["adf"],
            "substance": ["CH4"],
            "cell": np.arange(5),
        },
    )

    inv1.set_profiles(profiles1, indexes=indexes)
    inv2.set_profiles(profiles2, indexes=indexes)

    new_profiles, new_indexes = add_profiles(inv1, inv2)

    w1 = inv1.gdf[("adf", "CH4")].to_numpy()
    w2 = inv2.gdf[("adf", "CH4")].to_numpy()
    # shape is (cell x week)
    expected = (
        profiles1.ratios * w1.reshape(-1, 1) + profiles2.ratios * w2.reshape(-1, 1)
    ) / (w1 + w2).reshape(-1, 1)

    output = new_profiles.ratios[
        new_indexes.sel(category="adf", substance="CH4").drop_vars(
            ["category", "substance"]
        )
    ]

    np.testing.assert_allclose(output, expected)


def test_add_profiles_different_category():
    """Test that adding two inventories with profiles on cells works correctly."""

    inv1 = test_inventories.inv.copy()
    inv2 = test_inventories.inv.copy()

    profile_1 = np.array([[1, 1, 1, 1, 1, 1, 1]], dtype=float)
    profile_2 = np.array([[1, -2, 3, 4, 5, -6, 7]], dtype=float)

    profiles1 = _weekly_profiles_from_ratios(profile_1)
    profiles2 = _weekly_profiles_from_ratios(profile_2)

    indexes1 = xr.DataArray(
        data=np.array([[[0] * 5]]),
        dims=["category", "substance", "cell"],
        coords={
            "category": ["adf"],
            "substance": ["CO2"],
            "cell": np.arange(5),
        },
    )
    indexes2 = xr.DataArray(
        data=np.array([[[0] * 5]]),
        dims=["category", "substance", "cell"],
        coords={
            "category": ["liku"],
            "substance": ["CO2"],
            "cell": np.arange(5),
        },
    )

    inv1.set_profiles(profiles1, indexes=indexes1)
    inv2.set_profiles(profiles2, indexes=indexes2)

    new_profiles, new_indexes = add_profiles(inv1, inv2)

    w1 = inv1.gdf[("adf", "CO2")].to_numpy()
    w2 = inv2.gdf[("liku", "CO2")].to_numpy()
    # shape is (category x cell x week)
    expected = np.stack(
        [
            profiles1.ratios * w1.reshape(-1, 1),
            profiles2.ratios * w2.reshape(-1, 1),
        ],
        axis=0,
    )

    output = new_profiles.ratios[
        new_indexes.sel(category=["adf", "liku"], substance="CO2").drop_vars(
            ["substance"]
        )
    ] * np.stack([w1.reshape(-1, 1), w2.reshape(-1, 1)], axis=0)

    np.testing.assert_allclose(output, expected)
