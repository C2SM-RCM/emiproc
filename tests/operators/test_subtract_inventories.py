"""Tests for the operators that avoid double counting emissions."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from emiproc.grids import RegularGrid, WGS84
from emiproc.inventories import Inventory
from emiproc.inventories.utils import (
    get_total_emissions,
    merge_inventories_without_double_counting,
    subtract_inventories,
)


@pytest.fixture
def grid() -> RegularGrid:
    return RegularGrid(xmin=0, ymin=0, xmax=2, ymax=2, nx=2, ny=2, name="g", crs=WGS84)


@pytest.fixture
def gridded(grid: RegularGrid) -> Inventory:
    inv = Inventory.from_gdf(
        gpd.GeoDataFrame(
            {
                ("ene", "CO2"): [100.0, 200.0, 300.0, 400.0],
                ("ind", "CO2"): [10.0, 20.0, 30.0, 40.0],
            },
            geometry=grid.gdf.geometry,
            crs=WGS84,
        ),
        name="gridded",
    )
    inv.grid = grid
    inv.year = 2025
    return inv


def _points(values: list[float], category: str = "power") -> Inventory:
    inv = Inventory()
    inv.name = "points"
    inv.year = 2025
    inv.gdf = None
    inv.gdfs = {
        category: gpd.GeoDataFrame(
            {"co2": values},
            geometry=[Point(0.5, 0.5), Point(1.5, 1.5)][: len(values)],
            crs=WGS84,
        )
    }
    return inv


def test_subtract_removes_the_points_from_the_right_cells(gridded: Inventory) -> None:
    out = subtract_inventories(
        gridded,
        _points([50.0, 25.0]),
        category_mapping={"power": "ene"},
        substance_mapping={"co2": "CO2"},
    )

    np.testing.assert_allclose(
        out.gdf[("ene", "CO2")].to_numpy(), [50.0, 200.0, 300.0, 375.0]
    )
    # The untouched category keeps its emissions.
    np.testing.assert_allclose(
        out.gdf[("ind", "CO2")].to_numpy(), [10.0, 20.0, 30.0, 40.0]
    )
    assert out.subtraction_summary["removed_kg_y"]["ene/CO2"] == pytest.approx(75.0)
    assert out.subtraction_summary["clipped_values"] == 0


def test_merge_conserves_the_total_and_keeps_the_point_geometry(
    gridded: Inventory,
) -> None:
    points = _points([50.0, 25.0])
    before = get_total_emissions(gridded)["CO2"]["__total__"]

    merged = merge_inventories_without_double_counting(
        gridded,
        points,
        category_mapping={"power": "ene"},
        substance_mapping={"co2": "CO2"},
    )

    after = get_total_emissions(merged)["CO2"]["__total__"]
    assert after == pytest.approx(before)
    # The detailed sources keep their own geometry.
    assert "ene" in merged.gdfs
    assert len(merged.gdfs["ene"]) == 2


def test_merge_can_put_the_points_on_the_grid(gridded: Inventory) -> None:
    merged = merge_inventories_without_double_counting(
        gridded,
        _points([50.0, 25.0]),
        category_mapping={"power": "ene"},
        substance_mapping={"co2": "CO2"},
        keep_detailed_shapes=False,
    )

    assert not merged.gdfs
    np.testing.assert_allclose(
        merged.gdf[("ene", "CO2")].to_numpy(), [100.0, 200.0, 300.0, 400.0]
    )


@pytest.mark.parametrize(
    "policy, expected",
    [("clip", 0.0), ("allow", 100.0 - 500.0)],
)
def test_negative_policies(gridded: Inventory, policy: str, expected: float) -> None:
    points = _points([500.0])
    if policy == "clip":
        with pytest.warns(RuntimeWarning, match="clipped to zero"):
            out = subtract_inventories(
                gridded,
                points,
                category_mapping={"power": "ene"},
                substance_mapping={"co2": "CO2"},
                negative_policy=policy,
            )
    else:
        out = subtract_inventories(
            gridded,
            points,
            category_mapping={"power": "ene"},
            substance_mapping={"co2": "CO2"},
            negative_policy=policy,
        )
    assert out.gdf[("ene", "CO2")].iloc[0] == pytest.approx(expected)


def test_negative_policy_raise(gridded: Inventory) -> None:
    with pytest.raises(ValueError, match="negative values"):
        subtract_inventories(
            gridded,
            _points([500.0]),
            category_mapping={"power": "ene"},
            substance_mapping={"co2": "CO2"},
            negative_policy="raise",
        )


def test_clipping_reports_what_could_not_be_removed(gridded: Inventory) -> None:
    with pytest.warns(RuntimeWarning):
        out = subtract_inventories(
            gridded,
            _points([500.0]),
            category_mapping={"power": "ene"},
            substance_mapping={"co2": "CO2"},
        )
    summary = out.subtraction_summary
    assert summary["requested_kg_y"]["ene/CO2"] == pytest.approx(500.0)
    assert summary["removed_kg_y"]["ene/CO2"] == pytest.approx(100.0)
    assert summary["clipped_values"] == 1


def test_missing_categories_are_reported(gridded: Inventory) -> None:
    out = subtract_inventories(gridded, _points([1.0], category="absent"))
    assert out.subtraction_summary["missing_columns"] == [["absent", "co2"]]
    # Nothing was removed.
    np.testing.assert_allclose(
        out.gdf[("ene", "CO2")].to_numpy(), [100.0, 200.0, 300.0, 400.0]
    )

    with pytest.raises(ValueError, match="not in"):
        subtract_inventories(
            gridded, _points([1.0], category="absent"), ignore_missing=False
        )


def test_subtracting_a_gridded_inventory_on_the_same_grid(
    gridded: Inventory, grid: RegularGrid
) -> None:
    other = Inventory.from_gdf(
        gpd.GeoDataFrame(
            {("ene", "CO2"): [1.0, 2.0, 3.0, 4.0]},
            geometry=grid.gdf.geometry,
            crs=WGS84,
        ),
        name="other",
    )
    other.grid = grid

    out = subtract_inventories(gridded, other)
    np.testing.assert_allclose(
        out.gdf[("ene", "CO2")].to_numpy(), [99.0, 198.0, 297.0, 396.0]
    )


def test_target_must_be_gridded() -> None:
    with pytest.raises(ValueError, match="must be defined on a grid"):
        subtract_inventories(_points([1.0]), _points([1.0]))


def test_unknown_policy_is_rejected(gridded: Inventory) -> None:
    with pytest.raises(ValueError, match="must be 'clip', 'raise' or 'allow'"):
        subtract_inventories(gridded, _points([1.0]), negative_policy="nope")


def test_mapping_may_cover_categories_that_are_absent(gridded: Inventory) -> None:
    """A shared mapping is a lookup table, not an assertion.

    ``DEFAULT_CORSO_SECTOR_MAPPING`` names every CORSO sector, but a cropped
    inventory usually holds only some of them.
    """
    out = subtract_inventories(
        gridded,
        _points([50.0]),
        category_mapping={
            "power": "ene",
            "cement": "ind",
            "absent_sector": "ene",
        },
        substance_mapping={"co2": "CO2", "absent_substance": "CO2"},
    )

    np.testing.assert_allclose(
        out.gdf[("ene", "CO2")].to_numpy(), [50.0, 200.0, 300.0, 400.0]
    )


def test_warns_loudly_when_nothing_matches(gridded: Inventory) -> None:
    """A wrong mapping would otherwise leave the emissions counted twice."""
    with pytest.warns(RuntimeWarning, match="Nothing was subtracted"):
        subtract_inventories(
            gridded,
            _points([50.0]),
            category_mapping={"power": "does_not_exist"},
            substance_mapping={"co2": "CO2"},
        )
