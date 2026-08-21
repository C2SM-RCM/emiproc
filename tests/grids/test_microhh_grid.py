import numpy as np

from emiproc.grids import MicroHHGrid


def test_microhh_grid_local_coordinates_and_anchor() -> None:
    grid = MicroHHGrid(
        xsize=10_000,
        ysize=8_000,
        itot=2,
        jtot=2,
        lon=0,
        lat=0,
        anchor="center",
        crs="EPSG:3857",
    )

    np.testing.assert_allclose(grid.x, [2500, 7500])
    np.testing.assert_allclose(grid.y, [2000, 6000])
    assert grid.nx == 2
    assert grid.ny == 2
    assert grid.x_offset == -5000
    assert grid.y_offset == -4000


def test_microhh_grid_infers_utm_crs() -> None:
    grid = MicroHHGrid(
        xsize=1_000,
        ysize=1_000,
        itot=2,
        jtot=2,
        lon=8.54,
        lat=47.38,
    )

    assert grid.crs.to_epsg() == 32632
