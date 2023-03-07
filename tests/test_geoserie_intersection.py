
import geopandas as gpd
from emiproc.regrid import geoserie_intersection
from shapely.geometry import  Polygon
import numpy as np


# prepare test data for the tests
serie = gpd.GeoSeries(
    [
        Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),
        Polygon(((2, 1), (2, 2), (3, 2), (3, 1))),
        Polygon(((0, 1), (0, 2), (1, 2), (1, 1))),
        Polygon(((1, 0), (1, 1), (2, 1), (2, 0))),
        Polygon(((1, 1), (1, 2), (2, 2), (2, 1))),
    ]
)
triangle = Polygon(((0.5, 0.5), (1.5, 0.5), (1.5, 1.5)))

expected_weights_droped = np.array([0.125, 0.25 , 0.125])
expected_weights_undroped = np.array([0.125, 0.,  0.   , 0.25 , 0.125])


def test_normal_intersection():

    intersection, weights = geoserie_intersection(serie, triangle)

    # Shape number 1 and 2 shoud have disappeared
    assert len(intersection) == 3
    # We cannot test the follwoing sadly
    print(intersection.iloc[0])
    
    # Test the shapes are what we expect
    assert intersection.iloc[0].equals(Polygon(((0.5, 0.5), (1, 0.5), (1, 1))))
    assert intersection.iloc[1].equals(Polygon(((1, 0.5), (1.5, 0.5), (1.5, 1), (1, 1))))
    assert intersection.iloc[2].equals(Polygon(((1.5, 1.5), (1.5, 1), (1, 1))))

    assert np.all(expected_weights_droped == weights)

def test_intersection_drop_unused():

    intersection, weights = geoserie_intersection(serie, triangle, drop_unused=False)

    # Shape number 1 shoud have disappeared
    assert 1  in intersection.index
    assert np.all(expected_weights_undroped == weights)

def test_intersection_keep_outside():

    intersection, weights = geoserie_intersection(serie, triangle, keep_outside=True)

    assert np.all(expected_weights_undroped == (1-weights))

