from emiproc.utilities import get_day_per_year


def test_get_day_per_year():
    days = get_day_per_year(2000)
    assert days == 366


def test_get_day_per_year_2():
    days = get_day_per_year(2001)
    assert days == 365


def test_get_day_per_year_none():
    days = get_day_per_year(None)
    assert days == 365.25


def test_get_day_per_year_century():
    days_2100 = get_day_per_year(2100)
    days_2000 = get_day_per_year(2000)
    assert days_2100 == 365
    assert days_2000 == 366

    days_2400 = get_day_per_year(2400)
    assert days_2400 == 366
