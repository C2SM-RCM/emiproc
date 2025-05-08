import pytest
import numpy as np
import json
import emiproc
from emiproc.profiles.temporal.profiles import (
    TemporalProfile,
    DailyProfile,
    WeeklyProfile,
    MounthsProfile,
    SpecificDayProfile,
    SpecificDay,
)
from emiproc.profiles.temporal.io import (
    from_csv,
    from_yaml,
    read_temporal_profiles,
    to_yaml,
)

from emiproc.profiles.vertical_profiles import read_vertical_profiles


def test_saving_and_loading_yamls_temporal_profiles():
    yaml_dir = emiproc.FILES_DIR / "profiles" / "yamls"
    yaml_profiles = {}
    for yml_file in yaml_dir.glob("*.yaml"):
        yaml_profiles[yml_file.stem] = from_yaml(yml_file)
    # We can also save time profiles to yaml
    loaded = {}
    for categorie in yaml_profiles.keys():
        yaml_file = emiproc.FILES_DIR / "outputs" / f"test_{categorie}.yaml"
        to_yaml(yaml_profiles[categorie], yaml_file)
        # Make sure we can load these yamls
        loaded[categorie] = from_yaml(yaml_file)
        # TODO: check that the loaded yaml is the same as the original one


@pytest.mark.parametrize(
    "name, profile",
    [
        ("daily", DailyProfile([1 / 24] * 24)),
        ("friday", SpecificDayProfile([1 / 24] * 24, specific_day=SpecificDay.FRIDAY)),
        (
            "weekend",
            SpecificDayProfile([1 / 24] * 24, specific_day=SpecificDay.WEEKEND),
        ),
        (
            "weekday4",
            SpecificDayProfile([1 / 24] * 24, specific_day=SpecificDay.WEEKDAY_4),
        ),
    ],
)
def test_saving_and_loading_yamls_specific_days(name, profile):

    yaml_dir = emiproc.TESTS_DIR / "profiles" / "specific_day_yaml"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    yaml_file = yaml_dir / f"{name}.yaml"
    to_yaml([profile], yaml_file)
    loaded = from_yaml(yaml_file)
    assert loaded[0] == profile


def test_load_csv_profiles():
    copernicus_profiles_dir = emiproc.FILES_DIR / "profiles" / "copernicus"

    profiles = ["hour_in_day", "day_in_week", "month_in_year"]
    profiles = {
        p: from_csv(copernicus_profiles_dir / f"timeprofiles-{p}.csv") for p in profiles
    }


def test_read_simple():
    """Test a simple file.

    Category,Substance,10,20,50
    blek,CO2,0.1,0.2,0.7
    liku,CO2,0.1,0.3,0.6
    liku,CH4,0,0.2,0.8
    blek,N20,0.2,0.4,0.4
    """
    proffile = emiproc.FILES_DIR / "test/profiles/simple_vertical/test_file.csv"

    profiles, indexes = read_vertical_profiles(proffile)

    # Test now the indexes are correct
    expect_present_dims = ["category", "substance"]
    for dim in expect_present_dims:
        assert dim in indexes.dims
    assert len(indexes.dims) == len(expect_present_dims)

    assert len(indexes.coords["category"]) == 2
    assert len(indexes.coords["substance"]) == 3
    # Test the indexes are correct
    # Note: this might be wrong if one day we change the order of indexing the profiles
    # Now we assume same order as in file
    assert indexes.loc["blek", "CO2"] == 0
    assert indexes.loc["liku", "CO2"] == 1
    assert indexes.loc["liku", "CH4"] == 2
    assert indexes.loc["blek", "N20"] == 3
    # Missing should be -1
    assert indexes.loc["blek", "CH4"] == -1
    assert indexes.loc["liku", "N20"] == -1

    # Test the profiles are correct
    assert len(profiles) == 4
    assert all(profiles.height == np.array([10, 20, 50]))


# Template to do the abovee test


@pytest.mark.parametrize(
    "name,profiles_dir,n_profiles,expected_dict",
    [
        (
            "Simple file",
            emiproc.FILES_DIR / "test/profiles/simple_vertical/test_file.csv",
            4,
            {
                '{"category":"blek","substance":"CO2"}': [0.1, 0.2, 0.7],
                '{"category":"liku","substance":"CO2"}': [0.1, 0.3, 0.6],
                '{"category":"liku","substance":"CH4"}': [0, 0.2, 0.8],
                '{"category":"blek","substance":"N20"}': [0.2, 0.4, 0.4],
                '{"category":"liku","substance":"N20"}': -1,
                '{"category":"blek","substance":"CH4"}': -1,
            },
        ),
        (
            "Two files, no conflict",
            emiproc.FILES_DIR / "test/profiles/multiple",
            5,
            {
                '{"category":"blek","substance":"CO2"}': [0.3, 0.4, 0.3],
                '{"category":"liku","substance":"CO2"}': [0.5, 0, 0.5],
                '{"category":"liku","substance":"CH4"}': [0.5, 0.1, 0.4],
                '{"category":"blek","substance":"N20"}': [0.2, 0.5, 0.3],
                '{"category":"liku","substance":"N20"}': [0.5, 0.2, 0.3],
                '{"category":"blek","substance":"CH4"}': -1,
            },
        ),
        (
            "Merging points and area sources",
            emiproc.FILES_DIR / "test/profiles/area_vs_point",
            7,
            {
                '{"category":"blek","substance":"CO2", "type":"gridded"}': [
                    0.3,
                    0.4,
                    0.3,
                ],
                '{"category":"liku","substance":"CO2", "type":"gridded"}': [
                    0.5,
                    0,
                    0.5,
                ],
                '{"category":"liku","substance":"CH4", "type":"gridded"}': [
                    0.5,
                    0.1,
                    0.4,
                ],
                '{"category":"blek","substance":"CH4", "type":"gridded"}': [
                    0.2,
                    0.5,
                    0.3,
                ],
                '{"category":"blek","substance":"CO2", "type":"shapped"}': [
                    0.1,
                    0.4,
                    0.5,
                ],
                '{"category":"liku","substance":"CO2", "type":"shapped"}': [
                    0.2,
                    0.7,
                    0.1,
                ],
                '{"category":"liku","substance":"CH4", "type":"shapped"}': [
                    0.3,
                    0.6,
                    0.1,
                ],
                '{"category":"blek","substance":"CH4", "type":"shapped"}': -1,
            },
        ),
        (
            "One file speciated, one file not",
            emiproc.FILES_DIR / "test/profiles/multiple_with_specification",
            4,
            {
                '{"category": "blek", "substance": "CO2"}': [0.4, 0.4, 0.2],
                '{"category": "blek", "substance": "N20"}': [0.1, 0.2, 0.7],
                # These are the same
                '{"category": "liku", "substance": "CO2"}': [0.5, 0.2, 0.3],
                '{"category": "liku", "substance": "N20"}': [0.5, 0.2, 0.3],
            },
        ),
    ],
)
def test_read_v_profiles(name, profiles_dir, n_profiles, expected_dict):
    profiles, indexes = read_vertical_profiles(profiles_dir)

    if profiles is None:
        raise AssertionError(f"Read vertical profile of test case '{name}' failed")

    # Test the values are correct
    assert len(profiles) == n_profiles
    for accessor, expected in expected_dict.items():
        index = indexes.loc[json.loads(accessor)]
        # Missing profile
        if index == -1:
            if expected == -1:
                continue
            else:
                raise AssertionError(
                    f"Read vertical profile of test case '{name}' failed\nExpected"
                    f" {expected} for {accessor=}, got {index=}"
                )
        received = profiles[index].ratios
        if not all(received == np.array(expected)):
            raise AssertionError(
                f"Read vertical profile of test case '{name}' failed\nExpected"
                f" {expected} for {accessor=}, got {received}"
            )


@pytest.mark.parametrize(
    "name,profiles_dir,n_profiles,expected_dict",
    [
        (
            "Simple file",
            emiproc.FILES_DIR / "test/profiles/simple_temporal/test_file.csv",
            4,
            {
                '{"category":"blek","substance":"CO2"}': [
                    WeeklyProfile([0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0])
                ],
                '{"category":"liku","substance":"CO2"}': [
                    WeeklyProfile([0.1, 0.3, 0.2, 0.4, 0.0, 0.0, 0.0])
                ],
                '{"category":"liku","substance":"CH4"}': [
                    WeeklyProfile([0.1, 0.4, 0.2, 0.3, 0.0, 0.0, 0.0])
                ],
                '{"category":"blek","substance":"N20"}': [
                    WeeklyProfile([0.1, 0.5, 0.2, 0.2, 0.0, 0.0, 0.0])
                ],
                '{"category":"liku","substance":"N20"}': -1,
                '{"category":"blek","substance":"CH4"}': -1,
            },
        ),
    ],
)
def test_t_profiles(name, profiles_dir, n_profiles, expected_dict):
    profiles, indexes = read_temporal_profiles(profiles_dir)

    if profiles is None:
        raise AssertionError(f"Read temporal profile of test case '{name}' failed")

    # Test the values are correct
    assert len(profiles) == n_profiles

    for accessor, expected in expected_dict.items():
        index = indexes.loc[json.loads(accessor)]
        # Missing profile
        if index == -1:
            if expected == -1:
                continue
            else:
                raise AssertionError(
                    f"Read temporal profile of test case '{name}' failed\nExpected"
                    f" {expected} for {accessor=}, got {index=}"
                )
        received = profiles[index]
        for r in received:
            assert isinstance(r, TemporalProfile)
        for e in expected:
            assert isinstance(e, TemporalProfile)

        if len(received) != len(expected):
            raise AssertionError(
                f"Read temporal profile of test case '{name}' failed\nExpected"
                f" {expected} for {accessor=}, got {received}"
            )

        for r in received:
            t = type(r)
            # Find the expected profile
            possilbe = [e for e in expected if isinstance(e, t)]
            if len(possilbe) != 1:
                raise AssertionError(
                    f"Read temporal profile of test case '{name}' failed\nExpected"
                    f" {expected} for {accessor=}, got {received}"
                )
            assert (
                r == possilbe[0]
            ), f" Temporal profile do not match received {r}, expected {possilbe[0]}"


def test_no_files():
    proffile = emiproc.FILES_DIR / "test/profiles/simple_vertical_doesnot_exist/"

    with pytest.raises(FileNotFoundError):
        profiles, indexes = read_vertical_profiles(proffile)


if __name__ == "__main__":
    # Test only this file
    pytest.main([__file__])
