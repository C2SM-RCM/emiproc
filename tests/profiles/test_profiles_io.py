import emiproc

from emiproc.profiles.temporal_profiles import (
    TemporalProfile,
    DailyProfile,
    WeeklyProfile,
    MounthsProfile,
    from_csv,
    from_yaml,
    to_yaml
)

def test_saving_and_loading_yamls_temporal_profiles():

    yaml_dir = emiproc.FILES_DIR / "profiles" / "yamls"
    yaml_profiles = {}
    for yml_file in yaml_dir.glob("*.yaml"):
        yaml_profiles[yml_file.stem] = from_yaml(yml_file)
    # We can also save time profiles to yaml
    loaded = {}
    for categorie in yaml_profiles.keys():
        yaml_file = emiproc.FILES_DIR / 'outputs' / f'test_{categorie}.yaml'
        to_yaml(yaml_profiles[categorie], yaml_file)
        # Make sure we can load these yamls 
        loaded[categorie] = from_yaml(yaml_file)
        # TODO: check that the loaded yaml is the same as the original one


def test_load_csv_profiles():
    copernicus_profiles_dir = emiproc.FILES_DIR / "profiles" / "copernicus"

    profiles = ['hour_in_day', 'day_in_week', 'month_in_year']
    profiles = {p: from_csv(copernicus_profiles_dir / f"timeprofiles-{p}.csv") for p in profiles}
