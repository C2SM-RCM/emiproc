
import numpy as np

def read_tracer_profiles(tracers, hod_input_file, dow_input_file, moy_input_file):

    daily_profiles = {}
    weekly_profiles = {}
    annual_profiles = {}

    countries = []

    # daily profiles
    snaps, daily_profiles = read_daily_profiles(hod_input_file)

    for tracer in tracers:

        # weekly
        c, s, d = read_temporal_profiles(tracer, "weekly", dow_input_file)
        weekly_profiles[tracer] = d
        countries += c
        snaps += s

        # weekly
        c, s, d = read_temporal_profiles(tracer, "annual", moy_input_file)
        annual_profiles[tracer] = d
        countries += c
        snaps += s

    return (
        sorted(set(countries)),
        sorted(set(snaps)),
        daily_profiles,
        weekly_profiles,
        annual_profiles,
    )


def read_daily_profiles(filename):

    snaps = []
    data = {}

    with open(filename) as profile_file:
        for line in profile_file:
            values = line.split()
            snap = values[0].strip()
            data[snap] = np.array(values[1:], "f4")

    return snaps, data


def read_temporal_profiles(tracer, kind, filename):
    """\
    Read temporal profiles for given `tracer` for
    'weekly' or 'annual' profiles.
    """
    data = {}
    countries = []
    snaps = []

    filename = filename.format(tracer=tracer)

    with open(filename, "r") as profile_file:
        for line in profile_file:
            values = line.split()
            country, snap = int(values[0]), str(values[1])

            countries.append(country)
            snaps.append(snap)

            data[country, snap] = np.array(values[2:], "f4")

    return list(set(countries)), list(set(snaps)), data
