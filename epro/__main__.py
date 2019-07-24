#! /usr/bin/env python

import argparse
import os

from epro.profiles import temporal_profiles_example as tp
from epro.profiles import vertical_profiles as vp
from epro import utilities as util

from epro import main_carbocount_example
from epro import main_tno_example
from epro import append_inventories

# TODO/FIXME
# - use temporal_profiles or temporal_profiles_example?

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'files')


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(dest='task',
                        help='name of task')

    parser.add_argument('-c', '--case', dest='case', default=None,
                        help='name of case')

    parser.add_argument('-cf', '--case-file', dest='case_file', default=None,
                        help='path to case file')

    parser.add_argument('--output-path', dest='output_path', default='.',
                        help='name of output path')

    args = parser.parse_args()

    return args



def main():

    args = parse_arguments()

    # make output path
    os.makedirs(args.output_path, exist_ok=True)

    if args.case is not None:
        cfg_path = os.path.join(os.path.dirname(__file__), '..', 'cases',
                                args.case)
        cfg = util.load_cfg(cfg_path)

    elif args.case_file is not None:
        cfg_path = args.case_file
        cfg = util.load_cfg(cfg_path)

    else:
        cfg = None


    if args.task in ['grid']:

        if cfg is None:
            raise RuntimeError("Please supply a config file.")

        if cfg.inventory == 'TNO':
            main_tno_example.main(cfg)

        elif cfg.inventory == 'carbocount':
            main_carbocount_example.main(cfg)

        else:
            raise NotImplementedError(
                'Gridding for "%s" inventory not implemented.'
                % cfg.inventory
            )

    elif args.task in ['merge']:

        if cfg is None:
            raise RuntimeError("Please supply a config file.")

        raise NotImplementedError


    elif args.task in ['append']:

        if cfg is None:
            raise RuntimeError("Please supply a config file.")

        append_inventories.main(cfg)


    elif args.task in ['vp']: # vertical profiles

        profile_filename = os.path.join(DATA_PATH, 'vert_profiles',
                                        'vert_prof_che_gnfr.dat')

        output_filename = os.path.join(args.output_path,
                                       'vertical_profiles.nc')

        vp.main(output_filename, profile_filename)

    elif args.task in ['tp']: # temporal profiles

        tp.main(args.output_path, DATA_PATH)

    elif args.task in ['hourly']:

        # create hourly (offline) emissions
        raise NotImplementedError

    else:
        raise ValueError('Unknown task "%s"' % task)


if __name__ == '__main__':
    main()
