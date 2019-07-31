#! /usr/bin/env python

import argparse
import os

import epro

from epro.profiles import temporal_profiles as tp
from epro.profiles import vertical_profiles as vp
from epro import utilities as util

from epro.merge_inventories import merge_inventories
from epro import append_inventories
from epro import merge_profiles
from epro import hourly_emissions


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

    parser.add_argument('--nomenclature', dest='nomenclature', default='GNFR',
                        help='GNFR or SNAP', choices=['GNFR', 'SNAP'])

    parser.add_argument('--offline', dest='offline', action='store_true',
                        help='')

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

    if args.offline:
        if hasattr(cfg, 'cosmo_grid'):
            print('Add two-cell boundary on COSMO grid')
            cfg.cosmo_grid.xmin -= 2 * cfg.cosmo_grid.dx
            cfg.cosmo_grid.ymin -= 2 * cfg.cosmo_grid.dy
            cfg.cosmo_grid.nx += 4
            cfg.cosmo_grid.ny += 4

        if hasattr(cfg, 'output_path'):
            cfg.output_path = cfg.output_path.format(online='offline')
    else:
        if hasattr(cfg, 'output_path'):
            cfg.output_path = cfg.output_path.format(online='online')

    if hasattr(cfg, 'output_path'):
        print('Output path: "%s"' % cfg.output_path)

    if args.task in ['grid']:

        if cfg is None:
            raise RuntimeError("Please supply a config file.")

        epro.main(cfg)


    elif args.task in ['merge']:

        if cfg is None:
            raise RuntimeError("Please supply a config file.")

        merge_inventories(cfg.base_inv, cfg.nested_invs, cfg.output_path)

    elif args.task in ['tp-merge']:

        if cfg is None:
            raise RuntimeError("Please supply a config file.")

        merge_profiles.main(cfg.inv1, cfg.inv2, cfg.countries,
                            cfg.profile_path_in, cfg.profile_path_out)

    elif args.task in ['append']:

        if cfg is None:
            raise RuntimeError("Please supply a config file.")

        if args.offline:
            cfg.inv_1 = cfg.inv_1.format(online='offline')
            cfg.inv_2 = cfg.inv_2.format(online='offline')
            cfg.inv_out = cfg.inv_out.format(online='offline')
        else:
            cfg.inv_1 = cfg.inv_1.format(online='online')
            cfg.inv_2 = cfg.inv_2.format(online='online')
            cfg.inv_out = cfg.inv_out.format(online='online')

        append_inventories.main(cfg)


    elif args.task in ['vp']:

        profile_filename = os.path.join(DATA_PATH, 'vert_profiles',
                                        'vert_prof_che_%s.dat' %
                                        args.nomenclature.lower())

        output_filename = os.path.join(args.output_path,
                                       'vertical_profiles.nc')

        vp.main(output_filename, profile_filename, prefix='%s_' %
                args.nomenclature)

    elif args.task in ['tp']: # temporal profiles

        if cfg is None:
            raise RuntimeError("Please supply a config file.")

        if cfg.profile_depends_on_species:
            tp.main_complex(cfg)
        else:
            tp.main_simple(cfg)


    elif args.task in ['offline']:

        if cfg is None:
            raise RuntimeError("Please supply a config file.")

        # create hourly (offline) emissions
        hourly_emissions.main(
            path_emi=cfg.path_emi,
            output_path=cfg.output_path,
            output_name=cfg.output_name,
            prof_path=cfg.prof_path,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            var_list=cfg.var_list,
            catlist=cfg.catlist,
            tplist=cfg.tplist,
            vplist=cfg.vplist
        )

    else:
        raise ValueError('Unknown task "%s"' % task)


if __name__ == '__main__':
    main()
