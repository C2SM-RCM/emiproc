

from fnmatch import fnmatch
import os

import numpy as np

# COSMOART speciation function starts
# calculate the fraction of single tracer out of inventory species    

# input files
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'files',
                         'speciation')

pm25composition_dir = os.path.join(DATA_PATH, 'pm25composition.csv')
tno_voc_dir= os.path.join(DATA_PATH, 'tno_voc.csv')


# mass fractions of NO2 and NO in NOX.
NOX_TO_NO2 = 0.18
NOX_TO_NO = (1.0 - NOX_TO_NO2) / 46.0 * 30.0

# ratios
j_ratio = 0.9


class WildcardsDict(dict):
    """
    Dict with get_wildcard method to match keys with Unix shell-style
    wildcards using `fnmatch` module.
    """
    def get_wildcard(self, pattern, default=None):
        """
        Return the value for key matching pattern or else default.
        """
        for key in sorted(self):
            if fnmatch(pattern, key):
                return self[key]
        return default


def pm2aerosol(j_ratio, pm25composition_dir):
    """ Calculate single aerosol emission from pm25 and pm10

        input:  j_ratio: fraction of j mode of total aerosol emission
                pm25composition_dir: a map of factors that assign pm25 to specific aerosol
        output: for a single tracer, fraction per each source out of pm25
    """

#    antha = pm10 - pm25  # check if <0

    #assign pm25 to so4, orgpa, soot, p25a
    pm25composition = np.loadtxt(pm25composition_dir, delimiter=',', skiprows=1, usecols=range(1,6))    

    so4 = pm25composition[:,0]
    orgpa = pm25composition[:,1]
    soot = pm25composition[:,2]

    p25a = 1 - so4 - orgpa - soot

    # assign to i,j mode
    so4j = so4 * j_ratio
    so4i = so4 * (1-j_ratio)
    orgpaj = orgpa * j_ratio
    orgpai = orgpa * (1-j_ratio)
    p25aj = p25a * j_ratio
    p25ai = p25a * (1-j_ratio)

    return soot, so4i, so4j, orgpai, orgpaj, p25ai, p25aj


def nmvoc2gas(tno_voc_dir):
    out = np.loadtxt(tno_voc_dir, delimiter=',', unpack=True, skiprows=1,
                     usecols=np.arange(14)+1)
    return out





def create_mapping(nomenclature='GNFR'):

    if nomenclature != 'GNFR':
        raise NotImplementedError('Only GNFR supported for nomenclature.')

    spe = {}

    # nitrogen oxides
    spe['NOe'] = {'NOX_*': NOX_TO_NO}
    spe['NO2e'] = {'NOX_*': NOX_TO_NO2}

    # NMVOC categories (without K)
    out = nmvoc2gas(tno_voc_dir)
    cats = ['A', 'B', 'C', 'D', 'E', 'F1', 'F2', 'F3', 'F4',
            'G', 'H', 'I', 'J', 'L']

    spe['ALDe'] = dict(('NMVOC_%s*' % c, out[0,i]) for i,c in enumerate(cats))
    spe['HCHOe'] = dict(('NMVOC_%s*' % c, out[1,i]) for i,c in enumerate(cats))
    spe['ORA2e'] = dict(('NMVOC_%s*' % c, out[2,i]) for i,c in enumerate(cats))
    spe['HC3e'] = dict(('NMVOC_%s*' % c, out[3,i]) for i,c in enumerate(cats))
    spe['HC5e'] = dict(('NMVOC_%s*' % c, out[4,i]) for i,c in enumerate(cats))
    spe['HC8e'] = dict(('NMVOC_%s*' % c, out[5,i]) for i,c in enumerate(cats))
    spe['ETHe'] = dict(('NMVOC_%s*' % c, out[6,i]) for i,c in enumerate(cats))
    spe['OL2e'] = dict(('NMVOC_%s*' % c, out[7,i]) for i,c in enumerate(cats))
    spe['OLTe'] = dict(('NMVOC_%s*' % c, out[8,i]) for i,c in enumerate(cats))
    spe['OLIe'] = dict(('NMVOC_%s*' % c, out[9,i]) for i,c in enumerate(cats))
    spe['TOLe'] = dict(('NMVOC_%s*' % c, out[10,i]) for i,c in enumerate(cats))
    spe['XYLe'] = dict(('NMVOC_%s*' % c, out[11,i]) for i,c in enumerate(cats))
    spe['KETe'] = dict(('NMVOC_%s*' % c, out[12,i]) for i,c in enumerate(cats))
    spe['CSLe'] = dict(('NMVOC_%s*' % c, out[13,i]) for i,c in enumerate(cats))


    # aerosols
    soot, so4i, so4j, orgpai, orgpaj, p25ai, p25aj = pm2aerosol(j_ratio,
                                                                pm25composition_dir)
    # SO2 categories
    cats = ['A', 'B', 'C', 'D', 'E', 'F1', 'F2', 'F3', 'F4',
            'G', 'H', 'I', 'J', 'K', 'L']

    spe['VSO4Ie'] = dict(('PM25_%s*' % c, so4i[i]) for i,c in enumerate(cats))
    spe['VSO4Je'] = dict(('PM25_%s*' % c, so4j[i]) for i,c in enumerate(cats))


    # others
    cats = ['A', 'B', 'C', 'D', 'E', 'F1', 'F2', 'F3', 'F4',
            'G', 'H', 'I', 'J', 'K', 'L']

    spe['VORGPAIe'] = dict(('PM25_%s*' % c, orgpai[i]) for i,c in enumerate(cats))
    spe['VORGPAJe'] = dict(('PM25_%s*' % c, orgpaj[i]) for i,c in enumerate(cats))


    spe['VP25AIe'] = dict(('PM25_%s*' % c, p25ai[i]) for i,c in enumerate(cats))
    spe['VP25AJe'] = dict(('PM25_%s*' % c, p25aj[i]) for i,c in enumerate(cats))


    spe['VANTHAe'] = {'PM10_*': 1.0, 'PM25_*': -1.0}
    spe['VSOOTe'] = dict(('PM25_%s*' % c, soot[i]) for i,c in enumerate(cats))
    spe['VSOOTe']['BC_*_ch'] = 1.0


    # gases without speciation
    spe['SO2e'] = {'SO2_*': 1.0}
    spe['NH3e'] = {'NH3_*': 1.0}
    spe['COe'] = {'CO_*': 1.0}


    # convert to WildcardsDict
    for key, value in spe.items():
        spe[key] = WildcardsDict(value)

    return spe
