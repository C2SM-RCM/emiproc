"""Legacy file from v1"""
catlist_prelim = [
    [
        'co2_ff_A_AREA_TNO',
        'co2_ff_A_POINT_TNO',
        'co2_ff_B_AREA_TNO',
        'co2_ff_B_POINT_TNO',
        'co2_ff_C_AREA_TNO',
        'co2_ff_F1_AREA_TNO',
        'co2_ff_F2_AREA_TNO',
        'co2_ff_F3_AREA_TNO',
        'co2_ff_D_AREA_TNO',
        'co2_ff_D_POINT_TNO',
        'co2_ff_E_AREA_TNO',
        'co2_ff_G_AREA_TNO',
        'co2_ff_H_AREA_TNO',
        'co2_ff_H_POINT_TNO',
        'co2_ff_I_AREA_TNO',
        'co2_ff_J_AREA_TNO',
        'co2_ff_J_POINT_TNO',
        'co2_ff_K_AREA_TNO',
        'co2_ff_L_AREA_TNO',
    ],  
]
for c in catlist_prelim[0].copy():
    catlist_prelim[0].append(c.replace('ff','bf'))
catlist_prelim[0] += [
    'CO2_B_Carbocount',
    'CO2_C_Carbocount',
    'CO2_J_Carbocount',
    'CO2_L_Carbocount',
    'CO2_F_Carbocount',
]

tplist_prelim = [
    [
        'GNFR_A_TNO',
        'GNFR_A_TNO',
        'GNFR_B_TNO',
        'GNFR_B_TNO',
        'GNFR_C_TNO',
        'GNFR_F_TNO',
        'GNFR_F_TNO',
        'GNFR_F_TNO',
        'GNFR_D_TNO',
        'GNFR_D_TNO',
        'GNFR_E_TNO',
        'GNFR_G_TNO',
        'GNFR_H_TNO',
        'GNFR_H_TNO',
        'GNFR_I_TNO',
        'GNFR_J_TNO',
        'GNFR_J_TNO',
        'GNFR_K_TNO',
        'GNFR_L_TNO',
    ],  
]
tplist_prelim[0] *= 2
tplist_prelim[0] += [
    'GNFR_B_Carbocount',
    'GNFR_C_Carbocount',
    'GNFR_J_Carbocount',
    'GNFR_L_Carbocount',
    'GNFR_F_Carbocount',
]

"""The vertical profile is only applied to area sources.
All area sources have emissions at the floor level.
As such, their profiles are using the area_sources profile"""



vplist_prelim = [
    [
        'GNFR_area_sources',
        'GNFR_A',
        'GNFR_area_sources',
        'GNFR_B',
        'GNFR_area_sources',
        'GNFR_area_sources',
        'GNFR_area_sources',
        'GNFR_D',
        'GNFR_area_sources',
        'GNFR_area_sources',
        'GNFR_area_sources',
        'GNFR_area_sources',
        'GNFR_area_sources',
        'GNFR_H',
        'GNFR_area_sources',
        'GNFR_area_sources',
        'GNFR_J',
        'GNFR_area_sources',
        'GNFR_area_sources',
    ],  
]

vplist_prelim[0] *= 2
vplist_prelim[0] += [
    'GNFR_B_Carbocount',
    'GNFR_C_Carbocount',
    'GNFR_J_Carbocount',
    'GNFR_L_Carbocount',
    'GNFR_F_Carbocount',
]


""" All emissions are applied on the floor in fact"""
vplist_prelim = [['GNFR_area_sources']]
vplist_prelim[0] *= len(tplist_prelim[0])
