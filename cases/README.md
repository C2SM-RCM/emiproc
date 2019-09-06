# Examples

This readme describes how to create the examples from the paper.

## HOWTO: COSMO-GHG online example
1. GRID TNO and SWISS inventories on COSMO grid:
```
python -m epro grid --case config_tno
python -m epro grid --case config_carbocount
```

2. Merge SWISS and TNO inventories:
```
python -m epro append --case config_append
```

3. Create temporal and vertical profiles
```
python -m epro tp --case test_tp_simple.py
python -m epro vp
```

4. Create new profiles to merge the Swiss and TNO inventories
```
python -m epro tp-merge --case config_tpmerge
```

5. Move all profiles together
```
mv vertical_profiles.nc outputs/profiles/
```

## HOWTO: COSMO-GHG offline example
1. GRID TNO and SWISS inventories on COSMO grid:
```
python -m epro grid --case config_tno --offline
python -m epro grid --case config_carbocount --offline
```

2. Merge SWISS and TNO inventories:
```
python -m epro append --case config_append --offline
```

3. Create temporal and vertical profiles
```
python -m epro tp --case test_tp_simple.py
python -m epro vp
```

4. Create new profiles to merge the Swiss and TNO inventories
```
python -m epro tp-merge --case config_tpmerge
```

5. Move all profiles together
```
mv vertical_profiles.nc outputs/profiles/
```

5. Create hourly emissions
```
python -m epro hourly --case config_hourly
```

## HOWTO: COSMO-ART online example

1. Grid TNO and SWISS inventories on COSMO grid:
```
python -m epro grid --case config_d1_tno_art
python -m epro grid --case config_d1_swiss_art
```

2. Split GNFR category F (see scripts folder)
```
python ./split_gnfr_f.py oae-art-example/online/tno/tno-art.nc
python ./split_gnfr_f.py oae-art-example/online/swiss/swiss-art.nc
```

3. Create temporal and vertical profiles
```
python -m epro tp --case config_d1_tp_art
python -m epro tp-merge --case config_d1_tpmerge_art
python -m epro vp
```

4. Move all profiles to single path:
```
mv vertical_profiles.nc oae-art-example/profiles
```

5. Merge SWISS and TNO inventories:
```
python -m epro append --case config_d1_append
```

## HOWTO: COSMO-ART offline example

1. Grid TNO and SWISS inventories on COSMO grid:
```
python -m epro grid --case config_d1_tno_art --offline
python -m epro grid --case config_d1_swiss_art --offline
```

2. Split GNFR category F (see scripts folder)
```
python ./split_gnfr_f.py oae-art-example/offline/tno/tno-art.nc
python ./split_gnfr_f.py oae-art-example/offline/swiss/swiss-art.nc
```

3. Create temporal and vertical profiles
```
python -m epro tp --case config_d1_tp_art
python -m epro tp-merge --case config_d1_tpmerge_art
python -m epro vp
```

4. Move all profiles to single path:
```
mv vertical_profiles.nc oae-art-example/profiles
```

5. Merge SWISS and TNO inventories:
```
python -m epro append --case config_d1_append --offline
```

6. Create hourly emissions
```
python -m epro hourly --case config_d1_hourly_art --offline
```

