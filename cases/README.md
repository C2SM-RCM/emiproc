# Examples

This readme describes how to create the examples from the paper.

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
python -m epro tp --case test_tp_simple.py
python -m epro vp
```

4. Move all profiles to single path:
```
mv test_time_profiles_simple profiles
mv vertical_profiles.nc profiles/
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
python -m epro tp --case test_tp_simple.py
python -m epro vp
```

4. Move all profiles to single path:
```
mv test_time_profiles_simple profiles
mv vertical_profiles.nc profiles/
```

5. Merge SWISS and TNO inventories:
```
python -m epro append --case config_d1_append --offline
```

6. Create hourly emissions
```
python -m epro hourly --case config_d1_hourly_art --offline
```

