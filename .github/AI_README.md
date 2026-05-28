# Emiproc AI README (Read This First)

This document gives AI agents the minimum project context needed to work safely in emiproc.

## What Emiproc Is

Emiproc is a Python package for processing emission inventories and exporting model-ready emission inputs.

Core workflow:
1. Read an inventory.
2. Apply transformations (grouping, speciation, scaling, cropping, remapping, ... ).
3. Export to a target format (for example NetCDF rasters, ICON-ART, WRF-Chem, GeoPackage, fluxie).

## Key Data Model

Main object: `emiproc.inventories.Inventory`.

Important attributes:
- `gdf`: main GeoDataFrame with geometry and emission columns.
- `gdfs`: per-category GeoDataFrames for inventories with category-specific shapes.
- `grid`: inventory grid object.
- `history`: operation history trail.
- `v_profiles`, `t_profiles_groups`: vertical/temporal profiles.
- `v_profiles_indexes`, `t_profiles_indexes`: profile assignment indexes.

Emission columns are typically organized by `(category, substance)` in the main `gdf`.

## Supported Domains (High Level)

How to find supported inventories:
1. Check API docs: `docs/source/api/inventories.rst`.
2. Check user docs overview: `docs/source/inventories.rst`.
3. Inspect implementation modules in `emiproc/inventories/` (classes usually represent supported readers).

How to find supported exports:
1. Check API docs: `docs/source/api/exports.rst`.
2. Check model/export overview: `docs/source/models.rst`.
3. Inspect implementation modules in `emiproc/exports/` for available export functions.

When in doubt, prefer API docs first, then confirm by locating the implementation in the package.

Profiles:
- Profiles are an advanced topic; only dive into them when the task requires temporal/vertical allocation.

## Units (Quick Rules)

- Internal inventory convention: emission values in `gdf`/`gdfs` are in `kg/y` per geometry (for gridded data: per cell).
- Profile context: annual total emission is handled as `kg/year/source`; temporal/vertical profile ratios are dimensionless.
- Exporters and Inventories convert units to target-specific output units automatically; always verify output variable/unit metadata after export.



## Practical Coding Rules For Agents

- Implement the smallest change that solves the task.
- Reuse existing inventory/export/regrid patterns before introducing abstractions.
- Avoid machine-local absolute paths in reusable package code.
- Keep behavior deterministic; avoid hidden fallback logic.
- Update docs when behavior visible to users changes.

## Validation Expectations

From repository root:

```bash
pytest
black --check ./emiproc
black --check ./tests
```

Prefer targeted tests for touched modules first, then broader checks as needed.

## Where To Look First In The Repo

- Core package: `emiproc/`
- Inventories: `emiproc/inventories/`
- Exports: `emiproc/exports/`
- Grid/regrid logic: `emiproc/grids.py`, `emiproc/regrid.py`
- Tests: `tests/`
- Examples/scripts: `scripts/`
- Docs/tutorials: `docs/source/` and `docs/source/tutos/`

## Documentation Lookup Checklist (Use Only What The Task Needs)

- Inventory readers and formats:
	- `docs/source/inventories.rst`
	- `docs/source/api/inventories.rst`
- Grids, CRS, and remapping:
	- `docs/source/api/grids.rst`
	- `docs/source/api/operators.rst`
	- `docs/source/tutos/grids.ipynb` (regular grid examples)
	- `docs/source/tutos/regridding.ipynb`
	- `emiproc/grids.py` (`RegularGrid` implementation)
- Categories and speciation logic:
	- `docs/source/api/categories.rst`
	- `docs/source/api/speciation.rst`
	- `docs/source/api/operators.rst`
- Export targets and output conventions:
	- `docs/source/models.rst`
	- `docs/source/api/exports.rst`
	- `docs/source/tutos/icon_oem.ipynb`
- Profiles (advanced; optional unless needed):
	- `docs/source/profiles.rst`
	- `docs/source/api/profiles.rst`
	- `docs/source/tutos/profiles.ipynb`
	- `docs/source/tutos/temporal_profiles_from_traffic_counter.ipynb`
- Emission models (VPRM, HDD, human respiration):
	- `docs/source/emissions_generation.rst`
	- `docs/source/api/models.rst`
- Contribution patterns and extension points:
	- `docs/source/contrib/new_inventory.rst`
	- `docs/source/contrib/contribute.rst`

## Start-Of-Task Checklist For AI

Before coding:
1. Identify the closest existing pattern in code and/or tutorials.
2. Confirm units, CRS, categories, and substances involved.
3. Decide and state transformation order.
4. Define minimal validation (totals + at least one behavior check).

Before finishing:
1. Re-check totals or conservation where relevant.
2. Run tests/format checks relevant to modified modules.
3. Note any assumptions or limitations explicitly.
