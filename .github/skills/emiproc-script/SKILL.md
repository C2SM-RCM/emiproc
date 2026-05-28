---
name: emiproc-script
description: "Use when processing emission inventories with emiproc. Helps create a processing script that loads, processes, and exports an inventory"
argument-hint: "inventory=<name> year=<year> grid=<target> output=<format>"
user-invocable: true
---

# Emiproc Operations Skill

## Mandatory First Step
- Read [.github/AI_README.md](../../AI_README.md) before doing any analysis, edits, or suggestions.
- Look at the tutorials to understand the typical workflow and processing steps in emiproc. (see link below)

## Goal
Provide a reliable workflow for operational emissions processing tasks in emiproc, from raw inventory loading to validated export files.

## Use This Skill When
- You need to build or update an emissions processing pipeline.
- You need to regrid or remap an inventory to a model or grid.
- You need to apply category grouping, speciation, or scaling before export.
- You need confidence checks on totals, units, and category consistency.

## Inputs To Clarify First
- Inventory source and year.
- Export domain or model.
- Target grid definition (resolution, extent, coordinates).
- Category framework (native categories vs grouped categories such as GNRF).
- Substances required in output.
- Output format and consumer (analysis, NetCDF rasters, model export).
- Unit convention expected by downstream model or analysis.


## Safe Transformation Order

When transforming inventories, you might aggregate data in ways that cause data loss. 
To minimize this risk, follow this order of operations:

1. Load inventory.
2. Non-aggregative transformations (for example cropping, masking, filtering, speciation, scaling).
3. Aggregative transformations (for example grouping, remapping/regridding).
4. Export.

If order changes, document why and validate totals before/after.

## Scientific Guardrails

- Validate emissions totals around every major transformation.
- Look at the data (plots) when possible. 
- Make sure the user is not implementing operations that could create silent data loss (dropping emission sources in a non justified way).
- Define a script version variable (for example `VERSION = "v1.0"`) and pass it into export metadata attributes.

## Script Style Preferences

- Write a flat script for operational workflows; avoid creating helper functions for each step when direct emiproc calls are sufficient.
- Do not wrap template scripts in a `main()` function
- Use Python `logging` for runtime status and checks instead of `print`, as emiproc is compatible with logging. Create a logger object for the script.
- Use f-strings for any string formatting. (also in logging messages)

## Project Pointers
- Inventory implementations: [emiproc/inventories](../../../emiproc/inventories)
- Exporters: [emiproc/exports](../../../emiproc/exports)
- Grid and regridding utilities: [emiproc/grids.py](../../../emiproc/grids.py), [emiproc/regrid.py](../../../emiproc/regrid.py)
- Example Zurich workflow script: [scripts/zh_2_raster.py](../../../scripts/zh_2_raster.py)
- Tutorials: [docs/source/tutos](../../../docs/source/tutos)
