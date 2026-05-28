---
name: emiproc-developer
description: "Use when developing emiproc internals: adding or changing inventories, exporters, regridding logic, profiles, scripts, and tests with maintainer-level quality checks for regressions, units, and API behavior."
argument-hint: "task=<feature|bugfix|refactor> area=<inventories|exports|regrid|profiles>"
user-invocable: true
---

# Emiproc Developer Skill

## Mandatory First Step
- Read [.github/AI_README.md](../../AI_README.md) before doing any analysis, edits, or suggestions.


## Goal
Support maintainers and contributors making code changes in emiproc with minimal regressions and clear validation.

## Use This Skill When
- Adding a new inventory reader or modifying an existing one.
- Extending export logic (NetCDF, rasters, model inputs).
- Implementing or improving a processing function (regridding, speciation, scaling, ...).
- Fixing bugs in emiproc.
- Preparing a PR with tests and docs updates.

## Change Guidance By Area

In general make sure to read the developer tutorials in the docs.

### Inventories
- Keep parsing and normalization logic explicit.
- Document assumptions about category and substance naming.
- Make sure to follow emiproc's internal data model conventions (for example units, category/substance organization in `gdf`).
- If external data is freely available, add a function to download it.

### Exports
- Preserve stable metadata and coordinate conventions.
- Verify required output units and variable naming consistency.
- Warn the user about potential limitations or incompatibilities in the export (example: if the model doesn't support temporal profiles as defined in the emiproc inventory).

## Project Pointers
- Package root: [emiproc](../../../emiproc)
- Tests: [tests](../../../tests)
- Scripts: [scripts](../../../scripts)
- Documentation source: [docs/source](../../../docs/source)
