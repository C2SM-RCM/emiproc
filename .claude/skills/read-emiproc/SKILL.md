---
name: read-emiproc
description: Read, navigate and understand the emiproc Python package - find an inventory reader, get an exporter's real signature, trace what an operator does to an Inventory, see where a symbol is used or tested. Use when asked to explain, explore, search or modify emiproc source (inventories, exports, grids, regrid, profiles, speciation, units), or to answer "which inventories are supported" / "what does remap_inventory do" / "where is X implemented" in this tree.
---

# Reading the emiproc source

emiproc converts emission inventories into model-ready input files. The
package is small — **79 modules, ~19.7k lines in `emiproc/`** — and could be
read exhaustively, except that the repo around it is not: `docs/` is 47 MB,
45 MB of which is tutorial notebooks with base64 images inside. A grep from
the repo root drowns in them. Scope every search to `emiproc/`.

Drive it with the navigator:

```
.claude/skills/read-emiproc/nav.py
```

Paths below are relative to the repo root
(`/users/vhess/python_packages/emiproc`). Pure Python 3 stdlib, no emiproc
import — so it works without the geo stack installed and every command
returns in ~0.3 s. `nav.py` finds the repo from its own real location, so it
runs from any working directory, including through the symlink at
`~/.claude/skills/read-emiproc`. Point it elsewhere with
`EMIPROC_ROOT=/path/to/emiproc`.

It parses with `ast`, not regexes, so signatures, base classes and
deprecation flags are exact rather than best-guess.

## Start here

```bash
python3 .claude/skills/read-emiproc/nav.py overview
```

Prints the subsystem map, the branch state, and the pipeline that is the
spine of the package:

```
1. read       Inventory subclass   ->  inv
2. transform  operators on inv     ->  inv
3. export     export_* function    ->  files
```

Order within step 2 matters: non-aggregative ops (crop, speciate, scale)
before aggregative ones (`group_categories`, `remap_inventory`) — the
aggregative ones destroy resolution you cannot recover.

## The navigator

| Command | Answers |
|---|---|
| `nav.py overview` | subsystem map, pipeline, branch delta vs `origin/master` |
| `nav.py inventories` | every `Inventory` reader, live vs deprecated |
| `nav.py exports` | every `export_*` entry point + its helpers |
| `nav.py grids` | `Grid` classes, including the ones **not** in `grids.py` |
| `nav.py operators` | the transformations, grouped by module |
| `nav.py profiles` | the temporal/vertical subsystem (the confusing part) |
| `nav.py fixtures` | ready-made `Inventory`/`Grid` objects in `tests_utils` |
| `nav.py module <name>` | public API of one module: classes, functions, constants |
| `nav.py sym <Name> [--methods] [--body]` | definition, signature, docstring |
| `nav.py uses <Name> [--all]` | where a symbol is referenced |
| `nav.py tests <Name>` | which tests exercise it, plus the `pytest -k` line |
| `nav.py find <regex> [--in sub] [--tests] [--all]` | search, scoped to source — never `docs/` |
| `nav.py show <file> [start] [count]` | print with line numbers |

### `sym` — the main entry point

```bash
python3 .claude/skills/read-emiproc/nav.py sym remap_inventory
```

gives the resolved signature with defaults, the base classes, and the
docstring — without reading the file. For a class, `--methods` lists the
public API with properties marked:

```bash
python3 .claude/skills/read-emiproc/nav.py sym Inventory --methods
```

Use this instead of opening a module and scrolling. `--body` prints the
source when you actually need the implementation.

## The data model

One object carries everything (`emiproc/inventories/__init__.py:75`):

- **`inv.gdf`** — the gridded emissions. A `GeoDataFrame` whose geometry
  column holds the cells, and whose emission columns are **`(category,
  substance)` tuples**, not flat strings. This is the single most common
  source of confusion when reading indexing code.
- **`inv.gdfs`** — `dict[category, GeoDataFrame]` for shaped/point sources
  that do not live on the grid. An inventory may have either, or both.
- **`inv.grid`**, **`inv.history`** (the operation trail),
  `inv.categories`, `inv.substances`, `inv.total_emissions`.
- **Profiles**, in two independent axes with identical machinery:
  `v_profiles` + `v_profiles_indexes`, `t_profiles_groups` +
  `t_profiles_indexes`.

**Units: always `kg/year` per cell (or per shape), everywhere inside the
package.** `emiproc/utils/units.py` states it and
`get_scaling_factor_to_emiproc` is where foreign units are converted in.
Readers convert on the way in, exporters convert on the way out — so a
number seen mid-pipeline is `kg/y` unless you are inside an exporter's
final write.

**Profile indexes** are `xr.DataArray`s of integers over some subset of
`(category, substance, cell, country)`, whose values point into the
profiles list. **`-1` means "no profile assigned"** — it is a sentinel, not
a Python negative index, and code that forgets this silently picks the last
profile. See `emiproc/profiles/operators.py:182` and
`emiproc/profiles/utils.py:207`.

## Gotchas

These cost real time if you don't know them.

- **Never grep from the repo root.** `docs/source/tutos` is 45 MB of
  `.ipynb` — 20 notebooks with embedded images. Any repo-wide search
  returns base64 before it returns code. `nav.py find` scans `emiproc/`
  only; `--tests` / `--all` widen it deliberately, and nothing reaches
  `docs/`.

- **`emiproc/inventories/zurich.py` is a 0-byte file**, shadowed by the
  package `emiproc/inventories/zurich/`. Python's FileFinder checks
  packages before modules, so the directory is what imports — the `.py` is
  dead weight that makes it look like the Zurich reader is missing.
  `MapLuftZurich` lives in `zurich/__init__.py:10`. `nav.py module
  inventories.zurich` resolves to the package and says so.

- **`emiproc/profiles/temporal_profiles.py` is a deprecated shim** that
  does `from emiproc.profiles.temporal.* import *` for seven modules and
  logs a deprecation on import. Every temporal profile name therefore
  appears twice in a grep, once at a location that is not the definition.
  Import from `emiproc.profiles.temporal.<module>`.

- **Some inventory classes exist only to raise.** `GFAS_Inventory`
  (`gfas.py:18`) raises `DeprecationWarning` from `__init__`;
  `EDGAR_Inventory` (`edgar/edgarv8.py:238`) is superseded by `EDGARv8`.
  `nav.py inventories` lists these under a separate deprecated heading
  rather than mixing them into the live readers.

- **Not every `Grid` is in `grids.py`.** `WRF_Grid` is in
  `emiproc/exports/wrf.py:23` and `GFED_Grid` in
  `emiproc/inventories/gfed.py:52`. `nav.py grids` finds subclasses across
  the package and flags the out-of-file ones.

- **`remap_inventory` accepts both `weights_file` and `weigths_file`** —
  the misspelling is the old name, deprecated but still honoured
  (`emiproc/regrid.py:439`); passing both raises. So a caller using
  `weigths_file` is dated, not broken, and existing scripts that pass it
  still work.

- **Docstring first lines are not always true.** `scale_inventory`
  (`emiproc/inventories/utils.py:634`) is documented as "Get the total
  emissions from the inventory" — a copy-paste from `get_total_emissions`
  just above it. `nav.py` reports docstrings faithfully, which means it
  reports this one faithfully too. Confirm behaviour against the body or a
  test before relying on a one-liner.

- **You do not need data files to explore.** `nav.py fixtures` lists
  importable, ready-made objects — `from emiproc.tests_utils.test_inventories
  import inv` gives a working `Inventory` in one line, which is much faster
  than reasoning about what a function does.

## Related material in this repo

This skill is for *reading*. The repo already ships two skills for
*writing*, and they are the right ones to reach for once you know where
things are:

- `.github/AI_README.md` — the maintainers' own orientation doc: data
  model, unit rules, validation expectations (`pytest`, `black --check
  ./emiproc ./tests`).
- `.github/skills/emiproc-script/SKILL.md` — building a processing script.
- `.github/skills/emiproc-developer/SKILL.md` — changing emiproc internals.

Reach into `docs/source/` deliberately, never by search: `docs/source/api/`
is the public-API index, `docs/source/tutos/` the notebooks.

## Worked example

"How would I get CAMS-GLOB-ANT emissions onto a MicroHH grid?"

```bash
N=.claude/skills/read-emiproc/nav.py
python3 $N inventories                     # CAMSGlobal, inventories/cams_global.py:253
python3 $N grids                           # MicroHHGrid, grids.py:589
python3 $N find MicroHH --all              # who constructs one
python3 $N module exports.microhh          # export_cams_global_to_microhh :204
python3 $N sym export_cams_global_to_microhh
python3 $N tests export_cams_global_to_microhh
```

Six commands, no full-file reads, and you have the reader, the grid, the
exporter's real signature and the tests that pin its behaviour.
