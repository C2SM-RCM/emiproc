#!/usr/bin/env python3
"""
nav.py - source navigator for the emiproc package.

Answers the questions an agent actually has when reading emiproc -- which
inventory readers exist, what an exporter's signature is, which operator
transforms an Inventory, where a symbol is used -- without dumping whole
files and without wading through docs/ (45 MB of tutorial notebooks that
bury every repo-wide grep).

The package is Python, so this parses it with `ast` rather than regexes:
signatures, base classes, decorators and docstrings come out exact, and
deprecated shims are flagged instead of silently offered as live API.

Pure stdlib, no emiproc import (so it works without the geo stack
installed). ~79 files, every command returns in well under a second.

Run `nav.py` with no arguments for the command list.
"""

from __future__ import annotations

import argparse
import ast
import functools
import os
import re
import subprocess
import sys

# Repo root = three levels above .claude/skills/read-emiproc/.
# realpath, not abspath: this script is normally reached through the symlink
# in ~/.claude/skills/, and abspath would resolve the repo relative to the
# link instead of the checkout.
ROOT = os.environ.get("EMIPROC_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..")
)
PKG = os.path.join(ROOT, "emiproc")

if not os.path.exists(os.path.join(PKG, "inventories", "__init__.py")):
    sys.exit(
        f"nav.py: {ROOT} is not an emiproc checkout "
        f"(no emiproc/inventories/__init__.py).\n"
        f"Set EMIPROC_ROOT=/path/to/emiproc to point it somewhere else."
    )

# Searchable trees, in the order a reader should prefer them. docs/ is
# excluded by default on purpose: docs/source/tutos is 45 MB of .ipynb with
# base64 images inside, and it dominates any repo-wide search.
TREES = {
    "pkg": PKG,
    "tests": os.path.join(ROOT, "tests"),
    "scripts": os.path.join(ROOT, "scripts"),
    "examples": os.path.join(ROOT, "examples"),
}

SUBSYSTEMS = [
    ("inventories", "emiproc/inventories", "readers: file/dataset -> Inventory"),
    ("exports", "emiproc/exports", "Inventory -> model input files"),
    ("profiles", "emiproc/profiles", "temporal + vertical allocation"),
    ("grids", "emiproc/grids.py", "Grid classes (geometry of the cells)"),
    ("regrid", "emiproc/regrid.py", "spatial remapping + weights"),
    ("speciation", "emiproc/speciation.py", "split/merge substances"),
    ("utilities", "emiproc/utilities.py", "countries, timezones, areas"),
    ("utils", "emiproc/utils", "units, translators, osm"),
    ("plots", "emiproc/plots", "plotting helpers"),
    ("tests_utils", "emiproc/tests_utils", "ready-made fixtures (see `nav.py fixtures`)"),
]

IGNORE_DIRS = {"__pycache__", ".git", ".pytest_cache", ".weights", ".outputs"}


# --------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------


def rel(path: str) -> str:
    return os.path.relpath(path, ROOT)


def py_files(tree: str):
    for dirpath, dirnames, filenames in os.walk(tree):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORE_DIRS)
        for name in sorted(filenames):
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def read(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


class Sym:
    """One top-level class or function, plus what a reader needs about it."""

    def __init__(self, node, path, module, parent=None):
        self.node = node
        self.path = path
        self.module = module
        self.parent = parent  # class name, for methods
        self.name = node.name
        self.line = node.lineno
        self.kind = "class" if isinstance(node, ast.ClassDef) else "def"
        self.doc = ast.get_docstring(node) or ""
        self.bases = (
            [unparse(b) for b in node.bases] if isinstance(node, ast.ClassDef) else []
        )
        self.decorators = [unparse(d) for d in node.decorator_list]

    @property
    def qualname(self) -> str:
        return f"{self.parent}.{self.name}" if self.parent else self.name

    @property
    def loc(self) -> str:
        return f"{rel(self.path)}:{self.line}"

    @property
    def summary(self) -> str:
        """First sentence of the docstring, on one line."""
        if not self.doc:
            return ""
        first = self.doc.strip().split("\n\n")[0].replace("\n", " ")
        first = re.sub(r"\s+", " ", first).strip()
        return shorten(first, 96)

    @property
    def deprecated(self) -> bool:
        if any("deprecated" in d.lower() for d in self.decorators):
            return True
        if ".. deprecated" in self.doc:
            return True
        # Several inventories keep the old name as a class whose __init__
        # does nothing but `raise DeprecationWarning(...)`.
        return "DeprecationWarning" in ast.dump(self.node)

    def signature(self) -> str:
        if isinstance(self.node, ast.ClassDef):
            init = next(
                (
                    n
                    for n in self.node.body
                    if isinstance(n, ast.FunctionDef) and n.name == "__init__"
                ),
                None,
            )
            if init is None:
                base = f"({', '.join(self.bases)})" if self.bases else ""
                return f"class {self.name}{base}"
            return f"{self.name}{render_args(init.args, drop_first=True)}"
        return f"{self.name}{render_args(self.node.args)}"

    def methods(self):
        return [
            Sym(n, self.path, self.module, parent=self.name)
            for n in self.node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]


def unparse(node) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "?"


def render_args(args: ast.arguments, drop_first: bool = False) -> str:
    """Render a signature, truncating long defaults so lines stay readable."""
    out = []
    posonly = list(args.posonlyargs)
    normal = list(args.args)
    positional = posonly + normal
    if drop_first and positional:
        positional = positional[1:]
    defaults = list(args.defaults)
    pad = len(posonly + normal) - len(defaults) - (1 if drop_first else 0)

    for i, a in enumerate(positional):
        piece = a.arg
        if a.annotation is not None:
            piece += f": {shorten(unparse(a.annotation))}"
        j = i - pad
        if 0 <= j < len(defaults):
            piece += f"={shorten(unparse(defaults[j]))}"
        out.append(piece)
        if posonly and i == len(posonly) - 1 - (1 if drop_first else 0):
            out.append("/")

    if args.vararg:
        out.append("*" + args.vararg.arg)
    elif args.kwonlyargs:
        out.append("*")
    for a, d in zip(args.kwonlyargs, args.kw_defaults):
        piece = a.arg
        if a.annotation is not None:
            piece += f": {shorten(unparse(a.annotation))}"
        if d is not None:
            piece += f"={shorten(unparse(d))}"
        out.append(piece)
    if args.kwarg:
        out.append("**" + args.kwarg.arg)
    return "(" + ", ".join(out) + ")"


def shorten(text: str, limit: int = 46) -> str:
    text = re.sub(r"\s+", " ", text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


class Module:
    def __init__(self, path: str):
        self.path = path
        self.name = module_name(path)
        self.source = read(path)
        try:
            self.tree = ast.parse(self.source, filename=path)
            self.ok = True
        except SyntaxError:
            self.tree = ast.Module(body=[], type_ignores=[])
            self.ok = False
        self.doc = ast.get_docstring(self.tree) or ""
        self.syms = [
            Sym(n, path, self.name)
            for n in self.tree.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        self.nlines = self.source.count("\n") + 1

    @property
    def constants(self):
        out = []
        for n in self.tree.body:
            targets = (
                n.targets
                if isinstance(n, ast.Assign)
                else [n.target] if isinstance(n, ast.AnnAssign) else []
            )
            for t in targets:
                if isinstance(t, ast.Name) and not t.id.startswith("_"):
                    out.append((t.id, n.lineno))
        return out

    @property
    def star_imports(self):
        return [
            n.module
            for n in self.tree.body
            if isinstance(n, ast.ImportFrom)
            and any(a.name == "*" for a in n.names)
            and n.module
        ]


def module_name(path: str) -> str:
    r = rel(path).replace(os.sep, ".")
    r = r[:-3] if r.endswith(".py") else r
    return r[: -len(".__init__")] if r.endswith(".__init__") else r


@functools.lru_cache(maxsize=1)
def index() -> list[Module]:
    return [Module(p) for p in py_files(PKG)]


def all_syms():
    for m in index():
        for s in m.syms:
            yield s


def find_module(query: str) -> Module | None:
    mods = index()
    q = query.replace("/", ".").removesuffix(".py").strip(".")
    exact = [m for m in mods if m.name in (q, f"emiproc.{q}")]
    if exact:
        # emiproc/inventories/zurich.py and emiproc/inventories/zurich/ both
        # claim the name `emiproc.inventories.zurich`. Python's FileFinder
        # checks packages before modules, so the package is what actually
        # gets imported -- resolve the same way, and say so.
        pkgs = [m for m in exact if m.path.endswith("__init__.py")]
        if pkgs and len(exact) > 1:
            other = [rel(m.path) for m in exact if m not in pkgs]
            print(
                f"  note: {', '.join(other)} also claims this module name, "
                f"but the package wins on import."
            )
            return pkgs[0]
        return exact[0]
    cands = [m for m in mods if m.name.endswith("." + q) or q in m.name]
    if len(cands) == 1:
        return cands[0]
    if cands:
        # prefer the shortest name: `profiles` should win over
        # `profiles.temporal.profiles`
        cands.sort(key=lambda m: (len(m.name), m.name))
        return cands[0]
    return None


def subclasses_of(base: str) -> list[Sym]:
    """Direct + transitive subclasses of `base`, by name (single package)."""
    known = {base}
    found, changed = [], True
    while changed:
        changed = False
        for s in all_syms():
            if s.kind != "class" or s in found:
                continue
            if any(b.split(".")[-1] in known for b in s.bases):
                found.append(s)
                known.add(s.name)
                changed = True
    return sorted(found, key=lambda s: (s.module, s.line))


# --------------------------------------------------------------------------
# output helpers
# --------------------------------------------------------------------------


def head(title: str):
    print(f"\n\033[1m{title}\033[0m" if sys.stdout.isatty() else f"\n== {title}")


def row(*cols, widths=(34, 30)):
    out = []
    for i, c in enumerate(cols[:-1]):
        w = widths[i] if i < len(widths) else 20
        c = str(c)
        out.append(c.ljust(w) if len(c) < w else c[: w - 1] + " ")
    out.append(str(cols[-1]))
    print("  " + " ".join(out).rstrip())


def print_syms(syms, note=True):
    for s in syms:
        flag = " [DEPRECATED]" if s.deprecated else ""
        row(s.qualname + flag, s.loc, s.summary if note else "")


def git(*args) -> str:
    try:
        return subprocess.run(
            ["git", "-C", ROOT, *args],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except Exception:
        return ""


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_overview(a):
    mods = index()
    print(f"emiproc at {ROOT}")
    nlines = sum(m.nlines for m in mods)
    print(f"{len(mods)} modules, ~{nlines:,} lines in emiproc/ (docs/ is 47 MB -- skip it)")

    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch:
        ahead = git("rev-list", "--count", "origin/master..HEAD")
        files = git("diff", "--name-only", "origin/master...HEAD")
        n = len([f for f in files.splitlines() if f])
        print(f"branch: {branch} ({ahead or '?'} commits / {n} files ahead of origin/master)")

    head("the pipeline (this is the spine of the package)")
    print("""  1. read     Inventory subclass  ->  inv          nav.py inventories
  2. transform  operators on inv   ->  inv          nav.py operators
  3. export     export_* function  ->  files        nav.py exports

  Order matters: non-aggregative ops (crop, speciate, scale) before
  aggregative ones (group_categories, remap_inventory), else you lose
  resolution you cannot get back.""")

    head("subsystems")
    for name, path, what in SUBSYSTEMS:
        full = os.path.join(ROOT, path)
        if os.path.isdir(full):
            n = len(list(py_files(full)))
            size = f"{n} modules"
        elif os.path.exists(full):
            size = f"{len(read(full).splitlines())} lines"
        else:
            size = "-"
        row(path, size, what, widths=(26, 12))

    head("the core object")
    inv = next((s for s in all_syms() if s.name == "Inventory"), None)
    if inv:
        print(f"  Inventory  {inv.loc}")
        print("  gdf / gdfs (emissions, kg/y per geometry), grid, history,")
        print("  v_profiles + v_profiles_indexes, t_profiles_groups + t_profiles_indexes")
        print("  `nav.py sym Inventory --methods` for the full API")

    print("\nNext: nav.py inventories | exports | operators | grids | profiles")


def cmd_inventories(a):
    syms = subclasses_of("Inventory")
    live = [s for s in syms if not s.deprecated]
    dead = [s for s in syms if s.deprecated]
    head(f"inventory readers ({len(live)} live)")
    for s in live:
        row(s.name, s.loc, s.summary, widths=(22, 44))
    if dead:
        head("deprecated (do not use; they raise or redirect)")
        for s in dead:
            row(s.name, s.loc, s.summary, widths=(22, 44))
    print("\n  nav.py sym <Name>  for the constructor signature and docstring")


def cmd_exports(a):
    head("export entry points")
    exp_dir = os.path.join(PKG, "exports")
    for m in index():
        if not m.path.startswith(exp_dir):
            continue
        fns = [
            s
            for s in m.syms
            if s.kind == "def"
            and not s.name.startswith("_")
            and not s.name.startswith("test_")
        ]
        if not fns:
            continue
        print(f"\n  {rel(m.path)}")
        for s in fns:
            mark = "*" if s.name.startswith("export") else " "
            row(f"{mark} {s.name}", f":{s.line}", s.summary, widths=(36, 6))
    print("\n  (*) = an export entry point; the rest are its helpers")
    print("  nav.py sym <name>  for the full signature")


def cmd_grids(a):
    base = [s for s in all_syms() if s.name == "Grid" and s.kind == "class"]
    subs = subclasses_of("Grid")
    core = [s for s in base + subs if s.path == os.path.join(PKG, "grids.py")]
    elsewhere = [s for s in subs if s.path != os.path.join(PKG, "grids.py")]
    head("grids (emiproc/grids.py)")
    for s in core:
        row(s.name, s.loc, s.summary, widths=(22, 24))
    if elsewhere:
        head("Grid subclasses defined OUTSIDE grids.py (easy to miss)")
        for s in elsewhere:
            row(s.name, s.loc, s.summary, widths=(22, 36))
    print("\n  A Grid provides the cell geometries; Inventory.gdf is indexed by cell.")
    print("  nav.py sym RegularGrid --methods")


OPERATOR_MODULES = [
    ("emiproc/inventories/utils.py", "inventory-level transformations"),
    ("emiproc/regrid.py", "spatial remapping"),
    ("emiproc/speciation.py", "substance splitting / merging"),
    ("emiproc/profiles/operators.py", "profile-aware operations"),
    ("emiproc/utilities.py", "countries, timezones, areas"),
]


def cmd_operators(a):
    for path, what in OPERATOR_MODULES:
        m = find_module(path)
        if m is None:
            continue
        head(f"{path}  -- {what}")
        for s in m.syms:
            if s.kind != "def" or s.name.startswith("_"):
                continue
            row(s.name, f":{s.line}", s.summary, widths=(34, 6))


def cmd_profiles(a):
    head("profiles subsystem -- the part that trips people up")
    print("""  Two independent axes, same indexing machinery:

    vertical   VerticalProfile(s)          emiproc/profiles/vertical_profiles.py
    temporal   TemporalProfile subclasses  emiproc/profiles/temporal/profiles.py
               CompositeTemporalProfiles   emiproc/profiles/temporal/composite.py

  An Inventory stores profiles + an *index array*:
    inv.v_profiles / inv.v_profiles_indexes
    inv.t_profiles_groups / inv.t_profiles_indexes
  The index is an xr.DataArray over some of (category, substance, cell,
  country); its values point into the profiles list. -1 means "no profile".
""")
    head("temporal profile classes")
    for s in subclasses_of("TemporalProfile"):
        if "tests_utils" in s.path:  # Profile2/Profile3 are fixtures, not API
            continue
        row(s.name, s.loc, s.summary, widths=(26, 46))
    head("operators")
    m = find_module("emiproc/profiles/operators.py")
    if m:
        for s in m.syms:
            if s.kind == "def" and not s.name.startswith("_"):
                row(s.name, f":{s.line}", s.summary, widths=(38, 6))
    head("gotcha")
    print("""  emiproc/profiles/temporal_profiles.py is a DEPRECATED shim that does
  `from emiproc.profiles.temporal.* import *`. Every name appears twice in a
  grep. Import from emiproc.profiles.temporal.<module>.""")


def cmd_fixtures(a):
    head("emiproc/tests_utils -- ready-made objects, no data files needed")
    print("""  Importing these gives you a working Inventory in one line, which is the
  fastest way to check what a function actually does:

    from emiproc.tests_utils.test_inventories import inv, inv_with_pnt_sources
    from emiproc.tests_utils.test_grids import regular_grid, gpd_grid
""")
    tu = os.path.join(PKG, "tests_utils")
    for m in index():
        if not m.path.startswith(tu):
            continue
        names = [n for n, _ in m.constants]
        fns = [s.name for s in m.syms if not s.name.startswith("_")]
        both = names + fns
        if not both:
            continue
        print(f"  {rel(m.path)}")
        print(f"      {', '.join(both[:14])}{' ...' if len(both) > 14 else ''}")


def cmd_module(a):
    m = find_module(a.name)
    if m is None:
        sys.exit(f"nav.py: no module matching {a.name!r}. Try `nav.py find {a.name}`.")
    head(f"{m.name}  ({rel(m.path)}, {m.nlines} lines)")
    if m.doc:
        print("  " + m.doc.strip().split("\n\n")[0].replace("\n", "\n  "))
    if not m.source.strip():
        print("\n  *** THIS FILE IS EMPTY ***")
    for star in m.star_imports:
        print(f"\n  re-exports * from {star}  (names here are defined elsewhere)")
    classes = [s for s in m.syms if s.kind == "class"]
    funcs = [s for s in m.syms if s.kind == "def"]
    if classes:
        head("classes")
        for s in classes:
            flag = " [DEPRECATED]" if s.deprecated else ""
            base = f"({', '.join(s.bases)})" if s.bases else ""
            row(s.name + base + flag, f":{s.line}", s.summary, widths=(38, 6))
    if funcs:
        head("functions")
        for s in funcs:
            if s.name.startswith("_"):
                continue
            row(s.signature(), f":{s.line}", "", widths=(70, 6))
    consts = m.constants
    if consts:
        head("module-level names")
        print("  " + ", ".join(f"{n}:{l}" for n, l in consts[:30]))


def cmd_sym(a):
    name = a.name
    hits = [s for s in all_syms() if s.name == name]
    if not hits:
        hits = [
            mth
            for s in all_syms()
            if s.kind == "class"
            for mth in s.methods()
            if mth.name == name or mth.qualname == name
        ]
    if not hits:
        close = sorted(
            {s.name for s in all_syms() if name.lower() in s.name.lower()}
        )
        msg = f"nav.py: no definition of {name!r} in emiproc/."
        if close:
            msg += "\nDid you mean: " + ", ".join(close[:12])
        sys.exit(msg)

    for s in hits:
        head(f"{s.qualname}   {s.loc}")
        if s.deprecated:
            print("  *** DEPRECATED ***")
        for d in s.decorators:
            print(f"  @{d}")
        print("  " + s.signature())
        if s.bases:
            print(f"  bases: {', '.join(s.bases)}")
        if s.doc:
            print()
            for line in s.doc.strip().splitlines()[: a.doc_lines]:
                print("    " + line)
        if a.methods and s.kind == "class":
            head("methods")
            for mth in s.methods():
                if mth.name.startswith("_") and mth.name != "__init__":
                    continue
                kind = "property" if any("property" in d for d in mth.decorators) else ""
                row(mth.signature(), f":{mth.line}", kind, widths=(58, 6))
        if a.body:
            end = getattr(s.node, "end_lineno", s.line + 40)
            print()
            print_lines(s.path, s.line, end - s.line + 1)


def cmd_uses(a):
    pat = re.compile(rf"\b{re.escape(a.name)}\b")
    trees = ["pkg", "tests", "scripts", "examples"] if a.all else ["pkg"]
    total = 0
    for key in trees:
        tree = TREES[key]
        if not os.path.isdir(tree):
            continue
        hits = []
        for path in py_files(tree):
            for i, line in enumerate(read(path).splitlines(), 1):
                if not pat.search(line):
                    continue
                stripped = line.strip()
                if not a.defs and re.match(
                    rf"(class|def)\s+{re.escape(a.name)}\b", stripped
                ):
                    continue
                hits.append((f"{rel(path)}:{i}", stripped))
        if hits:
            head(f"{key}  ({len(hits)} references)")
            for loc, line in hits[: a.limit]:
                row(loc, shorten(line, 100), widths=(46,))
            if len(hits) > a.limit:
                print(f"  ... {len(hits) - a.limit} more (--limit N)")
        total += len(hits)
    if not total:
        print(f"no references to {a.name!r} (searched: {', '.join(trees)})")
    elif not a.all:
        print("\n  (emiproc/ only -- add --all for tests/, scripts/, examples/)")


def cmd_find(a):
    try:
        pat = re.compile(a.pattern, 0 if a.case else re.I)
    except re.error as e:
        sys.exit(f"nav.py: bad regex: {e}")
    trees = ["pkg"]
    if a.all:
        trees = ["pkg", "tests", "scripts", "examples"]
    elif a.tests:
        trees = ["tests"]
    n = 0
    for key in trees:
        tree = TREES[key]
        if not os.path.isdir(tree):
            continue
        for path in py_files(tree):
            if a.in_ and a.in_ not in rel(path):
                continue
            for i, line in enumerate(read(path).splitlines(), 1):
                if pat.search(line):
                    n += 1
                    if n > a.limit:
                        print(f"  ... stopped at {a.limit} hits (--limit N)")
                        return
                    row(f"{rel(path)}:{i}", shorten(line.strip(), 110), widths=(46,))
    if not n:
        print(f"no match for {a.pattern!r}")


def cmd_tests(a):
    """Which tests exercise a symbol or module."""
    tests = TREES["tests"]
    if not os.path.isdir(tests):
        sys.exit("nav.py: no tests/ directory")
    pat = re.compile(rf"\b{re.escape(a.name)}\b")
    for path in py_files(tests):
        src = read(path)
        if not pat.search(src):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        funcs = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name.startswith("test")
        ]
        print(f"\n  {rel(path)}  ({len(funcs)} tests)")
        for n in funcs:
            seg = ast.get_source_segment(src, n) or ""
            if pat.search(seg):
                print(f"      {n.name}  :{n.lineno}")
    print(f"\n  run: pytest {rel(tests)} -k {a.name}")


def print_lines(path: str, start: int, count: int):
    lines = read(path).splitlines()
    start = max(1, start)
    for i in range(start - 1, min(len(lines), start - 1 + count)):
        print(f"{i + 1:5d}  {lines[i]}")


def cmd_show(a):
    path = a.file if os.path.isabs(a.file) else os.path.join(ROOT, a.file)
    if not os.path.exists(path):
        m = find_module(a.file)
        if m is None:
            sys.exit(f"nav.py: no such file: {a.file}")
        path = m.path
    print_lines(path, a.start, a.count)


# --------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        prog="nav.py", description="navigate the emiproc source"
    )
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("overview", help="package map, pipeline, branch state").set_defaults(
        fn=cmd_overview
    )
    sub.add_parser("inventories", help="every Inventory reader").set_defaults(
        fn=cmd_inventories
    )
    sub.add_parser("exports", help="every export_* entry point").set_defaults(
        fn=cmd_exports
    )
    sub.add_parser("grids", help="Grid classes").set_defaults(fn=cmd_grids)
    sub.add_parser("operators", help="inventory transformations").set_defaults(
        fn=cmd_operators
    )
    sub.add_parser("profiles", help="temporal/vertical profile subsystem").set_defaults(
        fn=cmd_profiles
    )
    sub.add_parser("fixtures", help="ready-made test objects").set_defaults(
        fn=cmd_fixtures
    )

    m = sub.add_parser("module", help="public API of one module")
    m.add_argument("name")
    m.set_defaults(fn=cmd_module)

    s = sub.add_parser("sym", help="definition, signature and docstring")
    s.add_argument("name")
    s.add_argument("--methods", action="store_true", help="list the class methods")
    s.add_argument("--body", action="store_true", help="print the source")
    s.add_argument("--doc-lines", type=int, default=25)
    s.set_defaults(fn=cmd_sym)

    u = sub.add_parser("uses", help="where a symbol is referenced")
    u.add_argument("name")
    u.add_argument("--all", action="store_true", help="also tests/, scripts/, examples/")
    u.add_argument("--defs", action="store_true", help="include the definition lines")
    u.add_argument("--limit", type=int, default=40)
    u.set_defaults(fn=cmd_uses)

    f = sub.add_parser("find", help="regex search, scoped to source (never docs/)")
    f.add_argument("pattern")
    f.add_argument("--in", dest="in_", help="only paths containing this substring")
    f.add_argument("--tests", action="store_true", help="search tests/ instead")
    f.add_argument("--all", action="store_true", help="search every source tree")
    f.add_argument("--case", action="store_true", help="case sensitive")
    f.add_argument("--limit", type=int, default=60)
    f.set_defaults(fn=cmd_find)

    t = sub.add_parser("tests", help="tests that exercise a name")
    t.add_argument("name")
    t.set_defaults(fn=cmd_tests)

    sh = sub.add_parser("show", help="print a file or module with line numbers")
    sh.add_argument("file")
    sh.add_argument("start", nargs="?", type=int, default=1)
    sh.add_argument("count", nargs="?", type=int, default=60)
    sh.set_defaults(fn=cmd_show)

    a = p.parse_args()
    if not a.cmd:
        p.print_help()
        return
    a.fn(a)


if __name__ == "__main__":
    main()
