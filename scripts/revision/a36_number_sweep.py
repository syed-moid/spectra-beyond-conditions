#!/usr/bin/env python3
"""Round 50 item 7: match every number in the manuscripts to a source file.

The manuscripts quote several hundred numbers. Rounds 46-49 each found at least
one that had gone stale -- a variance printed in a standard-deviation column, a
fitting row regenerated everywhere but one table, a gap fraction quoted from a
retired evaluation policy. Spot-checking does not find those; this does.

**What it does.** It reads DRAFT_v8_9_main.md and DRAFT_v8_9_supplement.md, pulls
out every numeric token, and asks whether some file in the value universe below
contains a number that rounds to it at the precision the prose used. A token that
no source reproduces is reported; the covering report lists them.

**The value universe.**

  1. the digest, `A12_<version>/consolidated_results.csv` (value and sd columns);
  2. every CSV and JSON under the active version's revision result directories --
     these are the files the ledger rows name;
  3. every figure CSV under `figures/`, which is what a reader
     checking a figure against the text would use;
  4. numeric literals in the scripts that define the generator, the fitting
     objective and its bounds, and the architectures. Fit bounds, the 600-bin
     grid, the penalty weight and the layer widths are specifications, not
     results: they are matched against the code that implements them.

**What it deliberately does not flag**, each pattern declared in EXEMPT below:
cross-references (section, figure, table, equation numbers), citation years,
DOIs and URLs, dataset version tags, markdown list markers, and the contents of
HTML comments. Every exemption is reported with a count so the exemption list
itself can be audited.

A number that is *matched* is not thereby correct -- it means some file contains
it. The value of the sweep is the unmatched list.
"""

from __future__ import annotations

import argparse, json, re, sys
from pathlib import Path

import pandas as pd
import dataset_paths as DS

def _manuscript_dir():
    """Where the manuscript sources live.

    The manuscript is not part of the code release -- it is the journal's record,
    not the package's -- so the scripts that read it take their location from
    SBC_MANUSCRIPT_DIR. Without it they fail with this message rather than with a
    confusing missing-path error.
    """
    import os
    d = os.environ.get("SBC_MANUSCRIPT_DIR")
    if not d:
        raise SystemExit(
            "This script reads the manuscript sources, which are not part of the "
            "code release. Set SBC_MANUSCRIPT_DIR to the directory holding "
            "DRAFT_*_main.md and DRAFT_*_supplement.md, for example:\n"
            "    SBC_MANUSCRIPT_DIR=../manuscript python " + __file__)
    return Path(d).resolve()


ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "results" / "revision"
SCRIPTS = Path(__file__).resolve().parent
MS = _manuscript_dir()
FIGS = ROOT / "figures"
OUT = ROOT / "reports"

SPEC_SOURCES = [
    SCRIPTS / "a3_fitting_baselines.py",
    SCRIPTS / "a3b_extra_baselines.py",
    SCRIPTS / "dataset_paths.py",
    ROOT / "sbc" / "data" / "spectrum_generator.py",
]
MAX_DECIMALS = 6

# Spans removed before tokenizing, with the reason. Order matters: URLs first.
EXEMPT = [
    ("html comment", re.compile(r"<!--.*?-->", re.S)),
    ("url or doi", re.compile(r"https?://\S+|doi:\S+|10\.\d{4,}/\S+")),
    ("code span", re.compile(r"`[^`]*`")),
    ("section cross-reference", re.compile(r"§\s?\d+(?:\.\d+)*")),
    ("supplement cross-reference", re.compile(r"\bS\d+(?:\.\d+)*\b")),
    ("figure or table cross-reference",
     re.compile(r"\b(?:Figure|Fig\.|Table|Appendix|Equation|Eq\.)\s*[-A-Za-z]?\d+[a-z]?\b")),
    ("dataset version tag", re.compile(r"\bv\d+\b")),
    ("citation year", re.compile(r"\(\s?\d{4}[a-z]?\s?\)|\b(?:19|20)\d{2}[a-z]?\b")),
    ("markdown heading number", re.compile(r"^#{1,6}\s*[A-Za-z]?\d+(?:\.\d+)*", re.M)),
    ("markdown list marker", re.compile(r"^\s{0,4}\d{1,2}\.\s", re.M)),
    ("table rule", re.compile(r"^\|[\s\-:|]+\|$", re.M)),
]
# The reference list carries volume, page and article numbers, which are bibliographic
# and have no source file. The whole section is removed rather than pattern-matched.
REF_SECTION = re.compile(r"^## References\s*$.*?(?=^## |\Z)", re.M | re.S)

NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?")


def _add(store, v, d):
    if v is None:
        return
    try:
        f = float(v)
    except (TypeError, ValueError):
        return
    if f != f or abs(f) == float("inf"):
        return
    store[d].add(f"{abs(f):.{d}f}")


def build_universe(verbose=False):
    """{plain: {d: {strings}}, pct: {d: {strings}}} plus the file list."""
    plain = {d: set() for d in range(MAX_DECIMALS + 1)}
    pct = {d: set() for d in range(MAX_DECIMALS + 1)}
    files, values = [], 0

    def take(f):
        nonlocal values
        for d in range(MAX_DECIMALS + 1):
            _add(plain, f, d)
            _add(pct, f * 100.0, d)
        values += 1

    def walk_json(o):
        if isinstance(o, dict):
            for v in o.values():
                walk_json(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk_json(v)
        elif isinstance(o, bool):
            pass
        elif isinstance(o, (int, float)):
            take(float(o))

    roots = sorted({p for p in REV.glob(f"*_{DS.VERSION}")} |
                   {p for p in REV.glob(f"*_{DS.VERSION}_*")})
    for base in roots + [FIGS]:
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.suffix not in (".csv", ".json") or p.stat().st_size > 60_000_000:
                continue
            files.append(str(p.relative_to(ROOT)))
            try:
                if p.suffix == ".json":
                    walk_json(json.loads(p.read_text()))
                else:
                    df = pd.read_csv(p, comment="#")
                    for c in df.columns:
                        s = pd.to_numeric(df[c], errors="coerce").dropna()
                        for f in s.unique():
                            take(float(f))
            except Exception as e:                       # a malformed file is a finding
                print(f"  ! could not read {p.relative_to(ROOT)}: "
                      f"{type(e).__name__}: {e}", flush=True)

    decl = Path(__file__).with_name("a36_verified_constants.json")
    declared = 0
    if decl.exists():
        files.append(str(decl.relative_to(ROOT)))
        o = json.loads(decl.read_text())
        for grp in ("constants", "superseded_values_quoted_as_superseded"):
            for e in o.get(grp, []):
                take(float(e["value"]))
                declared += 1
        if verbose:
            print(f"  declared constants: {declared}", flush=True)

    for p in SPEC_SOURCES:                                # specification constants
        if not p.exists():
            continue
        files.append(str(p.relative_to(ROOT)))
        for m in re.finditer(r"(?<![\w.])\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", p.read_text()):
            take(float(m.group(0)))

    if verbose:
        print(f"  universe: {values} distinct values from {len(files)} files", flush=True)
    return plain, pct, files


def strip_exempt(text):
    counts = {}
    text, n = REF_SECTION.subn(lambda m: " " * len(m.group(0)), text)
    counts["reference list (volume, page, article numbers)"] = n
    for name, rx in EXEMPT:
        text, n = rx.subn(lambda m: " " * len(m.group(0)), text)
        counts[name] = n
    return text, counts


def sweep(path, plain, pct):
    raw = path.read_text()
    text, counts = strip_exempt(raw)
    lines = raw.split("\n")
    starts, off = [], 0
    for ln in lines:
        starts.append(off)
        off += len(ln) + 1

    def line_of(pos):
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    seen, unmatched, matched = set(), [], 0
    for m in NUM.finditer(text):
        whole, frac = m.group(1).replace(",", ""), m.group(2) or ""
        d = len(frac)
        if d > MAX_DECIMALS:
            d = MAX_DECIMALS
            frac = frac[:MAX_DECIMALS]
        tok = f"{int(whole)}.{frac}" if frac else whole
        key = f"{int(whole)}" + (f".{frac}" if frac else "")
        probe = f"{float(key):.{d}f}"
        tail = text[m.end():m.end() + 2]
        is_pct = tail.startswith("%")
        hit = probe in plain[d] or (is_pct and probe in pct[d])
        if hit:
            matched += 1
            continue
        ln = line_of(m.start())
        ident = (tok, ln)
        if ident in seen:
            continue
        seen.add(ident)
        ctx = lines[ln - 1].strip()
        unmatched.append({"file": path.name, "line": ln, "token": tok + ("%" if is_pct else ""),
                          "context": ctx[:180]})
    return matched, unmatched, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(OUT / "CONSISTENCY_SWEEP.md"))
    ap.add_argument("--fail-over", type=int, default=None,
                    help="exit 1 if more than this many numbers are unmatched")
    args = ap.parse_args()

    print(f"  dataset version {DS.VERSION}", flush=True)
    plain, pct, files = build_universe(verbose=True)

    rows, total_matched, exempt_total = [], 0, {}
    for name in ("DRAFT_v8_9_main.md", "DRAFT_v8_9_supplement.md"):
        p = MS / name
        if not p.exists():
            raise SystemExit(f"missing manuscript: {p}")
        matched, un, counts = sweep(p, plain, pct)
        total_matched += matched
        rows += un
        for k, v in counts.items():
            exempt_total[k] = exempt_total.get(k, 0) + v
        print(f"  {name}: {matched} matched, {len(un)} unmatched", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["file", "line", "token", "context"])
    csv = Path(args.report).resolve().with_suffix(".csv")
    df.to_csv(csv, index=False)

    lines = [f"# Number sweep — DRAFT_v8_9 ({DS.VERSION})", "",
             f"Every numeric token in the main text and supplement, matched against "
             f"{len(files)} source files.", "",
             f"- matched: **{total_matched}**",
             f"- unmatched: **{len(df)}**", "",
             "**How to read this.** A *matched* number means some source file contains a value "
             "that rounds to it at the precision the prose used. With a universe this large that "
             "is weak evidence on its own: it rules out a number that exists nowhere, not a "
             "number copied from the wrong row. The useful output is the unmatched list, which is "
             "where rounds 46-49 each had at least one stale figure. Provenance — which file a "
             "given paragraph is entitled to read from — is the ledger's job, and the assertion "
             "in `a12_main_text_provenance.py` is what enforces it.", "",
             "## Exempt spans (removed before tokenizing)", "",
             "| pattern | occurrences |", "|---|---|"]
    for k, v in exempt_total.items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Unmatched numbers", ""]
    if df.empty:
        lines.append("None. Every number in both documents is reproduced by a source file.")
    else:
        lines += ["| file | line | number | context |", "|---|---|---|---|"]
        for r in df.itertuples():
            ctx = r.context.replace("|", "\\|")
            lines.append(f"| {r.file} | {r.line} | `{r.token}` | {ctx} |")
    rep = Path(args.report).resolve()
    rep.write_text("\n".join(lines) + "\n")
    try:
        shown = rep.relative_to(ROOT)
    except ValueError:                                  # a report written outside the repo
        shown = rep
    print(f"  wrote {shown} and {csv.name}", flush=True)

    if args.fail_over is not None and len(df) > args.fail_over:
        raise SystemExit(f"{len(df)} unmatched numbers, limit {args.fail_over}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
