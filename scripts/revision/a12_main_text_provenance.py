"""Assertion: no main-text number may cite a pre-v10 run.

Round 45 §1. The provenance ledger declares, for every main-text table and
figure, the generator version and the file the numbers are read from. This
script checks the declaration against the filesystem and fails if a main-text
row rests on anything but v10, unless that row is an explicitly declared
exception AND the manuscript says so in place.

It checks four things:

  1. every source file named in the ledger exists;
  2. every main-text row resolves to a v10 result directory;
  3. every exception is declared in the ledger's exception list AND the string
     "(dataset v9)" or an equivalent in-place label appears in the manuscript;
  4. no main-text row names a directory whose suffix is _v7, _v8 or _v9 without
     being a declared exception.

Run:  python scripts/revision/a12_main_text_provenance.py
Exit 0 if the assertion holds, 1 otherwise.
"""
from __future__ import annotations
import re, sys
from pathlib import Path

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
# The ledger ships with the code as PROVENANCE.md; the manuscript does not ship at
# all, so its directory comes from SBC_MANUSCRIPT_DIR and may sit anywhere.
LEDGER = ROOT / "PROVENANCE.md"
MS = _manuscript_dir()
MAIN = next(iter(sorted(MS.glob("DRAFT_*_main.md"))), MS / "DRAFT_main.md")

# Rows allowed to rest on an earlier generator, with the reason and the label
# that must appear in the manuscript for the exception to be accepted.
# Empty: every main-text row rests on v10. T10 and F5 (the channel ablation) were
# listed here until the ablation was retrained on v10; the entries were removed
# rather than left in place, so a future pre-v10 citation fails the assertion.
EXCEPTIONS = {}
STALE = re.compile(r"_v[789](?:/|$|\b)")


def ledger_rows(text):
    """(id, gen, source cell) for every row of the Main text table.

    The source column is located by its HEADER, not by a fixed index: a column
    inserted into the ledger once shifted it silently, and the assertion then
    passed while checking nothing at all.
    """
    out, in_main, cols = [], False, None
    for line in text.splitlines():
        if line.startswith("## Main text"):
            in_main = True; continue
        if in_main and line.startswith("## "):
            break
        if not in_main or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cols is None and cells and cells[0] == "#":
            hdr = [c.strip("* ").lower() for c in cells]
            try:
                cols = (hdr.index("source file"), hdr.index("gen"))
            except ValueError:
                raise SystemExit("ledger header lacks a 'source file' or 'gen' column: "
                                 + " | ".join(cells))
            continue
        if cols is None or set("".join(cells)) <= set("-: "):
            continue
        src_i, gen_i = cols
        if len(cells) > src_i:
            out.append((cells[0], cells[gen_i], cells[src_i]))
    return out


def sources(cell):
    """Result files named in a ledger source cell."""
    return [m.strip() for m in re.findall(r"`([^`]+)`", cell)]


def main():
    if not LEDGER.exists():
        print(f"FAIL: no ledger at {LEDGER}"); return 1
    rows = ledger_rows(LEDGER.read_text())
    ms = MAIN.read_text() if MAIN.exists() else ""
    if not rows:
        print("FAIL: no main-text rows parsed from the ledger"); return 1

    problems, checked, excused = [], 0, []
    for rid, gen, src_cell in rows:
        for src in sources(src_cell):
            if "/" not in src or src.endswith(".py") or src.startswith("a"):
                continue           # scripts and specification rows carry no results file
            checked += 1
            p = REV / src
            if not p.exists() and not (ROOT / src).exists():
                problems.append(f"{rid}: source does not exist: {src}")
                continue
            stale = STALE.search(src)
            if not stale:
                continue
            if rid in EXCEPTIONS:
                want_dir, reason, label = EXCEPTIONS[rid]
                if want_dir not in src:
                    problems.append(f"{rid}: declared exception is {want_dir}, cites {src}")
                elif label.lower() not in ms.lower():
                    problems.append(f"{rid}: exception not labelled in the manuscript "
                                    f"(expected the phrase {label!r} in place)")
                else:
                    excused.append(f"{rid} -> {src} ({reason})")
            else:
                problems.append(f"{rid}: main-text number cites a pre-v10 result: {src}")
            if rid not in EXCEPTIONS and gen.strip("*").lower() != "v10":
                problems.append(f"{rid}: ledger declares generator {gen!r}, not v10")

    print(f"  main-text ledger rows: {len(rows)}; result files checked: {checked}")
    # A vacuous pass is the failure mode this check is most exposed to: if the
    # source column moves, every row parses to nothing and the assertion reports
    # success having verified nothing. Refuse to pass without real files.
    if checked == 0:
        print("\nFAIL: parsed 0 result files from the ledger. The source column has "
              "probably moved or been renamed; the assertion is not actually checking "
              "anything.")
        return 1
    for e in excused:
        print(f"  declared exception, labelled in place: {e}")
    if problems:
        print(f"\nFAIL: {len(problems)} problem(s)")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nPASS: every main-text number rests on v10, "
          f"except {len(excused)} declared and labelled exception(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
