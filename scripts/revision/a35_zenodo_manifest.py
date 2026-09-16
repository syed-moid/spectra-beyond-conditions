"""Deposit manifest: the plotting inputs that reproduce the submitted figures.

Round 48 item 22 / round 49 item 11. For every figure, records the inputs its CSV
header declares, with a SHA-256 of each, so a depositor can verify that the files
in the deposit are the files the figures were drawn from.

Also re-checks freshness: every figure PDF must be newer than every input it
names. A stale PDF with correct-looking inputs is the failure this catches.
"""
from __future__ import annotations
import hashlib, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "figures"
OUT = ROOT / "ZENODO_MANIFEST.md"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    csvs = sorted(FIGS.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"no figure CSVs in {FIGS}")
    inputs, rows, stale, missing, extras = {}, [], [], [], []
    for c in csvs:
        stem = c.stem
        pdf = c.with_suffix(".pdf")
        hdr = [l[1:].strip() for l in c.read_text().splitlines() if l.startswith("#")]
        # A figure's plotting data is written by save() and carries a "# figure:" header.
        # Other CSVs in this directory are figure artefacts -- fig7's per-panel parameter
        # table, for one -- and are deposited, but they are not figures and must not be
        # counted as one (round 50).
        if not any(h.startswith("figure:") for h in hdr):
            extras.append(stem)
            continue
        got = [h.split(":", 1)[1] for h in hdr if h.startswith("inputs:")]
        ins = [i.strip() for i in got[0].split(";")] if got else []
        for i in ins:
            p = ROOT / i
            if not p.exists():
                missing.append((stem, i)); continue
            inputs.setdefault(i, None)
            if pdf.exists() and pdf.stat().st_mtime < p.stat().st_mtime:
                stale.append((stem, i))
        rows.append((stem, pdf.exists(), ins))
    for i in list(inputs):
        inputs[i] = sha(ROOT / i)

    L = ["# Zenodo deposit manifest — plotting inputs",
         "",
         "Every file below is an input that a submitted figure declares in its CSV header.",
         "A deposit containing these files, at these hashes, reproduces the figures in the",
         "package when passed through `scripts/revision/generate_figures_v10.py`.",
         "", f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}.", "",
         "## Figures and the inputs they declare", "",
         "| figure | PDF present | inputs |", "|---|---|---|"]
    for stem, has_pdf, ins in rows:
        L.append(f"| `{stem}` | {'yes' if has_pdf else '**NO**'} | " +
                 ("<br>".join(f"`{i}`" for i in ins) if ins else "—") + " |")
    L += ["", f"## Input files and SHA-256 ({len(inputs)} distinct)", "",
          "| file | sha256 |", "|---|---|"]
    for i in sorted(inputs):
        L.append(f"| `{i}` | `{inputs[i]}` |")
    L += ["", "## Checks", ""]
    L.append(f"- figures with a PDF: **{sum(1 for _, h, _ in rows if h)} / {len(rows)}**")
    if extras:
        L.append(f"- additional figure artefacts deposited (not figures): "
                 f"**{len(extras)}** — {', '.join(extras)}")
    L.append(f"- declared inputs missing from the tree: "
             f"**{len(missing)}**" + (f" — {missing}" if missing else ""))
    L.append(f"- figures older than one of their inputs (stale): "
             f"**{len(stale)}**" + (f" — {stale}" if stale else ""))
    if not missing and not stale:
        L.append("")
        L.append("Every figure is present and newer than every input it names.")
    OUT.write_text("\n".join(L) + "\n")
    print(f"  {len(rows)} figures, {len(inputs)} distinct inputs, {len(extras)} artefact file(s)")
    print(f"  missing: {len(missing)}   stale: {len(stale)}")
    print(f"  wrote {OUT.relative_to(ROOT)}")
    return 1 if (missing or stale) else 0


if __name__ == "__main__":
    sys.exit(main())
