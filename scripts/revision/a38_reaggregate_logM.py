#!/usr/bin/env python3
"""Round 51 item 1: apply the log-M fallback to the stored per-spectrum table.

`a32_primary_comparison.py` is fixed, but re-running it would re-fit 3000 spectra
for a change that touches only a derived column. It does not have to: the saved
`primary_per_spectrum.csv` carries each method's success flag and `logM_cond`,
which is everything the substitution needs. A successful fit's log M is already
right; a failed one takes `logM_cond`.

Idempotent -- run it twice and the second run reports nothing to do. It writes a
before/after record next to the table so the change is auditable, and refuses to
touch a table that does not carry the columns it needs.

This changes **only** the log-M columns of the fitting methods. Gamma and omega0
errors, failure counts, the successful-fit-only diagnostics and the 40.0% / 52.7%
gap fractions are computed from the fit parameters and are untouched; the script
asserts that the untouched columns are bit-identical.
"""

from __future__ import annotations

import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import dataset_paths as DS
from a32_primary_comparison import logM_with_fallback

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out = args.outdir or DS.rev("A23")
    p = out / "primary_per_spectrum.csv"
    if not p.exists():
        raise SystemExit(f"missing {p}")

    d = pd.read_csv(p)
    for c in ("logM_true", "logM_cond"):
        if c not in d.columns:
            raise SystemExit(f"{p.name} has no {c!r} column; cannot apply the fallback")
    methods = [c[len("ok_"):] for c in d.columns if c.startswith("ok_")]
    if not methods:
        raise SystemExit(f"{p.name} carries no ok_<method> columns")

    y = d.logM_true.to_numpy()
    lc = d.logM_cond.to_numpy()
    before_frame = d.copy()

    rows, changed = [], 0
    for m in methods:
        ok = d[f"ok_{m}"].to_numpy().astype(bool)
        v_before = d[f"logM_{m}"].to_numpy().astype(float)
        v_after = logM_with_fallback(ok, v_before, lc)
        n_moved = int(np.count_nonzero(~np.isclose(v_before, v_after, rtol=0, atol=1e-12)))
        changed += n_moved
        rows.append({
            "method": m, "n": len(d), "n_failed": int((~ok).sum()),
            "n_values_changed": n_moved,
            "MAE_logM_all_before": float(np.abs(v_before - y).mean()),
            "MAE_logM_all_after": float(np.abs(v_after - y).mean()),
            "max_abs_change_per_case": float(np.max(np.abs(v_after - v_before))) if len(d) else 0.0,
            "MAE_logM_ok_only": float(np.abs(v_before[ok] - y[ok]).mean()),
        })
        d[f"logM_{m}"] = v_after

    rep = pd.DataFrame(rows)
    rep["delta"] = rep.MAE_logM_all_after - rep.MAE_logM_all_before
    print(rep[["method", "n_failed", "n_values_changed", "MAE_logM_all_before",
               "MAE_logM_all_after", "delta", "max_abs_change_per_case"]]
          .round(6).to_string(index=False), flush=True)

    # Nothing outside the fitting methods' log-M columns may move.
    touched = {f"logM_{m}" for m in methods}
    for c in d.columns:
        if c in touched:
            continue
        a, b = before_frame[c].to_numpy(), d[c].to_numpy()
        if a.dtype.kind in "fc":
            assert np.array_equal(a, b, equal_nan=True), f"column {c} changed"
        else:
            assert (a == b).all(), f"column {c} changed"

    if changed == 0:
        print("  nothing to do: the fallback is already applied", flush=True)
        return 0
    if args.dry_run:
        print(f"  dry run: {changed} value(s) would change", flush=True)
        return 0

    d.to_csv(p, index=False)
    rec = out / "logM_fallback_reaggregation.json"
    rec.write_text(json.dumps({
        "task": "A38_logM_fallback", "dataset_version": DS.VERSION,
        "table": str(p.relative_to(ROOT)),
        "what": ("failed fits' log M replaced by the conditional median of log M; "
                 "previously log merit(median omega0, median Gamma)"),
        "refitted": False,
        "source_of_truth": "success flags and logM_cond already in the table",
        "per_method": rows,
        "untouched": ("every other column asserted bit-identical: omega0/Gamma "
                      "predictions, success flags, network columns, logM_cond, logM_true"),
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n")
    print(f"  wrote {p.relative_to(ROOT)} ({changed} values) and {rec.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
