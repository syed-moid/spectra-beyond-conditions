"""Ranking-boundary threshold sensitivity: the severity at which rho crosses 0.3 / 0.5 / 0.7.

Round-38 repair 1. **Practical boundaries are interpolated on `rho_mean`** -- the
central estimate -- and those are the columns the manuscript quotes. The
`*_hi` columns repeat the interpolation on `rho_hi`, the upper bound of the
within-tuple rho CI, which is what the ad hoc v8 table used; they are kept only
so the v8 -> v9 comparison can be made like-for-like on either convention.

The statistical boundary is unchanged: the first severity at which the
tuple-bootstrap CI includes zero, carried from `two_boundaries.csv`.

  * interpolation is linear in severity (not log-severity) between the two grid
    points that bracket the first downward crossing.

The `*_hi` columns reproduce the stored v8 table to 4.4e-16 on all 6 models x 3
thresholds; run with --validate to repeat that check.

`value_crossover_severity` and `ranking_statistical` are carried from
`two_boundaries.csv` so the table is self-contained for figure 3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_paths as DS  # noqa: E402

THRESHOLDS = (0.3, 0.5, 0.7)
CURVE = "rho_mean"          # practical boundaries, quoted in the manuscript
CURVE_ALT = "rho_hi"        # the ad hoc v8 convention, kept for comparison


def crossing(severity, rho, threshold):
    """First downward crossing of `threshold`, linear in severity."""
    for i in range(len(rho) - 1):
        if rho[i] >= threshold >= rho[i + 1]:
            span = rho[i] - rho[i + 1]
            f = (rho[i] - threshold) / span if span else 0.0
            return float(severity[i] + f * (severity[i + 1] - severity[i]))
    return float("nan")


def build(a14_dir: Path) -> pd.DataFrame:
    rho = pd.read_csv(a14_dir / "extended_rho_replicate.csv")
    two = pd.read_csv(a14_dir / "two_boundaries.csv").set_index("model")
    rows = []
    for model in two.index:
        g = rho[rho.model == model].sort_values("severity")
        if g.empty:
            continue
        sev = g.severity.values
        row = {"model": model}
        for t in THRESHOLDS:
            row[f"rho_{t}"] = crossing(sev, g[CURVE].values, t)
        for t in THRESHOLDS:
            row[f"rho_{t}_hi"] = crossing(sev, g[CURVE_ALT].values, t)
        row["value_crossover_severity"] = two.loc[model, "value_crossover_severity"]
        rank = two.loc[model, "ranking_crossover_severity"]
        row["ranking_statistical"] = (f">{sev.max():g}" if pd.isna(rank) else f"{rank:g}")
        rows.append(row)
    return pd.DataFrame(rows)


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=None, help="A14 directory (default: active version)")
    ap.add_argument("--validate", action="store_true",
                    help="rebuild the v8 table and diff against the stored one")
    args = ap.parse_args()

    if args.validate:
        v8 = DS.REV / "A14"
        got = build(v8).set_index("model")
        ref = pd.read_csv(v8 / "boundaries_threshold_sensitivity.csv").set_index("model")
        cols = [f"rho_{t}_hi" for t in THRESHOLDS]
        ref = ref.rename(columns={f"rho_{t}": f"rho_{t}_hi" for t in THRESHOLDS})
        worst = max(float(np.abs(got.loc[m, c] - ref.loc[m, c])) for m in ref.index for c in cols)
        print(f"  v8 validation: {len(ref)} models x {len(cols)} thresholds, "
              f"worst |diff| = {worst:.2e}")
        same = all(str(got.loc[m, c2]) == str(ref.loc[m, c2])
                   for m in ref.index for c2 in ("value_crossover_severity", "ranking_statistical"))
        print(f"  carried columns identical: {same}")
        return 0 if worst < 1e-12 and same else 1

    out = args.dir or DS.rev("A14")
    out.mkdir(parents=True, exist_ok=True)
    df = build(out)
    df.to_csv(out / "boundaries_threshold_sensitivity.csv", index=False)
    (out / "run_sidecar_boundary_thresholds.json").write_text(json.dumps({
        "task": "A14_boundary_thresholds",
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_commit": git_sha(), "dataset_version": DS.VERSION,
        "practical_curve": CURVE, "comparison_curve": CURVE_ALT,
        "thresholds": list(THRESHOLDS),
        "interpolation": "linear in severity",
        "statistical_boundary": "first severity at which the tuple-bootstrap CI includes zero",
        "date_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
