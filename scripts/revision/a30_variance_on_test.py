"""Variance components on one common evaluation set (round 46 item 7).

The S5.3 table was computed from each run's own validation score. Those come
from three DIFFERENT validation splits, one per dataset realization, so the
between-realization mean square confounds two things: how much the trained model
changes with the realization, and how much the evaluation set changes with it.
Only the first is what the table claims to report.

This recomputes the same one-way random-effects decomposition from the nine runs
per model evaluated on the single fixed `test_v10` set, which every run sees
identically. No retraining: the scores are the existing A22 evaluations.

Both tables are written so the difference is inspectable.
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from realization_stats import variance_components
import dataset_paths as DS

ORDER = ["cnn_kernel45", "tf_patch60", "fusion", "cnn", "5a", "5b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--eval-set", type=str, default="test")
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()
    out = args.outdir or DS.rev("A21"); out.mkdir(parents=True, exist_ok=True)

    src = DS.rev("A22") / "test_eval_runs.csv"
    d = pd.read_csv(src)
    d = d[(d.eval_set == args.eval_set) & (d.severity == args.severity)]
    rows = []
    for m in ORDER:
        g = d[d.model == m]
        if len(g) != 9:
            print(f"  WARNING {m}: {len(g)} cells, expected 9"); continue
        vc = variance_components(g, value="MAE_logM")
        if vc is None:
            print(f"  WARNING {m}: degenerate grid"); continue
        rows.append({"model": m, **vc})
    t = pd.DataFrame(rows)
    t["eval_set"] = f"{args.eval_set}_{DS.VERSION}"
    t["severity"] = args.severity
    t.to_csv(out / "variance_components_test.csv", index=False)

    old = pd.read_csv(out / "variance_components.csv").set_index("model")
    print(f"{'model':14s} {'mean':>8s} {'SD9':>8s} {'SD_train':>9s} {'SD_data':>8s} "
          f"{'share':>7s}   |  val-split share")
    for r in t.itertuples():
        o = old.loc[r.model]
        print(f"{r.model:14s} {r.mean:8.4f} {r.sd_over_cells:8.4f} {r.sd_training_seed:9.4f} "
              f"{r.sd_dataset_seed:8.4f} {r.frac_dataset_seed:7.1%}   |  {o.frac_dataset_seed:7.1%}")

    det = int(t.dataset_component_detectable.sum())
    fr = t[t.dataset_component_detectable].frac_dataset_seed
    print(f"\n  dataset component non-zero for {det} of {len(t)} architectures")
    if len(fr):
        print(f"  share range among those: {fr.min():.1%} to {fr.max():.1%}")

    (out / "run_sidecar_variance_test.json").write_text(json.dumps({
        "task": "A30_variance_on_test", "dataset_version": DS.VERSION,
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "source": str(src.relative_to(ROOT)),
        "eval_set": f"{args.eval_set}_{DS.VERSION}", "severity": args.severity,
        "estimator": "one-way random effects (ANOVA), dataset component truncated at 0",
        "why": ("the val-split version confounds realization effects with evaluation-set "
                "effects, because each realization has its own validation split; this "
                "version uses the single fixed test set every run is scored on"),
        "training": "none; reuses the A22 evaluations of the existing A21 checkpoints",
        "date_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"  wrote {out}/variance_components_test.csv")


if __name__ == "__main__":
    main()
