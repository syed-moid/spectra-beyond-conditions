"""A3c - combine the A3 / A3b baselines with paired-bootstrap CIs.

Produces one table over the stress base: for each method and severity, the
MAE_logM (failures replaced by the metadata-only prediction) plus paired
bootstrap 95% CIs of the difference against (a) ST-5a and (b) the metadata-only
reference (A2 mlp_l1, severity-independent).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import dataset_paths as DS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
A3 = DS.rev("A3")
A2_PRED = ROOT / "results" / "revision" / "A2" / "predictions_seedavg.csv"
SEVERITIES = (0.25, 0.5, 1.0, 2.0, 4.0)


def paired(a, b, n_boot, rng):
    d = a - b
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--dir", type=Path, default=A3)
    ap.add_argument("--dataset-version", type=str, default=DS.VERSION)
    ap.add_argument("--a2-pred", type=Path, default=A2_PRED)
    ap.add_argument("--metadata-model", type=str, default="mlp_l1")
    ap.add_argument("--reference", type=str, default="ST-5a",
                    help="model every other row is compared against")
    args = ap.parse_args()
    A3D = args.dir
    rng = np.random.default_rng(args.seed)

    fits = pd.concat([pd.read_csv(A3D / "fit_results_per_spectrum.csv"),
                      pd.read_csv(A3D / "fit_results_extra_variants.csv")], ignore_index=True)
    st = pd.read_csv(A3D / "st_predictions_stress_base.csv")
    feat = pd.read_csv(A3D / "feature_gbm_predictions.csv")
    a2 = pd.read_csv(args.a2_pred)
    meta = a2[(a2.model == args.metadata_model) & (a2.split == "stress_base")]
    y = meta["logM_true"].to_numpy()
    n = len(y)
    err_meta = np.abs(meta["logM_pred_seedavg"].to_numpy() - y)

    # per-severity per-spectrum absolute errors for every method
    err = {}
    for sev in SEVERITIES:
        e = {}
        for m in fits.method.unique():
            d = fits[(fits.method == m) & (fits.severity == sev)].sort_values("idx")
            ok = d.fit_ok.astype(bool).to_numpy() & np.isfinite(d.logM_fit.to_numpy())
            pred = np.where(ok, d.logM_fit.to_numpy(), d.logM_metadata_fallback.to_numpy())
            e[m] = np.abs(pred - d.logM_true.to_numpy())
        for arch in sorted(st.model.unique()):
            d = st[(st.model == arch) & (st.severity == sev)]
            pred = d.groupby("idx")["logM_pred"].mean().to_numpy()
            e[arch] = np.abs(pred - y)
        d = feat[feat.severity == sev]
        e["feat_gbm"] = np.abs(d.groupby("idx")["logM_pred"].mean().to_numpy() - y)
        e["metadata_mlp_l1"] = err_meta
        err[sev] = e

    rows = []
    preferred = ["metadata_mlp_l1", "dho_single", "dho_single_win", "dho_single_track",
                 "dho_two_mode", "dho_matched", "feat_gbm"]
    order = ([m for m in preferred if m in err[SEVERITIES[0]]]
             + [m for m in sorted(err[SEVERITIES[0]]) if m not in preferred])
    ref = args.reference
    for sev in SEVERITIES:
        for m in order:
            a = err[sev][m]
            d1, lo1, hi1 = paired(a, err[sev][ref], args.n_boot, rng)
            d2, lo2, hi2 = paired(a, err[sev]["metadata_mlp_l1"], args.n_boot, rng)
            rows.append({"dataset_version": args.dataset_version,
                         "method": m, "severity": sev, "N": n, "MAE_logM": float(a.mean()),
                         "reference_model": ref,
                         "vs_ref_diff": d1, "vs_ref_lo": lo1, "vs_ref_hi": hi1,
                         "vs_metadata_diff": d2, "vs_metadata_lo": lo2, "vs_metadata_hi": hi2,
                         "beats_metadata_95CI": bool(hi2 < 0.0),
                         "beats_reference_95CI": bool(hi1 < 0.0)})
    df = pd.DataFrame(rows)
    df.to_csv(A3D / "paired_comparison_stress_base.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    (A3D / "run_sidecar_paired.json").write_text(json.dumps(
        {"task": "A3c", "seed": args.seed, "n_boot": args.n_boot,
         "dataset_version": args.dataset_version, "reference_model": ref,
         "metadata_reference": f"A2 {args.metadata_model} seed-averaged on stress_base",
         "failure_handling": "fit failures replaced by the metadata-only prediction"}, indent=2))


if __name__ == "__main__":
    main()
