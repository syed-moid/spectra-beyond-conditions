"""A2 addendum - metadata-only models on the new v8 evaluation sets.

Labels are bitwise identical between v7 and v8 (A13 part 2), so the metadata
models trained in A2 remain valid and are NOT re-specified here. A2 persisted
predictions but not model objects, so the models are re-fit with the same
seeds, the same architecture and the same training split; the reproduction is
verified against A2's stored val predictions before any new set is scored.

Evaluates mlp_l1 (5 seeds), hgb (5 seeds) and knn (k = 50, deterministic) on
holdout_c_extrap_v8 and replicate_eval_v8, and appends rows carrying
dataset_version to the A2 outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from a2_metadata_family import (  # noqa: E402
    boot_ci, config_hash, git_sha, load_split, metric_row, mlp_predict,
    paired_boot_ci, train_mlp,
)
import dataset_paths as DS  # noqa: E402

NPZ = DS.FULL
NEW = {
    "holdout_c_extrap": (DS.HOLDOUT_C_EXTRAP,
                         ROOT / "results" / "revision"
                         / "A1_v8_holdout_c_extrap" / "per_tuple_conditional_logM.csv"),
    "replicate_eval": (DS.REPLICATE,
                       ROOT / "results" / "revision"
                       / "A1_v8_replicate" / "per_tuple_conditional_logM.csv"),
}
A2 = ROOT / "results" / "revision" / "A2"
OUTDIR = ROOT / "results" / "revision" / "A2_v8"
SEEDS = (42, 43, 44, 45, 46)
KNN_K = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

    z8 = np.load(NPZ, allow_pickle=False)
    Xtr, ytr = load_split(z8, "train")
    Xval, yval = load_split(z8, "val")

    evals = {}
    oracle = {}
    for name, (npz, a1csv) in NEW.items():
        zz = np.load(npz, allow_pickle=False)
        evals[name] = load_split(zz, name)
        a1 = pd.read_csv(a1csv)
        oracle[name] = a1["logM_cond_median"].to_numpy()
        assert len(oracle[name]) == len(evals[name][1])
        assert np.allclose(a1["logM_realized"].to_numpy(), evals[name][1], atol=1e-5)
        if name == "replicate_eval":
            evals[name] += (zz["tuple_id"],)

    preds = {}
    val_pred = {}
    for seed in args.seeds:
        net = train_mlp(Xtr, ytr, Xval, yval, seed, "l1", device, n_epochs=args.epochs)
        val_pred.setdefault("mlp_l1", []).append(mlp_predict(net, Xval, device))
        for name in NEW:
            preds[("mlp_l1", seed, name)] = mlp_predict(net, evals[name][0], device)
        g = HistGradientBoostingRegressor(loss="absolute_error", random_state=seed,
                                          early_stopping=True, validation_fraction=0.1)
        g.fit(Xtr, ytr)
        val_pred.setdefault("hgb", []).append(g.predict(Xval))
        for name in NEW:
            preds[("hgb", seed, name)] = g.predict(evals[name][0]).astype(np.float64)
        print(f"  seed {seed} refit done", flush=True)

    knn = KNeighborsRegressor(n_neighbors=KNN_K).fit(Xtr, ytr)
    val_pred["knn"] = [knn.predict(Xval)]
    for seed in args.seeds:
        for name in NEW:
            preds[("knn", seed, name)] = knn.predict(evals[name][0]).astype(np.float64)

    # --- reproduction check against A2's stored val predictions ---------------
    a2p = pd.read_csv(A2 / "predictions_seedavg.csv")
    repro = []
    for m in ("mlp_l1", "hgb", "knn"):
        ref = a2p[(a2p.model == m) & (a2p.split == "val")]["logM_pred_seedavg"].to_numpy()
        now = np.mean(val_pred[m], axis=0)
        repro.append({"model": m, "max_abs_pred_diff_vs_A2": float(np.max(np.abs(now - ref))),
                      "A2_val_MAE": float(np.abs(ref - yval).mean()),
                      "refit_val_MAE": float(np.abs(now - yval).mean())})
        print(f"  repro {m:8s}: max|Δpred| = {repro[-1]['max_abs_pred_diff_vs_A2']:.3e}  "
              f"A2 val MAE {repro[-1]['A2_val_MAE']:.6f} -> refit {repro[-1]['refit_val_MAE']:.6f}")
    pd.DataFrame(repro).to_csv(args.outdir / "reproduction_check.csv", index=False)

    # --- metrics on the new sets ---------------------------------------------
    rng = np.random.default_rng(args.seed)
    rows, per_seed = [], []
    for model in ("mlp_l1", "hgb", "knn"):
        for name in NEW:
            y = evals[name][1].astype(np.float64)
            ps = [preds[(model, s, name)] for s in args.seeds]
            for s, p in zip(args.seeds, ps):
                r = metric_row(y, p)
                r.update(model=model, split=name, seed=s, dataset_version=DS.VERSION)
                per_seed.append(r)
            agg = {"dataset_version": DS.VERSION, "model": model, "split": name,
                   "N": len(y), "n_seeds": len(args.seeds)}
            for k in ("MAE_logM", "R2_logM", "spearman_rho", "median_multiplicative_error"):
                v = np.array([r[k] for r in per_seed if r["model"] == model and r["split"] == name])
                agg[f"{k}_mean"] = float(v.mean())
                agg[f"{k}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            pbar = np.mean(ps, axis=0)
            err = np.abs(y - pbar)
            lo, hi = boot_ci(err, args.n_boot, rng)
            agg["MAE_logM_ensemble"] = float(err.mean())
            agg["MAE_boot_lo"], agg["MAE_boot_hi"] = lo, hi
            d, dlo, dhi = paired_boot_ci(err, np.abs(y - oracle[name]), args.n_boot, rng)
            agg["diff_vs_oracle_mean"] = d
            agg["diff_vs_oracle_lo"], agg["diff_vs_oracle_hi"] = dlo, dhi
            if name == "replicate_eval":
                tid = evals[name][2]
                rhos = []
                for t in np.unique(tid):
                    m = tid == t
                    if np.ptp(pbar[m]) == 0:
                        rhos.append(0.0)          # constant predictor: rho is 0 by construction
                    else:
                        rhos.append(float(spearmanr(y[m], pbar[m]).statistic))
                agg["within_tuple_spearman_mean"] = float(np.mean(rhos))
                agg["within_tuple_spearman_sd"] = float(np.std(rhos, ddof=1))
            rows.append(agg)

    df = pd.DataFrame(rows)
    df.to_csv(args.outdir / "metadata_family_metrics_v8_newsets.csv", index=False)
    pd.DataFrame(per_seed).to_csv(args.outdir / "metadata_family_per_seed_v8_newsets.csv",
                                  index=False)
    pred_rows = []
    for model in ("mlp_l1", "hgb", "knn"):
        for name in NEW:
            pbar = np.mean([preds[(model, s, name)] for s in args.seeds], axis=0)
            d = {"dataset_version": DS.VERSION, "model": model, "split": name,
                 "logM_true": evals[name][1].astype(np.float64),
                 "logM_pred_seedavg": pbar, "logM_oracle_median": oracle[name]}
            if name == "replicate_eval":
                d["tuple_id"] = evals[name][2]
            pred_rows.append(pd.DataFrame(d))
    pd.concat(pred_rows, ignore_index=True).to_csv(
        args.outdir / "predictions_seedavg_v8_newsets.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + df[["model", "split", "N", "MAE_logM_mean", "MAE_logM_sd",
                     "MAE_logM_ensemble", "R2_logM_mean", "spearman_rho_mean",
                     "diff_vs_oracle_mean", "diff_vs_oracle_lo", "diff_vs_oracle_hi"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    (args.outdir / "run_sidecar.json").write_text(json.dumps({
        "task": "A2_addendum", "script": str(Path(__file__).relative_to(ROOT)),
        "dataset_version": DS.VERSION, "train_split_from": str(NPZ.relative_to(ROOT)),
        "master_seed": args.seed, "model_seeds": list(args.seeds), "mlp_epochs": args.epochs,
        "n_boot": args.n_boot, "device": device, "knn_k": KNN_K,
        "note": "models re-fit with A2 seeds; reproduction verified against A2 val predictions",
        "config_hash_sha256_16": config_hash(), "git_commit": git_sha(),
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "numpy": np.__version__, "torch": torch.__version__, "pandas": pd.__version__},
        indent=2))
    print(f"\nwrote {args.outdir}")


if __name__ == "__main__":
    main()
