"""A5 negative controls + A14 within-condition ranking, on replicate_eval_v8.

A14 metric
    For each of the 250 condition tuples, the Spearman rho between predicted and
    realized log M across that tuple's 20 latent realizations, averaged over
    tuples, at s in {0.5, 1, 2, 4}. Also for omega0 and Gamma where the model
    exposes those heads. A metadata-only model is constant within a tuple and
    scores 0 by construction; this metric therefore isolates realization-level
    information, which is the paper's actual question.

A5 within-condition shuffle
    Permute spectra among the 20 realizations of each tuple with a DERANGEMENT
    (no realization keeps its own spectrum), fixed seed, targets left in place.
    Any model whose apparent skill came from conditions rather than lineshape is
    unaffected; a model reading realization-level structure must collapse to the
    metadata floor in MAE and to rho ~ 0 in ranking.

Label permutation is handled separately by a5_label_permutation.py.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from a4_eval import load_a4_model  # noqa: E402
from replicate_io import block_key, load_replicate  # noqa: E402
import dataset_paths as DS

OUTDIR = DS.rev("A14")
A2_V8 = (ROOT / "results" / "revision" / "A2_v8"
         / "predictions_seedavg_v8_newsets.csv")
SEVERITIES = (0.5, 1.0, 2.0, 4.0)
MAIN = [("5a", "v7", (42, 43, 44, 45, 46)), ("5b", "v7", (42, 43, 44, 45, 46)),
        ("fusion", "v7", (42, 43, 44, 45, 46)), ("cnn", "v7", (42, 43, 44, 45, 46))]


def derangement(rng, k):
    """A permutation of range(k) with no fixed point."""
    while True:
        p = rng.permutation(k)
        if not np.any(p == np.arange(k)):
            return p


def per_tuple_rho(tid, y, pred):
    rhos = []
    for t in np.unique(tid):
        m = tid == t
        if np.ptp(pred[m]) == 0 or np.ptp(y[m]) == 0:
            rhos.append(0.0)
        else:
            r = spearmanr(y[m], pred[m]).statistic
            rhos.append(0.0 if not np.isfinite(r) else float(r))
    return np.array(rhos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--severities", type=float, nargs="+", default=list(SEVERITIES))
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--ckpt-dir", type=Path,
                    default=DS.rev("A4_extended") / "checkpoints")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    clean, blocks = load_replicate()
    tid = clean["tuple_id"]
    T, c, E = clean["T_K"].astype(float), clean["c_pct"].astype(float), clean["E_kVcm"].astype(float)
    logM = np.log(np.clip(clean["M"].astype(float), 1e-9, None))
    om_t, gm_t = clean["omega_Q"].astype(float), clean["Gamma_Q"].astype(float)
    log_gm_t = np.log(np.clip(gm_t, 1e-9, None))
    n = len(tid)

    # derangement index per tuple, fixed seed, shared across models and severities
    rng = np.random.default_rng(args.seed)
    shuffle_idx = np.arange(n)
    for t in np.unique(tid):
        pos = np.nonzero(tid == t)[0]
        shuffle_idx[pos] = pos[derangement(rng, len(pos))]
    assert not np.any(shuffle_idx == np.arange(n))
    assert np.array_equal(tid[shuffle_idx], tid)

    rows, rho_rows = [], []
    for arch, protocol, seeds in MAIN:
        models = [load_a4_model(arch, protocol, s, device=args.device,
                                ckpt_dir=args.ckpt_dir) for s in seeds]
        for sev in args.severities:
            X = blocks[block_key(sev)]
            batch = {"spectrum": X, "T_K": T, "c_pct": c, "E_kVcm": E}
            batch_shuf = {"spectrum": X[shuffle_idx], "T_K": T, "c_pct": c, "E_kVcm": E}
            for label, b in (("unshuffled", batch), ("within_condition_shuffled", batch_shuf)):
                per_seed_mae, per_seed_rho = [], []
                preds = []
                for m in models:
                    out = m.predict_all(b)
                    p = np.log(np.clip(out["M"], 1e-9, None))
                    preds.append(p)
                    per_seed_mae.append(float(np.abs(p - logM).mean()))
                    per_seed_rho.append(float(per_tuple_rho(tid, logM, p).mean()))
                pbar = np.mean(preds, axis=0)
                out = models[0].predict_all(b)
                rows.append({
                    "dataset_version": DS.VERSION, "model": f"{arch}_{protocol}", "condition": label,
                    "severity": sev, "N": n, "n_seeds": len(seeds),
                    "MAE_logM_mean": float(np.mean(per_seed_mae)),
                    "MAE_logM_sd": float(np.std(per_seed_mae, ddof=1)),
                    "MAE_logM_ensemble": float(np.abs(pbar - logM).mean()),
                    "within_tuple_rho_logM_mean": float(np.mean(per_seed_rho)),
                    "within_tuple_rho_logM_sd": float(np.std(per_seed_rho, ddof=1)),
                })
                if label == "unshuffled":
                    for tgt_name, tgt, key, tf in (
                            ("logM", logM, "M", lambda v: np.log(np.clip(v, 1e-9, None))),
                            ("omega0", om_t, "omega_Q", lambda v: v),
                            ("log_Gamma", log_gm_t, "Gamma_Q",
                             lambda v: np.log(np.clip(v, 1e-9, None)))):
                        rr = [per_tuple_rho(tid, tgt, tf(m.predict_all(b)[key])).mean()
                              for m in models]
                        rho_rows.append({
                            "dataset_version": DS.VERSION, "model": f"{arch}_{protocol}",
                            "target": tgt_name, "severity": sev,
                            "rho_mean": float(np.mean(rr)), "rho_sd": float(np.std(rr, ddof=1))})
            print(f"  {arch}/{protocol} s={sev}: "
                  f"MAE {rows[-2]['MAE_logM_mean']:.4f} -> shuffled {rows[-1]['MAE_logM_mean']:.4f} | "
                  f"rho {rows[-2]['within_tuple_rho_logM_mean']:+.4f} -> "
                  f"{rows[-1]['within_tuple_rho_logM_mean']:+.4f}", flush=True)

    # metadata reference rows: constant within a tuple, so rho is 0 by construction
    if A2_V8.exists():
        a2 = pd.read_csv(A2_V8)
        for model in ("mlp_l1", "hgb", "knn"):
            d = a2[(a2.model == model) & (a2.split == "replicate_eval")]
            if d.empty:
                continue
            p = d["logM_pred_seedavg"].to_numpy()
            for sev in args.severities:
                rows.append({"dataset_version": DS.VERSION, "model": f"metadata_{model}",
                             "condition": "unshuffled", "severity": sev, "N": n, "n_seeds": 5,
                             "MAE_logM_mean": float(np.abs(p - logM).mean()),
                             "MAE_logM_sd": 0.0,
                             "MAE_logM_ensemble": float(np.abs(p - logM).mean()),
                             "within_tuple_rho_logM_mean": 0.0,
                             "within_tuple_rho_logM_sd": 0.0})
                rho_rows.append({"dataset_version": DS.VERSION, "model": f"metadata_{model}",
                                 "target": "logM", "severity": sev, "rho_mean": 0.0,
                                 "rho_sd": 0.0})

    pd.DataFrame(rows).to_csv(args.outdir / "a5_shuffle_and_a14_rho.csv", index=False)
    pd.DataFrame(rho_rows).to_csv(args.outdir / "a14_within_tuple_rho_by_target.csv", index=False)
    (args.outdir / "run_sidecar_a5_a14.json").write_text(json.dumps({
        "task": "A5+A14", "script": str(Path(__file__).relative_to(ROOT)),
        "dataset_version": DS.VERSION, "shuffle": "within-tuple derangement, no fixed points",
        "shuffle_seed": args.seed, "severities": list(args.severities),
        "date_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"\nwrote {args.outdir}")


if __name__ == "__main__":
    main()
