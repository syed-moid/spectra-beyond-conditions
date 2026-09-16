"""Headline evaluation on the untouched test sets (round 42/43 Part 3.2, 3.7, 3.9).

Every number the manuscript reports as a headline comes from here. The models
were selected on each realization's own `val` split; `test_v10` and
`replicate_test_v10` were generated with fresh seeds after selection was frozen
and are read for the first time in this script.

Emits, per model and per cell (realization x training seed):
  * MAE on log M, and per-spectrum absolute errors for the paired comparisons;
  * within-condition Spearman rho AND pairwise ranking accuracy with half credit
    for ties, whose chance baseline is 0.5 and is therefore defined for the
    conditions-only models, unlike rho;
  * the derangement shuffle control with its expected offset -rho/(n-1), and an
    ordinary-permutation null for a zero-centred reference;
  * parameter recovery R^2 and within-condition pairwise accuracy for omega0 and
    Gamma, with the conditions-only comparator alongside.
"""

from __future__ import annotations

import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
import a4_train_v8 as A4  # noqa: E402
from a17_arch_sweep import build_variant  # noqa: E402
from sbc.data.dataset import InsSpectraDataset  # noqa: E402
from sbc.models.spectral_transformer import TargetStats, unstandardize_outputs  # noqa: E402
from realization_stats import pairwise_ranking_accuracy, expected_deranged_rho  # noqa: E402
import dataset_paths as DS  # noqa: E402

MODELS = [("5a", "base"), ("5b", "base"), ("fusion", "base"), ("cnn", "base"),
          ("cnn_kernel45", "variant"), ("tf_patch60", "variant")]
DATASET_SEEDS = (0, 1, 2)
TRAIN_SEEDS = (42, 43, 44)
SEVERITIES = (0.5, 1.0, 2.0, 4.0)


def load(model, kind, ds_seed, seed, ckpt_root, device="cpu"):
    path = ckpt_root / f"ds{ds_seed}" / f"{model}_v7_seed{seed}.pt"
    ck = torch.load(path, map_location=device, weights_only=False)
    net = build_variant(model, seed) if kind == "variant" else A4.build_model(ck["arch"], seed)
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net, TargetStats.from_dict(ck["target_stats"])


@torch.no_grad()
def predict(net, stats, X, cond=None, chunk=2048):
    outs = []
    for i in range(0, len(X), chunk):
        xb = torch.from_numpy(X[i:i + chunk].astype(np.float32))
        outs.append(net(xb).numpy() if cond is None else
                    net(xb, torch.from_numpy(cond[i:i + chunk].astype(np.float32))).numpy())
    return unstandardize_outputs(np.concatenate(outs, 0), stats)


def derangement(rng, n):
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--ckpt-root", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--severities", type=float, nargs="+", default=list(SEVERITIES))
    args = ap.parse_args()
    out = args.outdir or DS.rev("A22")
    out.mkdir(parents=True, exist_ok=True)
    ck_root = args.ckpt_root or (DS.rev("A21") / "checkpoints")
    rng = np.random.default_rng(args.seed)

    # ---- untouched sets -----------------------------------------------------
    te = InsSpectraDataset(DS.TEST, "test", severity=1.0, as_torch=False)
    rp = InsSpectraDataset(DS.REPLICATE_TEST, "replicate_test", severity=1.0, as_torch=False)
    zt = np.load(DS.REPLICATE_TEST, allow_pickle=False)
    tid = zt["tuple_id"][zt["split"] == "replicate_test"]
    from sbc.models.nonlinear_conditions_mlp import normalize_conditions
    cond_te = normalize_conditions(te._T_K, te._c_pct, te._E_kVcm)
    cond_rp = normalize_conditions(rp._T_K, rp._c_pct, rp._E_kVcm)
    y_te = np.log(np.clip(te._M.astype(float), 1e-9, None))
    y_rp = np.log(np.clip(rp._M.astype(float), 1e-9, None))
    om_te, gm_te = te._omega_Q.astype(float), np.log(np.clip(te._Gamma_Q.astype(float), 1e-9, None))

    X_te = {s: np.stack([te.get_augmented(i, s) for i in range(len(te))]).astype(np.float32)
            for s in args.severities}
    X_rp1 = np.stack([rp.get_augmented(i, 1.0) for i in range(len(rp))]).astype(np.float32)
    print(f"  test n={len(y_te)}  replicate_test n={len(y_rp)} ({len(np.unique(tid))} tuples)")

    # derangement + ordinary permutation, fixed, shared across models
    der = np.arange(len(y_rp)); perm = np.arange(len(y_rp))
    for t in np.unique(tid):
        p = np.nonzero(tid == t)[0]
        der[p] = p[derangement(rng, len(p))]
        perm[p] = p[rng.permutation(len(p))]

    rows, err_store = [], {}
    for model, kind in MODELS:
        for ds in DATASET_SEEDS:
            for sd in TRAIN_SEEDS:
                try:
                    net, st = load(model, kind, ds, sd, ck_root)
                except FileNotFoundError:
                    print(f"  MISSING checkpoint {model} ds{ds} seed{sd}"); continue
                c_te = cond_te if model == "fusion" else None
                c_rp = cond_rp if model == "fusion" else None
                for s in args.severities:
                    P = predict(net, st, X_te[s], c_te)
                    p = np.log(np.clip(P["M"], 1e-9, None))
                    e = np.abs(p - y_te)
                    rows.append({"dataset_version": DS.VERSION, "model": model,
                                 "dataset_seed": ds, "seed": sd, "eval_set": "test",
                                 "severity": s, "n": len(y_te), "MAE_logM": float(e.mean()),
                                 "in_training_range": bool(s <= 4.0)})
                    if s == 1.0:
                        err_store[(model, ds, sd)] = e
                        mae_s1 = float(e.mean())
                        rows[-1]["R2_omega0"] = float(1 - ((P["omega_Q"] - om_te) ** 2).sum()
                                                      / ((om_te - om_te.mean()) ** 2).sum())
                        lg = np.log(np.clip(P["Gamma_Q"], 1e-9, None))
                        rows[-1]["R2_logGamma"] = float(1 - ((lg - gm_te) ** 2).sum()
                                                        / ((gm_te - gm_te.mean()) ** 2).sum())
                # replicate_test: ranking metrics at s = 1
                P = predict(net, st, X_rp1, c_rp)
                pr = np.log(np.clip(P["M"], 1e-9, None))
                rho = float(np.mean([spearmanr(y_rp[tid == t], pr[tid == t]).statistic
                                     for t in np.unique(tid)]))
                acc = pairwise_ranking_accuracy(tid, y_rp, pr)
                rho_d = float(np.mean([spearmanr(y_rp[tid == t], pr[der][tid == t]).statistic
                                       for t in np.unique(tid)]))
                rho_p = float(np.mean([spearmanr(y_rp[tid == t], pr[perm][tid == t]).statistic
                                       for t in np.unique(tid)]))
                n_draw = int(len(y_rp) / len(np.unique(tid)))
                rows.append({"dataset_version": DS.VERSION, "model": model, "dataset_seed": ds,
                             "seed": sd, "eval_set": "replicate_test", "severity": 1.0,
                             "n": len(y_rp), "MAE_logM": float(np.abs(pr - y_rp).mean()),
                             "rho_logM": rho, "pairwise_accuracy": acc,
                             "rho_deranged": rho_d,
                             "rho_deranged_expected": expected_deranged_rho(rho, n_draw),
                             "rho_permuted": rho_p,
                             "pairwise_accuracy_deranged": pairwise_ranking_accuracy(tid, y_rp, pr[der]),
                             "in_training_range": True})
                print(f"  {model:14s} ds{ds} s{sd}: test MAE(s=1) {mae_s1:.4f}  "
                      f"rho {rho:.4f}  pair-acc {acc:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "test_eval_runs.csv", index=False)
    np.savez_compressed(out / "per_spectrum_errors_s1.npz",
                        **{f"{m}|{d}|{s}": v for (m, d, s), v in err_store.items()})
    (out / "run_sidecar.json").write_text(json.dumps({
        "task": "A22_test_eval", "dataset_version": DS.VERSION,
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "test_set": str(DS.TEST.relative_to(ROOT)),
        "replicate_test_set": str(DS.REPLICATE_TEST.relative_to(ROOT)),
        "selection_note": "models selected on each realization's val split; these sets were "
                          "generated with fresh seeds after selection was frozen",
        "severities": list(args.severities),
        "out_of_training_range": [s for s in args.severities if s > 4.0],
        "seed": args.seed,
        "date_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
