"""A2 - metadata-only model family, 5 seeds each.

Replaces the single-seed conditions-only MLP of v7 with:
  * mlp_mse   : the v7 architecture and training protocol (MSE on log M)
  * mlp_l1    : same, L1 loss (so it targets the conditional median)
  * mlp_mse_innerval : same as mlp_mse but model selection on a 10% inner split
                       of train instead of on the evaluation split
  * hgb       : sklearn HistGradientBoostingRegressor on (T, c, E)
  * knn       : k-NN on normalized (T, c, E), k chosen by 5-fold CV on train

Every model sees only (T, c, E) and predicts log M. Evaluation on val,
stress_base, and the three hold-outs. Metrics: MAE_logM, R^2, Spearman rho,
median multiplicative error exp(|dlog M|), bootstrap CI of the MAE, and a
paired-bootstrap CI of the difference against the A1 oracle conditional-median
predictor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from sbc.models.nonlinear_conditions_mlp import (  # noqa: E402
    NonlinearConditionsMLP, normalize_conditions,
)

NPZ = ROOT / "data" / "full_dataset_phase1" / "dataset.npz"
A1_CSV = ROOT / "results" / "revision" / "A1" / "per_tuple_conditional_logM.csv"
OUTDIR = ROOT / "results" / "revision" / "A2"
EVAL_SPLITS = ("val", "stress_base", "holdout_T", "holdout_c", "holdout_E")
SEEDS = (42, 43, 44, 45, 46)


# --------------------------------------------------------------------------- #
# data                                                                        #
# --------------------------------------------------------------------------- #

def load_split(z, name):
    idx = np.nonzero(z["split"] == name)[0]
    X = normalize_conditions(z["T_K"][idx].astype(np.float32),
                             z["c_pct"][idx].astype(np.float32),
                             z["E_kVcm"][idx].astype(np.float32))
    y = np.log(np.clip(z["M"][idx].astype(np.float64), 1e-9, None)).astype(np.float32)
    return X, y


# --------------------------------------------------------------------------- #
# MLP training (v7 protocol, with a loss switch)                              #
# --------------------------------------------------------------------------- #

def train_mlp(Xtr, ytr, Xsel, ysel, seed, loss_name, device,
              lr=1e-3, n_epochs=1500, batch_size=256, weight_decay=1e-4):
    torch.manual_seed(seed)
    if device == "mps":
        torch.mps.manual_seed(seed)
    net = NonlinearConditionsMLP().to(device)
    Xt = torch.from_numpy(Xtr).to(device); yt = torch.from_numpy(ytr).to(device)
    Xs = torch.from_numpy(Xsel).to(device); ys = torch.from_numpy(ysel).to(device)
    lossf = nn.functional.mse_loss if loss_name == "mse" else nn.functional.l1_loss
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    n = Xt.shape[0]
    best = float("inf")
    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    for _ in range(n_epochs):
        net.train()
        perm = rng.permutation(n)
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            loss = lossf(net(Xt[b]), yt[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            sel = float(lossf(net(Xs), ys).cpu())
        if sel < best:
            best = sel
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    return net


@torch.no_grad()
def mlp_predict(net, X, device):
    return net(torch.from_numpy(X).to(device)).cpu().numpy().astype(np.float64)


# --------------------------------------------------------------------------- #
# metrics                                                                     #
# --------------------------------------------------------------------------- #

def metric_row(y, p):
    err = np.abs(y - p)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "MAE_logM": float(err.mean()),
        "R2_logM": 1.0 - ss_res / ss_tot,
        "spearman_rho": float(spearmanr(y, p).statistic),
        "median_multiplicative_error": float(np.exp(np.median(err))),
    }


def boot_ci(vals, n_boot, rng):
    n = len(vals)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = vals[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_boot_ci(a, b, n_boot, rng):
    """95% CI of mean(a) - mean(b) resampling spectra jointly."""
    d = a - b
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# --------------------------------------------------------------------------- #

def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def config_hash():
    h = hashlib.sha256()
    for p in (ROOT / "sbc" / "models" / "nonlinear_conditions_mlp.py",
              Path(__file__)):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260905, help="master seed for bootstrap + CV")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device} seeds={args.seeds} epochs={args.epochs}")

    z = np.load(NPZ, allow_pickle=False)
    Xtr, ytr = load_split(z, "train")
    evals = {s: load_split(z, s) for s in EVAL_SPLITS}

    # Inner split of train for the leakage-free MLP variant.
    inner_rng = np.random.default_rng(args.seed)
    perm = inner_rng.permutation(len(Xtr))
    n_inner = len(Xtr) // 10
    inner_val, inner_tr = perm[:n_inner], perm[n_inner:]

    preds = {}          # (model, seed) -> {split: prediction array}
    timings = {}

    for seed in args.seeds:
        for loss_name in ("mse", "l1"):
            t0 = time.time()
            net = train_mlp(Xtr, ytr, evals["val"][0], evals["val"][1], seed, loss_name, device,
                            n_epochs=args.epochs)
            preds[(f"mlp_{loss_name}", seed)] = {s: mlp_predict(net, evals[s][0], device)
                                                 for s in EVAL_SPLITS}
            timings[f"mlp_{loss_name}_{seed}"] = time.time() - t0
            print(f"  mlp_{loss_name} seed={seed} val MAE="
                  f"{np.abs(preds[(f'mlp_{loss_name}', seed)]['val'] - evals['val'][1]).mean():.4f}"
                  f"  ({timings[f'mlp_{loss_name}_{seed}']:.0f}s)")

        t0 = time.time()
        net = train_mlp(Xtr[inner_tr], ytr[inner_tr], Xtr[inner_val], ytr[inner_val],
                        seed, "mse", device, n_epochs=args.epochs)
        preds[("mlp_mse_innerval", seed)] = {s: mlp_predict(net, evals[s][0], device)
                                             for s in EVAL_SPLITS}
        timings[f"mlp_mse_innerval_{seed}"] = time.time() - t0
        print(f"  mlp_mse_innerval seed={seed} val MAE="
              f"{np.abs(preds[('mlp_mse_innerval', seed)]['val'] - evals['val'][1]).mean():.4f}"
              f"  ({timings[f'mlp_mse_innerval_{seed}']:.0f}s)")

        g = HistGradientBoostingRegressor(loss="absolute_error", random_state=seed,
                                          early_stopping=True, validation_fraction=0.1)
        g.fit(Xtr, ytr)
        preds[("hgb", seed)] = {s: g.predict(evals[s][0]).astype(np.float64) for s in EVAL_SPLITS}
        print(f"  hgb seed={seed} n_iter={g.n_iter_} val MAE="
              f"{np.abs(preds[('hgb', seed)]['val'] - evals['val'][1]).mean():.4f}")

    # k-NN: k chosen by 5-fold CV on train (MAE); deterministic given the fold split.
    ks = [1, 3, 5, 10, 20, 50, 100, 200]
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv = {}
    for k in ks:
        errs = []
        for tr_i, te_i in kf.split(Xtr):
            m = KNeighborsRegressor(n_neighbors=k).fit(Xtr[tr_i], ytr[tr_i])
            errs.append(np.abs(m.predict(Xtr[te_i]) - ytr[te_i]).mean())
        cv[k] = float(np.mean(errs))
    k_best = min(cv, key=cv.get)
    print(f"  knn CV MAE by k: { {k: round(v,4) for k,v in cv.items()} } -> k={k_best}")
    knn = KNeighborsRegressor(n_neighbors=k_best).fit(Xtr, ytr)
    for seed in args.seeds:  # deterministic; recorded per seed for a uniform table
        preds[("knn", seed)] = {s: knn.predict(evals[s][0]).astype(np.float64) for s in EVAL_SPLITS}

    # Oracle predictions from A1 (conditional median), aligned by split order.
    a1 = pd.read_csv(A1_CSV)
    oracle = {s: a1[a1.split == s]["logM_cond_median"].to_numpy() for s in EVAL_SPLITS}
    for s in EVAL_SPLITS:
        assert len(oracle[s]) == len(evals[s][1]), s
        assert np.allclose(a1[a1.split == s]["logM_realized"].to_numpy(), evals[s][1], atol=1e-5), s

    models = ["mlp_mse", "mlp_l1", "mlp_mse_innerval", "hgb", "knn"]
    rng = np.random.default_rng(args.seed)
    rows, per_seed_rows = [], []
    for model in models:
        for s in EVAL_SPLITS:
            y = evals[s][1].astype(np.float64)
            per_seed = []
            for seed in args.seeds:
                p = preds[(model, seed)][s]
                r = metric_row(y, p)
                r.update(model=model, split=s, seed=seed)
                per_seed_rows.append(r)
                per_seed.append(r)
            agg = {"model": model, "split": s, "N": len(y), "n_seeds": len(args.seeds)}
            for k in ("MAE_logM", "R2_logM", "spearman_rho", "median_multiplicative_error"):
                v = np.array([r[k] for r in per_seed])
                agg[f"{k}_mean"] = float(v.mean())
                agg[f"{k}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            # bootstrap over spectra, using the seed-averaged prediction
            pbar = np.mean([preds[(model, seed)][s] for seed in args.seeds], axis=0)
            err = np.abs(y - pbar)
            lo, hi = boot_ci(err, args.n_boot, rng)
            agg["MAE_logM_seedavg"] = float(err.mean())
            agg["MAE_boot_lo"], agg["MAE_boot_hi"] = lo, hi
            d, dlo, dhi = paired_boot_ci(err, np.abs(y - oracle[s]), args.n_boot, rng)
            agg["diff_vs_oracle_mean"] = d
            agg["diff_vs_oracle_lo"], agg["diff_vs_oracle_hi"] = dlo, dhi
            rows.append(agg)

    df = pd.DataFrame(rows)
    df.to_csv(args.outdir / "metadata_family_metrics.csv", index=False)
    pd.DataFrame(per_seed_rows).to_csv(args.outdir / "metadata_family_per_seed.csv", index=False)

    pred_rows = []
    for model in models:
        for s in EVAL_SPLITS:
            pbar = np.mean([preds[(model, seed)][s] for seed in args.seeds], axis=0)
            pred_rows.append(pd.DataFrame({"model": model, "split": s,
                                           "logM_true": evals[s][1].astype(np.float64),
                                           "logM_pred_seedavg": pbar,
                                           "logM_oracle_median": oracle[s]}))
    pd.concat(pred_rows, ignore_index=True).to_csv(args.outdir / "predictions_seedavg.csv", index=False)

    sidecar = {"task": "A2", "script": str(Path(__file__).relative_to(ROOT)),
               "master_seed": args.seed, "model_seeds": list(args.seeds),
               "mlp_epochs": args.epochs, "n_boot": args.n_boot, "device": device,
               "knn_k_selected": int(k_best), "knn_cv_mae": cv,
               "dataset": str(NPZ.relative_to(ROOT)),
               "dataset_master_seed": int(z["master_seed"]),
               "config_hash_sha256_16": config_hash(), "git_commit": git_sha(),
               "date_utc": datetime.now(timezone.utc).isoformat(),
               "timings_sec": timings,
               "numpy": np.__version__, "torch": torch.__version__, "pandas": pd.__version__}
    (args.outdir / "run_sidecar.json").write_text(json.dumps(sidecar, indent=2))
    print(f"\nwrote {args.outdir}")


if __name__ == "__main__":
    main()
