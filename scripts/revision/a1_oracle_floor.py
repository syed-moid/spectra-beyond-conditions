"""A1 - irreducible (oracle) metadata-only error for log M.

For each evaluation-set condition tuple (T, c, E) we redraw N latent
realizations from the Phase 1 latent model and recompute log M for each.
This gives the exact conditional distribution P(log M | T, c, E) that the
conditions-only MLP can only approximate.

Two reference predictors are scored against the realized log M stored in the
dataset:

  * conditional MEDIAN of log M  -- the MAE-optimal predictor;
  * conditional MEAN of log M    -- what an MSE-trained model converges to.

We also report the Monte-Carlo expectation of the same MAEs (averaging the
absolute deviation over all N draws at every tuple, not just the one stored
realization); that is the lower-variance estimate of the irreducible error.

Only the soft mode enters M, so only alpha, xi1[soft] and beta*xi2 matter.
Draws use the same distributions as src/data/latent_perturbations.py; the
per-spectrum draw ORDER differs (vectorised here), which is statistically
irrelevant for a Monte-Carlo expectation.
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
from scipy.stats import skewnorm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sbc.data.merit import merit, K_B_meV_per_K, T_C_K, ALPHA_DEFAULT  # noqa: E402
from sbc.data.latent_perturbations import (  # noqa: E402
    ALPHA_MU_LOG, ALPHA_SIGMA_LOG, BETA_XI2_HIGH, BETA_XI2_LOW, BETA_XI2_SKEW,
    GAMMA_FLOOR, _BETA_XI2_LOC, _BETA_XI2_SCALE,
)
from sbc.data.spectrum_generator import Gamma_Q, omega_Q  # noqa: E402
import dataset_paths as DS  # noqa: E402

NPZ = DS.FULL
OUTDIR = DS.rev("A1")
SPLITS = ("val", "stress_base", "holdout_T", "holdout_c", "holdout_E")


def merit_vec(om, gm, T, E):
    """Vectorised src.data.merit.merit (validated against it in main())."""
    Q = om / gm
    F_prox = np.exp(-ALPHA_DEFAULT * np.abs((T - T_C_K) / T_C_K))
    F_therm = np.exp(-om / (2.0 * K_B_meV_per_K * T))
    F_coh = 1.0 / (1.0 + (gm / (om / 2.0)) ** 2)
    F_field = 1.0 + 0.05 * E / (1.0 + E)
    return Q * F_prox * F_therm * F_coh * F_field


def conditional_logM(T, c_pct, E, n_draws, rng):
    """(n_draws,) log M values at one nominal tuple."""
    om0 = omega_Q(T, c_pct / 100.0, E)
    gm0 = Gamma_Q(T, c_pct / 100.0, E)
    alpha = rng.lognormal(mean=ALPHA_MU_LOG, sigma=ALPHA_SIGMA_LOG, size=n_draws)
    xi1 = rng.standard_normal(n_draws)
    raw = skewnorm.rvs(a=BETA_XI2_SKEW, size=n_draws, random_state=rng)
    bx2 = np.clip(raw * _BETA_XI2_SCALE + _BETA_XI2_LOC, BETA_XI2_LOW, BETA_XI2_HIGH)
    om = om0 * (1.0 + bx2)
    gm = gm0 * np.maximum(GAMMA_FLOOR, 1.0 + alpha * xi1)
    return np.log(np.clip(merit_vec(om, gm, T, E), 1e-9, None)), om0, gm0


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def config_hash():
    h = hashlib.sha256()
    for p in (ROOT / "configs" / "phase1_parameter_card.yaml",
              ROOT / "configs" / "augmentation_realistic.yaml",
              ROOT / "src" / "data" / "latent_perturbations.py",
              ROOT / "src" / "data" / "merit.py",
              ROOT / "src" / "data" / "spectrum_generator.py",
              Path(__file__)):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--n-draws", type=int, default=2000)
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    ap.add_argument("--npz", type=Path, default=NPZ)
    ap.add_argument("--dataset-version", type=str, default=DS.VERSION)
    ap.add_argument("--splits", type=str, nargs="+", default=list(SPLITS))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Validate the vectorised merit against the reference scalar implementation.
    chk = [(8.2, 1.05, 300.0, 0.0), (3.5, 1.80, 400.0, 2.0),
           (11.2, 0.60, 100.0, 0.5), (6.5, 26.0, 600.0, 4.0)]
    for om, gm, T, E in chk:
        a = merit(om, gm, T, E)
        b = float(merit_vec(np.array([om]), np.array([gm]), T, E)[0])
        assert abs(a - b) <= 1e-12 * max(1.0, abs(a)), (om, gm, T, E, a, b)

    z = np.load(args.npz, allow_pickle=False)
    split = z["split"]
    rng = np.random.default_rng(args.seed)

    summary = {}
    rows = []
    for sp in args.splits:
        idx = np.nonzero(split == sp)[0]
        T = z["T_K"][idx].astype(float)
        c = z["c_pct"][idx].astype(float)
        E = z["E_kVcm"][idx].astype(float)
        logM_real = np.log(np.clip(z["M"][idx].astype(float), 1e-9, None))
        n = len(idx)
        med = np.empty(n); mean = np.empty(n); sd = np.empty(n)
        p5 = np.empty(n); p25 = np.empty(n); p75 = np.empty(n); p95 = np.empty(n)
        e_med = np.empty(n); e_mean = np.empty(n)
        om0 = np.empty(n); gm0 = np.empty(n)
        for i in range(n):
            lm, o, g = conditional_logM(T[i], c[i], E[i], args.n_draws, rng)
            med[i] = np.median(lm); mean[i] = lm.mean(); sd[i] = lm.std()
            p5[i], p25[i], p75[i], p95[i] = np.percentile(lm, [5, 25, 75, 95])
            e_med[i] = np.mean(np.abs(lm - med[i]))
            e_mean[i] = np.mean(np.abs(lm - mean[i]))
            om0[i], gm0[i] = o, g
        mae_med = float(np.mean(np.abs(logM_real - med)))
        mae_mean = float(np.mean(np.abs(logM_real - mean)))
        summary[sp] = {
            "N": int(n),
            "MAE_logM_conditional_median": mae_med,
            "MAE_logM_conditional_mean": mae_mean,
            "MAE_logM_conditional_median_MC_expectation": float(np.mean(e_med)),
            "MAE_logM_conditional_mean_MC_expectation": float(np.mean(e_mean)),
            "MAE_logM_marginal_median": float(np.mean(np.abs(logM_real - np.median(logM_real)))),
            "mean_conditional_sd_logM": float(np.mean(sd)),
            "mean_conditional_IQR_logM": float(np.mean(p75 - p25)),
            "mean_conditional_5_95_width_logM": float(np.mean(p95 - p5)),
            "std_logM_realized": float(np.std(logM_real)),
        }
        rows.append(pd.DataFrame({
            "dataset_version": args.dataset_version,
            "split": sp, "T_K": T, "c_pct": c, "E_kVcm": E,
            "omega0_baseline_meV": om0, "Gamma0_baseline_meV": gm0,
            "logM_realized": logM_real,
            "logM_cond_median": med, "logM_cond_mean": mean, "logM_cond_sd": sd,
            "logM_p5": p5, "logM_p25": p25, "logM_p75": p75, "logM_p95": p95,
            "abs_err_median_pred": np.abs(logM_real - med),
            "abs_err_mean_pred": np.abs(logM_real - mean),
            "MC_expected_abs_dev_from_median": e_med,
        }))
        print(f"{sp:12s} N={n:5d}  MAE(median)={mae_med:.4f}  MAE(mean)={mae_mean:.4f}  "
              f"MC-expected MAE(median)={np.mean(e_med):.4f}  mean IQR={np.mean(p75-p25):.4f}")

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(args.outdir / "per_tuple_conditional_logM.csv", index=False)

    # Conditional spread vs each condition axis, on val + stress_base pooled.
    sub = df[df["split"].isin(["val", "stress_base"])]
    if sub.empty:
        sub = df
    spread_rows = []
    for axis, nbins in (("T_K", 10), ("c_pct", 10), ("E_kVcm", 10)):
        edges = np.quantile(sub[axis], np.linspace(0, 1, nbins + 1))
        edges[-1] += 1e-9
        b = np.digitize(sub[axis], edges[1:-1])
        for k in range(nbins):
            m = b == k
            if not m.any():
                continue
            spread_rows.append({
                "dataset_version": args.dataset_version,
                "axis": axis, "bin": k,
                "lo": float(edges[k]), "hi": float(edges[k + 1]),
                "center": float(np.median(sub[axis][m])), "n": int(m.sum()),
                "median_cond_sd": float(np.median(sub["logM_cond_sd"][m])),
                "median_cond_IQR": float(np.median((sub["logM_p75"] - sub["logM_p25"])[m])),
                "median_cond_5_95": float(np.median((sub["logM_p95"] - sub["logM_p5"])[m])),
                "median_cond_median_logM": float(np.median(sub["logM_cond_median"][m])),
            })
    pd.DataFrame(spread_rows).to_csv(args.outdir / "conditional_spread_vs_conditions.csv", index=False)

    v7 = {"conditions_only_MLP_val_v7": 0.6232216787498219,
          "conditions_only_MLP_stress_base_v7": 0.6509906503359798}
    out = {"task": "A1", "dataset_version": args.dataset_version,
           "summary": summary, "v7_reference": v7}
    if "val" in summary:
        out["slack_val_MLP_minus_oracle_median"] = (
            v7["conditions_only_MLP_val_v7"] - summary["val"]["MAE_logM_conditional_median"])
    if "stress_base" in summary:
        out["slack_stress_MLP_minus_oracle_median"] = (
            v7["conditions_only_MLP_stress_base_v7"]
            - summary["stress_base"]["MAE_logM_conditional_median"])
    (args.outdir / "oracle_floor.json").write_text(json.dumps(out, indent=2))

    sidecar = {"task": "A1", "script": str(Path(__file__).relative_to(ROOT)),
               "seed": args.seed, "n_draws": args.n_draws,
               "dataset_version": args.dataset_version,
               "dataset": str(Path(args.npz).resolve().relative_to(ROOT)),
               "dataset_master_seed": int(z["master_seed"]),
               "dataset_generator_git_sha": str(z["generator_git_sha"]),
               "config_hash_sha256_16": config_hash(),
               "git_commit": git_sha(),
               "date_utc": datetime.now(timezone.utc).isoformat(),
               "numpy": np.__version__, "pandas": pd.__version__}
    (args.outdir / "run_sidecar.json").write_text(json.dumps(sidecar, indent=2))
    print(f"\nwrote {args.outdir}")


if __name__ == "__main__":
    main()
