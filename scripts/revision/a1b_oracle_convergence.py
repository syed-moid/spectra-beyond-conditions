"""Monte Carlo convergence of the conditions-only Bayes risk (round 42/43 Part 3.8).

The quantity is a **Monte Carlo estimate of the conditions-only Bayes risk under
absolute loss**: for conditions C = (T, c, E) the optimal predictor under
absolute loss is the conditional median m(C) = median(Y | C), and the risk is
R* = E|Y - m(C)|. Three things must not be conflated:

  1. the theoretical optimal predictor and its risk -- exact by definition;
  2. the finite-draw Monte Carlo approximation of m(C) -- has sampling error;
  3. the risk estimated on a finite evaluation set -- has sampling error too.

Only (1) is exact. This script quantifies (2) by sweeping the number of latent
draws per condition, and (3) by reporting the standard error over evaluation
spectra. Neither is "an exact information bound", and MAE in nats is not mutual
information.

Latent draws are independent of the realizations stored in the evaluation set:
the draws are fresh from the latent model at each condition tuple, never the
stored realization.
"""

from __future__ import annotations

import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skewnorm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from sbc.data.merit import ALPHA_DEFAULT, K_B_meV_per_K, T_C_K  # noqa: E402
from sbc.data.latent_perturbations import (  # noqa: E402
    ALPHA_MU_LOG, ALPHA_SIGMA_LOG, BETA_XI2_HIGH, BETA_XI2_LOW, BETA_XI2_SKEW,
    GAMMA_FLOOR, _BETA_XI2_LOC, _BETA_XI2_SCALE)
from sbc.data.spectrum_generator import Gamma_Q, omega_Q  # noqa: E402
import dataset_paths as DS  # noqa: E402

DRAWS = (250, 500, 1000, 2000, 4000)


def merit_vec(om, gm, T, E):
    return ((om / gm) * np.exp(-ALPHA_DEFAULT * np.abs((T - T_C_K) / T_C_K))
            * np.exp(-om / (2.0 * K_B_meV_per_K * T))
            * (1.0 / (1.0 + (gm / (om / 2.0)) ** 2)) * (1.0 + 0.05 * E / (1.0 + E)))


def conditional_logM(T, c_pct, E, n, rng):
    om0, gm0 = omega_Q(T, c_pct / 100.0, E), Gamma_Q(T, c_pct / 100.0, E)
    ax = rng.lognormal(ALPHA_MU_LOG, ALPHA_SIGMA_LOG, n) * rng.standard_normal(n)
    raw = skewnorm.rvs(a=BETA_XI2_SKEW, size=n, random_state=rng)
    bx2 = np.clip(raw * _BETA_XI2_SCALE + _BETA_XI2_LOC, BETA_XI2_LOW, BETA_XI2_HIGH)
    om = om0 * (1.0 + bx2)
    gm = gm0 * np.maximum(GAMMA_FLOOR, 1.0 + ax)
    return np.log(np.clip(merit_vec(om, gm, T, E), 1e-9, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, default=None, help="default: the untouched test set")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--draws", type=int, nargs="+", default=list(DRAWS))
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    npz = args.npz or DS.TEST
    out = args.outdir or DS.rev("A1")
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(npz, allow_pickle=False)
    m = z["split"] == args.split if args.split else np.ones(len(z["M"]), bool)
    y = np.log(np.clip(z["M"][m].astype(float), 1e-9, None))
    T, c, E = (z["T_K"][m].astype(float), z["c_pct"][m].astype(float),
               z["E_kVcm"][m].astype(float))
    print(f"  {npz.parent.name}/{args.split}: n={len(y)}")

    rows = []
    for nd in args.draws:
        rng = np.random.default_rng(args.seed)
        med = np.array([np.median(conditional_logM(T[i], c[i], E[i], nd, rng))
                        for i in range(len(y))])
        err = np.abs(y - med)
        mae = float(err.mean()); se = float(err.std(ddof=1) / np.sqrt(len(err)))
        rows.append({"dataset_version": DS.VERSION, "split": args.split, "n_eval": int(len(y)),
                     "draws_per_condition": int(nd), "bayes_risk_MAE_logM": mae,
                     "se_over_eval_set": se,
                     "marginal_median_MAE": float(np.abs(y - np.median(y)).mean())})
        print(f"    draws={nd:5d}  MAE={mae:.6f}  SE(eval)={se:.6f}")
    df = pd.DataFrame(rows)
    df["delta_vs_finest"] = df.bayes_risk_MAE_logM - df.bayes_risk_MAE_logM.iloc[-1]
    df.to_csv(out / "oracle_convergence.csv", index=False)

    ref = df[df.draws_per_condition == 2000]
    (out / "run_sidecar_convergence.json").write_text(json.dumps({
        "task": "A1b_oracle_convergence",
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "quantity": "Monte Carlo estimate of the conditions-only Bayes risk under absolute loss",
        "estimator": "conditional median of log M given (T, c, E), latents redrawn per condition",
        "draws_per_condition_reported": 2000,
        "draws_swept": list(args.draws),
        "mc_drift_2000_vs_4000": float(ref.delta_vs_finest.iloc[0]) if len(ref) else None,
        "se_over_eval_set": float(ref.se_over_eval_set.iloc[0]) if len(ref) else None,
        "independence": "latent draws are fresh at each condition tuple and never reuse the "
                        "stored realization of the evaluation spectrum",
        "caveat": "not an exact information bound; MAE in nats is not mutual information",
        "seed": args.seed, "dataset": str(npz.relative_to(ROOT)),
        "date_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"\n  MC drift 2000 -> 4000 draws: {df.delta_vs_finest.iloc[-2]:+.2e}")
    print(f"  SE over the evaluation set at 2000 draws: {ref.se_over_eval_set.iloc[0]:.6f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
