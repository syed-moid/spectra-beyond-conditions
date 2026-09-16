"""Fitting baselines on the untouched test set, including the MAP comparator.

Specification (round 42 Part 3.5), per method:
  dho_matched        full generator forward model (3 DHOs + central peak + linear
                     background + offset + resolution), omega0 and Gamma fitted,
                     mode positions of the fixed modes free within bounds,
                     least_squares TRF, max_nfev 4000, residuals scaled by max(y).
                     No prior. Has access to (T, c, E) through the baseline
                     initialisation only.
  dho_matched_prior  identical, plus the generator's own latent prior as a
                     log-prior penalty (MAP). This is the per-spectrum Bayesian
                     comparator for the amortization caveat.
  dho_single_win     single-mode DHO on a window around the soft mode.
  dho_two_mode       two DHOs, no prior; bounds audited before any R^2 is quoted.
Failure handling: a fit is failed if the optimizer does not converge or if
omega0 or Gamma lands within 0.1% of a bound. Failures are reported, and
metrics are given both excluding them and replacing them with the
conditions-only prediction.
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import a3_fitting_baselines as A3  # noqa: E402
from sbc.data.dataset import InsSpectraDataset  # noqa: E402
from sbc.data.latent_perturbations import (ALPHA_MU_LOG, ALPHA_SIGMA_LOG, BETA_XI2_HIGH,  # noqa: E402
    BETA_XI2_LOW, BETA_XI2_SKEW, GAMMA_FLOOR, _BETA_XI2_LOC, _BETA_XI2_SCALE)
from sbc.data.spectrum_generator import Gamma_Q, omega_Q  # noqa: E402
from scipy.stats import skewnorm  # noqa: E402
import dataset_paths as DS  # noqa: E402


def conditional_params(T, c, E, n, rng):
    """Conditions-only Monte Carlo conditional median of (omega0, Gamma).

    Used as the replacement value under the second failure policy, so a failed
    fit is scored at what the measurement conditions alone imply rather than
    being dropped.
    """
    om0, gm0 = omega_Q(T, c / 100.0, E), Gamma_Q(T, c / 100.0, E)
    ax = rng.lognormal(ALPHA_MU_LOG, ALPHA_SIGMA_LOG, n) * rng.standard_normal(n)
    raw = skewnorm.rvs(a=BETA_XI2_SKEW, size=n, random_state=rng)
    bx2 = np.clip(raw * _BETA_XI2_SCALE + _BETA_XI2_LOC, BETA_XI2_LOW, BETA_XI2_HIGH)
    return (float(np.median(om0 * (1.0 + bx2))),
            float(np.median(gm0 * np.maximum(GAMMA_FLOOR, 1.0 + ax))))

METHODS = {"dho_matched": A3.fit_matched, "dho_matched_prior": A3.fit_matched_prior,
           "dho_single_win": A3.fit_single}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()
    out = args.outdir or DS.rev("A23"); out.mkdir(parents=True, exist_ok=True)
    d = InsSpectraDataset(DS.TEST, "test", severity=args.severity, as_torch=False)
    g = d._omega_grid
    n = min(args.n, len(d))
    idx = np.linspace(0, len(d) - 1, n).astype(int)
    rows = []; t0 = time.time()
    for name, fn in METHODS.items():
        for i in idx:
            y = d.get_augmented(int(i), args.severity).astype(float)
            T, c, E = float(d._T_K[i]), float(d._c_pct[i]), float(d._E_kVcm[i])
            r = fn(y, g, T, c, E)
            rows.append({"dataset_version": DS.VERSION, "method": name, "idx": int(i),
                         "severity": args.severity,
                         "omega0_true": float(d._omega_Q[i]), "Gamma_true": float(d._Gamma_Q[i]),
                         "omega0_fit": r["omega0"], "Gamma_fit": r["Gamma"],
                         "ok": bool(r["ok"]), "reason": r["reason"]})
        print(f"  {name}: {n} spectra  {time.time()-t0:.0f}s cumulative", flush=True)
    df = pd.DataFrame(rows); df.to_csv(out / "fit_results_test.csv", index=False)
    # Conditions-only replacement values, for the second failure policy.
    rng = np.random.default_rng(20260914)
    repl = {int(i): conditional_params(float(d._T_K[i]), float(d._c_pct[i]),
                                       float(d._E_kVcm[i]), 2000, rng) for i in idx}

    s = []
    for name, gg in df.groupby("method"):
        ok = gg[gg.ok]
        over = ok.Gamma_true > ok.omega0_true
        # Policy B: failures replaced by the conditions-only prediction, so every
        # method is scored on all n cases rather than on the subset it managed.
        om_b = np.where(gg.ok, gg.omega0_fit, [repl[int(i)][0] for i in gg.idx])
        gm_b = np.where(gg.ok, gg.Gamma_fit, [repl[int(i)][1] for i in gg.idx])
        over_all = (gg.Gamma_true > gg.omega0_true).to_numpy()
        e_gm_b = np.abs(gm_b - gg.Gamma_true.to_numpy())
        s.append({"method": name, "n": len(gg), "failure_rate": float(1 - gg.ok.mean()),
                  "failure_definition": ("optimizer status <= 0, or omega0 or Gamma within a "
                                         "relative 1e-3 of either of its bounds"),
                  "headline_policy": "failures_excluded",
                  # headline: failures excluded
                  "MAE_omega0": float(np.abs(ok.omega0_fit - ok.omega0_true).mean()),
                  "MAE_Gamma": float(np.abs(ok.Gamma_fit - ok.Gamma_true).mean()),
                  "median_abs_err_Gamma": float(np.median(np.abs(ok.Gamma_fit - ok.Gamma_true))),
                  "MAE_Gamma_overdamped": float(np.abs(ok.Gamma_fit - ok.Gamma_true)[over].mean()),
                  "n_overdamped": int(over.sum()),
                  # policy B: failures replaced by the conditions-only prediction
                  "MAE_omega0_failures_replaced": float(np.abs(om_b - gg.omega0_true.to_numpy()).mean()),
                  "MAE_Gamma_failures_replaced": float(e_gm_b.mean()),
                  "median_abs_err_Gamma_failures_replaced": float(np.median(e_gm_b)),
                  "MAE_Gamma_overdamped_failures_replaced": float(e_gm_b[over_all].mean()),
                  "n_overdamped_all": int(over_all.sum())})
    sm = pd.DataFrame(s); sm.to_csv(out / "fit_summary_test.csv", index=False)
    print("\n" + sm.round(4).to_string(index=False))
    (out / "run_sidecar.json").write_text(json.dumps({
        "task": "A23_fitting_test", "dataset_version": DS.VERSION,
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "eval_set": str(DS.TEST.relative_to(ROOT)), "n_spectra": int(n),
        "severity": args.severity, "methods": list(METHODS),
        "regime_definition": "overdamped = Gamma > omega0 (poles at -i*Gamma +- sqrt(omega0^2-Gamma^2))",
        "prior_weight": A3._PRIOR_WEIGHT,
        "prior_residual": ("sqrt(2*W*(nlp - nlp_min)) on the continuous interior; nlp_min is the "
                           "global minimum over the support so the MAP is unchanged. Replaces the "
                           "earlier max(nlp, 0) clip, which zeroed the penalty and its gradient "
                           "over most of the support (round 46)"),
        "nlp_min_interior": A3._nlp_min_interior(),
        "failure_definition": ("optimizer status <= 0, or omega0 or Gamma within a relative 1e-3 "
                               "of either of its bounds"),
        "failure_policies": ["failures_excluded (headline)",
                             "failures replaced by the conditions-only conditional median"],
        "forward_model_scope": ("three DHOs, central peak, linear background, energy offset and a "
                                "pseudo-Voigt resolution with eta fixed at 0.3. The generator's "
                                "lineshape skew is NOT in the fit model, and the generator draws "
                                "eta per spectrum on [0.1, 0.5]. Temperature enters the fit through "
                                "the Bose factor and the central-peak envelope; c and E enter only "
                                "through the initialisation."),
        "date_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
