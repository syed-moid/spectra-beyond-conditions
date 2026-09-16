"""Round 48 item 8: can the three-way map be restored across severity?

Runs the three matched fitting procedures -- trust-region (no penalty), the
W = 0 control (same solver as the regularized fit, no penalty) and the
prior-regularized fit -- at s in {0.25, 0.5, 2, 4} on the same 600 test
spectra used at s = 1, and reports all-case log M error under the replacement
policy.

Bounds travel with the severity. The generator draws the energy zero-offset
uniformly on +-0.5*s meV, so the fit's offset bound is set to +-0.6*s, the same
1.2x margin used at s = 1. Resolution scales as sqrt(s) and stays inside its
existing bound at every severity tested. Both are recorded per severity.

**Stop rule.** If any method exceeds a 25% failure rate at any severity, the map
is not restorable on these procedures and the narrowed claim is used instead.
The script reports the verdict; it does not edit the manuscript.

Evaluation only; no training.
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import skewnorm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import a3_fitting_baselines as A3
from sbc.data.dataset import InsSpectraDataset
from sbc.data.merit import merit
from sbc.data.latent_perturbations import (ALPHA_MU_LOG, ALPHA_SIGMA_LOG, BETA_XI2_HIGH,
    BETA_XI2_LOW, BETA_XI2_SKEW, GAMMA_FLOOR, _BETA_XI2_LOC, _BETA_XI2_SCALE)
from sbc.data.spectrum_generator import Gamma_Q, omega_Q
import dataset_paths as DS

SEVS = (0.25, 0.5, 2.0, 4.0)
FAIL_LIMIT = 0.25


def methods():
    """Bounds are fixed, not severity-dependent (round 49 item 3).

    Rounds 46-48 scaled what they believed was the energy-offset bound with the
    severity. That bound is index 8, the background slope; the energy offset is
    index 5 and is fixed at +-3.0 meV, which covers 100% of the generator's draws
    at every severity tested (max |offset| 2.0 meV at s = 4). No bound moves.
    """
    return {
        "dho_matched":       (A3.fit_matched, "matched-model fit (trust region, no penalty)"),
        "dho_matched_w0":    (lambda y, g, T, c, E: A3.fit_matched_prior(y, g, T, c, E, weight=0.0),
                              "W = 0 control (L-BFGS-B, no penalty)"),
        "dho_matched_prior": (A3.fit_matched_prior, "prior-regularized forward-model fit"),
    }


def conditional_logM(T, c, E, n, rng):
    om0, gm0 = omega_Q(T, c / 100.0, E), Gamma_Q(T, c / 100.0, E)
    ax = rng.lognormal(ALPHA_MU_LOG, ALPHA_SIGMA_LOG, n) * rng.standard_normal(n)
    raw = skewnorm.rvs(a=BETA_XI2_SKEW, size=n, random_state=rng)
    bx2 = np.clip(raw * _BETA_XI2_SCALE + _BETA_XI2_LOC, BETA_XI2_LOW, BETA_XI2_HIGH)
    om, gm = om0 * (1.0 + bx2), gm0 * np.maximum(GAMMA_FLOOR, 1.0 + ax)
    lm = [np.log(max(merit(float(a), float(b), T, E), 1e-9)) for a, b in zip(om, gm)]
    return float(np.median(lm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--severities", type=float, nargs="+", default=list(SEVS))
    ap.add_argument("--oracle-draws", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()
    out = args.outdir or DS.rev("A23"); out.mkdir(parents=True, exist_ok=True)

    d = InsSpectraDataset(DS.TEST, "test", severity=1.0, as_torch=False)
    g = d._omega_grid
    idx = np.linspace(0, len(d) - 1, min(args.n, len(d))).astype(int)
    T = d._T_K[idx].astype(float); c = d._c_pct[idx].astype(float); E = d._E_kVcm[idx].astype(float)
    y_true = np.log(np.clip(d._M[idx].astype(float), 1e-9, None))
    # Round 49 item 7: read the authoritative per-spectrum conditions-only
    # prediction rather than re-estimating it. Three scripts previously produced
    # three different Monte Carlo estimates of the same reference.
    ps_path = out / "primary_per_spectrum.csv"
    if not ps_path.exists():
        raise SystemExit(f"run a32_primary_comparison.py first: {ps_path} is missing")
    ps = pd.read_csv(ps_path).set_index("idx").loc[idx]
    repl_lm = ps.logM_cond.to_numpy()
    ref = json.loads((out / "conditions_only_reference.json").read_text())["MAE_logM"]
    print(f"  {len(idx)} spectra; conditions-only reference MAE {ref:.6f} "
          f"(authoritative, from a32)")

    rows, worst = [], 0.0
    for sev in args.severities:
        for name, (fn, label) in methods().items():
            t0 = time.time(); lm, ok_all = [], []
            for j, i in enumerate(idx):
                yy = d.get_augmented(int(i), sev).astype(float)
                r = fn(yy, g, float(T[j]), float(c[j]), float(E[j]))
                good = bool(r["ok"]) and np.isfinite(r["omega0"]) and np.isfinite(r["Gamma"]) \
                    and r["Gamma"] > 0
                lm.append(np.log(max(merit(r["omega0"], r["Gamma"], float(T[j]), float(E[j])), 1e-9))
                          if good else repl_lm[j])
                ok_all.append(good)
            lm = np.asarray(lm); ok_all = np.asarray(ok_all)
            fr = float(1 - ok_all.mean()); worst = max(worst, fr)
            e = np.abs(lm - y_true)
            rows.append({"severity": sev, "method": name, "label": label, "n": len(idx),
                         "energy_offset_bound_meV": 3.0, "failure_rate": fr,
                         "MAE_logM_all": float(e.mean()),
                         "MAE_logM_ok": float(np.abs(lm - y_true)[ok_all].mean()),
                         "n_ok": int(ok_all.sum())})
            print(f"  s={sev:<5g} {name:20s} fail {fr:6.2%}  MAE(all) {e.mean():.4f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    t = pd.DataFrame(rows); t.to_csv(out / "severity_map_fits.csv", index=False)

    verdict = "restore" if worst <= FAIL_LIMIT else "narrow"
    print(f"\n  worst failure rate across all severities and methods: {worst:.2%}")
    print(f"  VERDICT: {verdict}  (limit {FAIL_LIMIT:.0%})")
    if verdict == "narrow":
        bad = t[t.failure_rate > FAIL_LIMIT][["severity", "method", "failure_rate"]]
        print(bad.to_string(index=False))

    (out / "severity_map_verdict.json").write_text(json.dumps({
        "task": "A33_severity_map", "dataset_version": DS.VERSION,
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "severities": list(args.severities), "n": int(len(idx)),
        "bounds_rule": ("fixed at every severity; energy offset (index 5) +-3.0 meV covers "
                        "100% of the generator's draws at every severity tested"),
        "resolution_note": "resolution scales as sqrt(severity) and stays inside its existing bound",
        "policy": "all cases; failures replaced by the conditions-only conditional median",
        "failure_limit": FAIL_LIMIT, "worst_failure_rate": worst, "verdict": verdict,
        "training": "none", "seed": args.seed,
        "date_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"  wrote {out}/severity_map_fits.csv")


if __name__ == "__main__":
    main()
