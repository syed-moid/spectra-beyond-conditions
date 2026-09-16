"""Round 46 Priority 1: what the prior residual does when nlp < 0.

`fit_matched_prior` formed its prior residual as

    nlp = max(-log p(m) - log p(r), 0.0)
    r_prior = sqrt(2 * nlp) * W

The `max(..., 0.0)` is a literal clip, not a constant offset. It exists because
sqrt() of a negative number is not real, and nlp IS negative over a large part
of the support: both factors are continuous densities that exceed 1 near their
modes, so their negative log is negative there.

Wherever the clip binds, the prior residual is exactly zero and its gradient is
exactly zero, so the penalty exerts no force at all across that region -- which
includes almost the whole support of the frequency shift. This script measures
how often a fit's optimiser trajectory entered that region, and reports the
point masses that the mixed mass/density treatment also mishandles.

Evaluation only.
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import a3_fitting_baselines as A3
from sbc.data.dataset import InsSpectraDataset
from sbc.data.latent_perturbations import (GAMMA_FLOOR, BETA_XI2_LOW, BETA_XI2_HIGH)
from sbc.data.spectrum_generator import Gamma_Q, omega_Q
import dataset_paths as DS


def nlp_raw(m, fr):
    """The penalty BEFORE the clip."""
    return -A3._log_prior_multiplier(m) - A3._log_prior_shift(fr)


def nlp_min_over_support(n=4001):
    """Global minimum of nlp over the support: parameter-independent."""
    ms = np.linspace(GAMMA_FLOOR, 6.0, n)
    rs = np.linspace(BETA_XI2_LOW, BETA_XI2_HIGH, n)
    lpm = max(A3._log_prior_multiplier(float(m)) for m in ms)
    lpr = max(A3._log_prior_shift(float(r)) for r in rs)
    return -(lpm + lpr), lpm, lpr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--atoms-only", action="store_true",
                    help="skip the fit loop; recompute the boundary-atom fractions only")
    args = ap.parse_args()
    out = args.outdir or DS.rev("A23"); out.mkdir(parents=True, exist_ok=True)

    nmin, lpm_max, lpr_max = nlp_min_over_support()
    print(f"  max log p(m) = {lpm_max:+.4f}   max log p(r) = {lpr_max:+.4f}")
    print(f"  nlp_min over the support = {nmin:+.4f}  (nlp < 0 is reachable, so the clip binds)")

    d = InsSpectraDataset(DS.TEST, "test", severity=args.severity, as_torch=False)
    g = d._omega_grid
    idx = np.linspace(0, len(d) - 1, min(args.n, len(d))).astype(int)   # identical to A23

    if args.atoms_only:
        df = pd.read_csv(out / "prior_clip_diagnostic.csv")
        rows = None
    else:
        rows = []
    t0 = time.time()
    for i in ([] if args.atoms_only else idx):
        y = d.get_augmented(int(i), args.severity).astype(float)
        T, c, E = float(d._T_K[i]), float(d._c_pct[i]), float(d._E_kVcm[i])
        om_b, gm_b = omega_Q(T, c / 100.0, E), Gamma_Q(T, c / 100.0, E)
        p0 = np.array([om_b, gm_b, A3._MODES["acoustic"]["Gamma"], A3._MODES["optical"]["Gamma"],
                       1.5, 0.0, 0.4, 0.02, 0.0, 1.0])
        lo = np.array([0.5, 0.02, 0.05, 0.05, 0.05, -3.0, 1e-4, 0.0, -0.2, 0.2])
        hi = np.array([15.0, 40.0, 10.0, 20.0, 12.0, 3.0, 4.0, 3.0, 0.2, 5.0])
        p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)
        scale = max(float(np.max(y)), 1e-12)
        stat = {"evals": 0, "clipped": 0, "min_nlp": np.inf}

        def resid(p):
            m = p[1] / max(gm_b, 1e-12)
            fr = p[0] / max(om_b, 1e-12) - 1.0
            raw = nlp_raw(m, fr)
            stat["evals"] += 1
            stat["min_nlp"] = min(stat["min_nlp"], raw)
            if raw < 0.0:
                stat["clipped"] += 1
            nlp = max(raw, 0.0)                                  # the shipped behaviour
            base = (A3.model_matched(p, g, T, E) - y) / scale
            return np.concatenate([base, [np.sqrt(2.0 * nlp) * A3._PRIOR_WEIGHT]])

        try:
            r = least_squares(resid, p0, bounds=(lo, hi), max_nfev=4000, method="trf")
            ok, status = r.status > 0, int(r.status)
            pf = r.x
        except Exception:
            ok, status, pf = False, -99, np.full(10, np.nan)
        m_f = pf[1] / max(gm_b, 1e-12); fr_f = pf[0] / max(om_b, 1e-12) - 1.0
        rows.append({"idx": int(i), "evals": stat["evals"], "clipped_evals": stat["clipped"],
                     "frac_evals_clipped": stat["clipped"] / max(stat["evals"], 1),
                     "entered_negative": stat["clipped"] > 0,
                     "min_nlp_on_trajectory": float(stat["min_nlp"]),
                     "final_nlp_raw": float(nlp_raw(m_f, fr_f)) if np.isfinite(m_f) else np.nan,
                     "final_in_clip": bool(np.isfinite(m_f) and nlp_raw(m_f, fr_f) < 0),
                     "ok": ok, "status": status})
    if rows is not None:
        df = pd.DataFrame(rows)
        df.to_csv(out / "prior_clip_diagnostic.csv", index=False)

    print(f"\n  {len(df)} fits in {time.time()-t0:.0f}s")
    print(f"  fits whose trajectory entered nlp < 0 : {df.entered_negative.mean():.1%} "
          f"({int(df.entered_negative.sum())}/{len(df)})")
    print(f"  mean fraction of evaluations clipped  : {df.frac_evals_clipped.mean():.1%}")
    print(f"  fits whose FINAL point is in the clip : {df.final_in_clip.mean():.1%}")
    print(f"  median min nlp on trajectory          : {df.min_nlp_on_trajectory.median():+.3f}")

    # Point masses in the realized parameters, read from the stored latents.
    # Do NOT back-compute these from realized/baseline ratios: omega0_base carries
    # a max(0.5, .) floor and Gamma0 its own terms, so the ratio is not beta*xi2
    # or the multiplier wherever those bind, and the atom fractions come out wrong.
    import json as _json
    z = np.load(DS.FULL, allow_pickle=False)
    lat = [_json.loads(str(x)) for x in z["latent_json"]]
    al = np.array([d["alpha"] for d in lat])
    b2 = np.array([d["beta_xi2"] for d in lat])
    xi = {k: np.array([d["xi1"][k] for d in lat]) for k in ("soft", "acoustic", "optical")}
    tol = 1e-12
    mult_soft = 1.0 + al * xi["soft"]
    mult_all = np.concatenate([1.0 + al * xi[k] for k in xi])
    atoms = {
        "n_spectra": int(len(lat)),
        "frac_multiplier_on_floor_soft": float(np.mean(mult_soft <= GAMMA_FLOOR + tol)),
        "frac_multiplier_on_floor_all_modes": float(np.mean(mult_all <= GAMMA_FLOOR + tol)),
        "frac_shift_on_lower_clip": float(np.mean(b2 <= BETA_XI2_LOW + tol)),
        "frac_shift_on_upper_clip": float(np.mean(b2 >= BETA_XI2_HIGH - tol)),
        "beta_xi2_min": float(b2.min()), "beta_xi2_max": float(b2.max()),
        "note": ("the lower clip at %.3f is never reached: the realized minimum is %.6f"
                 % (BETA_XI2_LOW, b2.min())),
    }
    atoms["frac_soft_pair_on_any_atom"] = float(np.mean(
        (mult_soft <= GAMMA_FLOOR + tol) | (b2 >= BETA_XI2_HIGH - tol) | (b2 <= BETA_XI2_LOW + tol)))
    print("\n  realized parameters on a boundary atom (from the stored latents):")
    for k, v in atoms.items():
        if isinstance(v, float):
            print(f"    {k:36s} {v:.4%}" if k.startswith("frac") else f"    {k:36s} {v:+.6f}")

    payload = {"task": "A29_prior_clip_diagnostic", "dataset_version": DS.VERSION,
               "script": str(Path(__file__).resolve().relative_to(ROOT)),
               "shipped_behaviour": "nlp = max(raw_nlp, 0.0) -- a literal clip, not an offset",
               "nlp_min_over_support": nmin,
               "max_log_p_multiplier": lpm_max, "max_log_p_shift": lpr_max,
               "n_fits": int(len(df)),
               "frac_fits_entered_negative": float(df.entered_negative.mean()),
               "mean_frac_evals_clipped": float(df.frac_evals_clipped.mean()),
               "frac_fits_final_in_clip": float(df.final_in_clip.mean()),
               "atoms": atoms,
               "date_utc": datetime.now(timezone.utc).isoformat()}
    (out / "prior_clip_diagnostic.json").write_text(json.dumps(payload, indent=2))
    print(f"\n  wrote {out}/prior_clip_diagnostic.{{csv,json}}")


if __name__ == "__main__":
    main()
