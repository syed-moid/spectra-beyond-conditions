"""A3b - fairer conventional-fitting variants + per-spectrum ST predictions.

The first-pass DHO-single of a3_fitting_baselines.py locks onto the fixed
acoustic mode at 9.2 meV for ~40% of spectra, which understates what a careful
experimentalist would achieve. Two fairer variants are added:

  dho_single_win   the same 7-parameter single DHO + central Lorentzian +
                   linear background, but fitted only over |omega| <= 8 meV,
                   i.e. the low-energy soft-mode region an experimentalist
                   would isolate. Structurally cannot represent the T <~ 250 K
                   spectra whose soft mode lies above 8 meV.

  dho_two_mode     soft DHO + a second free DHO (to absorb the acoustic
                   contribution) + central Lorentzian + linear background,
                   fitted over the full window. Mode assignment uses the
                   BASELINE omega0(T, c, E) - metadata knowledge an
                   experimentalist has from a published dispersion - never the
                   realized latent value.

Also materialises per-spectrum ST-5a / ST-5b predictions on the stress base at
every severity from the existing v7 checkpoints, so the fitting baselines can
be compared to the spectral models with a paired bootstrap rather than by
eyeballing two tables.
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
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from a3_fitting_baselines import _at_bound, model_single  # noqa: E402
from sbc.data.dataset import InsSpectraDataset  # noqa: E402
from sbc.data.merit import merit  # noqa: E402
from sbc.data.spectrum_generator import Gamma_Q, bose, omega_Q  # noqa: E402
import a3_fitting_baselines as A3  # noqa: E402
import dataset_paths as DS  # noqa: E402

NPZ = DS.FULL
ST_RUN = ROOT / "results" / "phase1_st_run"
A2_PRED = ROOT / "results" / "revision" / "A2" / "predictions_seedavg.csv"
OUTDIR = DS.rev("A3")
SEVERITIES = (0.25, 0.5, 1.0, 2.0, 4.0)
ST_SEEDS = (42, 43, 44)
WINDOW_MEV = 8.0
TRACK_HALF_WIDTH = 4.0


def model_two(p, grid, T):
    """p = [A1, om1, g1, A2, om2, g2, A_cp, w_cp, b0, b1]."""
    A1, om1, g1, A2, om2, g2, A_cp, w_cp, b0, b1 = p
    out = b0 + b1 * grid + A_cp * w_cp / (np.pi * (grid ** 2 + w_cp ** 2))
    for A, om, g in ((A1, om1, g1), (A2, om2, g2)):
        out = out + A3._dho_lineshape(grid, A, om, g, T)
    return out


def _amp_init(ymax, om, gm, T):
    """Delegates to the shared initialiser, which matches the corrected lineshape."""
    return A3._amp_for_peak(ymax, om, gm, T)


def fit_single_windowed(y, grid, T, c, E, half_width=WINDOW_MEV):
    m = np.abs(grid) <= half_width
    g, yy = grid[m], y[m]
    om_b = omega_Q(T, c / 100.0, E); gm_b = Gamma_Q(T, c / 100.0, E)
    ymax = max(float(np.max(yy)), 1e-12); ymin = float(np.min(yy))
    om0 = min(om_b, half_width - 0.5)
    p0 = np.array([_amp_init(ymax, om0, gm_b, T), om0, gm_b, 0.05 * ymax, 1.5, max(ymin, 0.0), 0.0])
    lo = np.array([1e-12, 0.5, 0.02, 0.0, 0.05, -0.5 * ymax, -0.5 * ymax / half_width])
    hi = np.array([1e6 * p0[0], half_width + 0.5, 40.0, 50.0 * ymax, 12.0, 1.5 * ymax,
                   0.5 * ymax / half_width])
    p0 = np.clip(p0, lo + 1e-12, hi - 1e-12)
    try:
        r = least_squares(lambda p: (model_single(p, g, T) - yy) / ymax, p0,
                          bounds=(lo, hi), max_nfev=4000, method="trf")
    except Exception:
        return dict(ok=False, reason="exception", omega0=np.nan, Gamma=np.nan, status=-99)
    p = r.x
    ok = r.status > 0
    reason = "" if ok else "no_convergence"
    if _at_bound(p[1], lo[1], hi[1]):
        ok, reason = False, "omega0_at_bound"
    elif _at_bound(p[2], lo[2], hi[2]):
        ok, reason = False, "Gamma_at_bound"
    return dict(ok=ok, reason=reason, omega0=float(p[1]), Gamma=float(p[2]),
                status=int(r.status), params=[float(v) for v in p])


def fit_single_tracked(y, grid, T, c, E, half_width=TRACK_HALF_WIDTH):
    """Single DHO fitted in a window CENTRED on the baseline omega0(T, c, E).

    The generator's own baseline dispersion (spectrum_generator.py:161-164) stands
    in for "the published dispersion": it is what the generator was fitted to, and
    it is metadata knowledge (a function of T, c, E) with no access to the realized
    latent state. Half-width 4 meV, clipped to the grid.
    """
    om_b = omega_Q(T, c / 100.0, E); gm_b = Gamma_Q(T, c / 100.0, E)
    lo_w, hi_w = om_b - half_width, om_b + half_width
    m = (grid >= lo_w) & (grid <= hi_w)
    if int(m.sum()) < 20:
        return dict(ok=False, reason="window_too_narrow", omega0=np.nan, Gamma=np.nan, status=-98)
    g, yy = grid[m], y[m]
    ymax = max(float(np.max(yy)), 1e-12); ymin = float(np.min(yy))
    p0 = np.array([_amp_init(ymax, om_b, gm_b, T), om_b, gm_b, 0.05 * ymax, 1.5,
                   max(ymin, 0.0), 0.0])
    lo = np.array([1e-12, max(0.5, lo_w), 0.02, 0.0, 0.05, -0.5 * ymax,
                   -0.5 * ymax / half_width])
    hi = np.array([1e6 * p0[0], hi_w, 40.0, 50.0 * ymax, 12.0, 1.5 * ymax,
                   0.5 * ymax / half_width])
    p0 = np.clip(p0, lo + 1e-12, hi - 1e-12)
    try:
        r = least_squares(lambda p: (model_single(p, g, T) - yy) / ymax, p0,
                          bounds=(lo, hi), max_nfev=4000, method="trf")
    except Exception:
        return dict(ok=False, reason="exception", omega0=np.nan, Gamma=np.nan, status=-99)
    p = r.x
    ok = r.status > 0
    reason = "" if ok else "no_convergence"
    if _at_bound(p[1], lo[1], hi[1]):
        ok, reason = False, "omega0_at_bound"
    elif _at_bound(p[2], lo[2], hi[2]):
        ok, reason = False, "Gamma_at_bound"
    return dict(ok=ok, reason=reason, omega0=float(p[1]), Gamma=float(p[2]),
                status=int(r.status), params=[float(v) for v in p])


def fit_two_mode(y, grid, T, c, E):
    om_b = omega_Q(T, c / 100.0, E); gm_b = Gamma_Q(T, c / 100.0, E)
    ymax = max(float(np.max(y)), 1e-12); ymin = float(np.min(y))
    p0 = np.array([_amp_init(ymax, om_b, gm_b, T), om_b, gm_b,
                   _amp_init(0.5 * ymax, 9.2, 0.6, T), 9.2, 0.6,
                   0.05 * ymax, 1.5, max(ymin, 0.0), 0.0])
    lo = np.array([1e-12, 0.5, 0.02, 1e-12, 0.5, 0.02, 0.0, 0.05, -0.5 * ymax, -0.5 * ymax / 15.0])
    hi = np.array([1e6 * p0[0], 26.0, 40.0, 1e6 * max(p0[3], 1e-9), 26.0, 40.0,
                   50.0 * ymax, 12.0, 1.5 * ymax, 0.5 * ymax / 15.0])
    p0 = np.clip(p0, lo + 1e-12, hi - 1e-12)
    try:
        r = least_squares(lambda p: (model_two(p, grid, T) - y) / ymax, p0,
                          bounds=(lo, hi), max_nfev=6000, method="trf")
    except Exception:
        return dict(ok=False, reason="exception", omega0=np.nan, Gamma=np.nan, status=-99)
    p = r.x
    # Mode assignment from the BASELINE omega0 (metadata, not the latent truth).
    pick = 0 if abs(p[1] - om_b) <= abs(p[4] - om_b) else 1
    om_f, gm_f = (p[1], p[2]) if pick == 0 else (p[4], p[5])
    i_om, i_gm = (1, 2) if pick == 0 else (4, 5)
    ok = r.status > 0
    reason = "" if ok else "no_convergence"
    if _at_bound(om_f, lo[i_om], hi[i_om]):
        ok, reason = False, "omega0_at_bound"
    elif _at_bound(gm_f, lo[i_gm], hi[i_gm]):
        ok, reason = False, "Gamma_at_bound"
    return dict(ok=ok, reason=reason, omega0=float(om_f), Gamma=float(gm_f),
                status=int(r.status), assigned_second=int(pick))


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--severities", type=float, nargs="+", default=list(SEVERITIES))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    ap.add_argument("--npz", type=Path, default=NPZ)
    ap.add_argument("--dataset-version", type=str, default=DS.VERSION)
    ap.add_argument("--a2-pred", type=Path, default=A2_PRED)
    ap.add_argument("--st-source", type=str, default="v7_run",
                    choices=("v7_run", "a4_v8"),
                    help="v7_run: results/phase1_st_run checkpoints; a4_v8: revision/A4 checkpoints")
    ap.add_argument("--st-archs", type=str, nargs="+", default=["5a", "5b"])
    ap.add_argument("--st-protocol", type=str, default="v7")
    ap.add_argument("--st-ckpt-dir", type=Path, default=None)
    ap.add_argument("--st-seeds", type=int, nargs="+", default=list(ST_SEEDS))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    ds = InsSpectraDataset(args.npz, "stress_base", severity=1.0, as_torch=False)
    grid = ds._omega_grid
    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    T = ds._T_K[:n].astype(float); c = ds._c_pct[:n].astype(float); E = ds._E_kVcm[:n].astype(float)
    om_true = ds._omega_Q[:n].astype(float); gm_true = ds._Gamma_Q[:n].astype(float)
    logM_true = np.log(np.clip(ds._M[:n].astype(float), 1e-9, None))
    a2 = pd.read_csv(args.a2_pred)
    fb = a2[(a2.model == "mlp_l1") & (a2.split == "stress_base")]["logM_pred_seedavg"].to_numpy()[:n]

    # cache augmented spectra once per severity (reused by fits and by the ST forward pass)
    rows = []
    st_rows = []
    import torch  # noqa: E402
    from sbc.models.spectral_transformer import SpectralTransformerModel  # noqa: E402
    st_models = {}
    for a in args.st_archs:
        for s in args.st_seeds:
            if args.st_source == "v7_run":
                path = ST_RUN / f"{a}_seed{s}" / "checkpoints" / "best.pt"
                st_models[(a, s)] = SpectralTransformerModel.from_checkpoint(path)
            else:
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                from a4_eval import CKPT_DIR, load_a4_model  # noqa: E402
                st_models[(a, s)] = load_a4_model(
                    a, args.st_protocol, s, ckpt_dir=args.st_ckpt_dir or CKPT_DIR)

    for sev in args.severities:
        t0 = time.time()
        Y = np.stack([ds.get_augmented(i, severity=sev) for i in range(n)])
        for method, fn in (("dho_single_win", fit_single_windowed),
                           ("dho_single_track", fit_single_tracked),
                           ("dho_two_mode", fit_two_mode)):
            for i in range(n):
                r = fn(Y[i], grid, float(T[i]), float(c[i]), float(E[i]))
                om, gm = r["omega0"], r["Gamma"]
                lm = (np.log(max(merit(om, gm, float(T[i]), float(E[i])), 1e-9))
                      if np.isfinite(om) and np.isfinite(gm) and gm > 0 else np.nan)
                rows.append({"dataset_version": args.dataset_version,
                             "method": method, "severity": sev, "idx": i,
                             "T_K": T[i], "c_pct": c[i], "E_kVcm": E[i],
                             "omega0_fit": om, "Gamma_fit": gm,
                             "omega0_true": om_true[i], "Gamma_true": gm_true[i],
                             "logM_fit": lm, "logM_true": logM_true[i],
                             "logM_metadata_fallback": fb[i],
                             "fit_ok": r["ok"], "fail_reason": r["reason"], "status": r["status"]})
        Xf = Y.astype(np.float32)
        for arch in args.st_archs:
            for s in args.st_seeds:
                m = st_models[(arch, s)]
                p = m.predict_M({"spectrum": Xf, "T_K": T, "c_pct": c, "E_kVcm": E})
                st_rows.append(pd.DataFrame({
                    "dataset_version": args.dataset_version,
                    "model": arch if arch in ("cnn", "fusion") else f"ST-{arch}",
                    "seed": s, "severity": sev, "idx": np.arange(n),
                    "logM_pred": np.log(np.clip(p, 1e-9, None)), "logM_true": logM_true}))
        print(f"  severity {sev}: done in {time.time()-t0:.0f}s", flush=True)

    extra = pd.DataFrame(rows)
    extra.to_csv(args.outdir / "fit_results_extra_variants.csv", index=False)
    st = pd.concat(st_rows, ignore_index=True)
    st.to_csv(args.outdir / "st_predictions_stress_base.csv", index=False)

    summ = []
    for method in ("dho_single_win", "dho_single_track", "dho_two_mode"):
        for sev in args.severities:
            d = extra[(extra.method == method) & (extra.severity == sev)]
            ok = d.fit_ok.astype(bool).to_numpy()
            gt = d.Gamma_true / d.omega0_true; ft = d.Gamma_fit / d.omega0_fit
            lm_repl = np.where(ok & np.isfinite(d.logM_fit), d.logM_fit, d.logM_metadata_fallback)
            summ.append({"dataset_version": args.dataset_version,
                         "method": method, "severity": sev, "N": len(d),
                         "fit_failure_rate": float(1 - ok.mean()),
                         "fail_no_convergence": float(np.mean(d.fail_reason == "no_convergence")),
                         "fail_Gamma_at_bound": float(np.mean(d.fail_reason == "Gamma_at_bound")),
                         "fail_omega0_at_bound": float(np.mean(d.fail_reason == "omega0_at_bound")),
                         "MAE_omega0_meV_ok": float(np.abs(d.omega0_fit - d.omega0_true)[ok].mean()),
                         "MAE_Gamma_meV_ok": float(np.abs(d.Gamma_fit - d.Gamma_true)[ok].mean()),
                         "MAE_Gamma_over_omega0_ok": float(np.abs(ft - gt)[ok].mean()),
                         "MAE_logM_failures_excluded":
                             float(np.abs(d.logM_fit - d.logM_true)[ok & np.isfinite(d.logM_fit)].mean()),
                         "MAE_logM_failures_replaced_by_metadata":
                             float(np.abs(lm_repl - d.logM_true).mean())})
    for arch in args.st_archs:
        label = arch if arch in ("cnn", "fusion") else f"ST-{arch}"
        for sev in args.severities:
            per = [np.abs(st[(st.model == label) & (st.severity == sev) & (st.seed == s)]
                          ["logM_pred"].to_numpy() - logM_true).mean() for s in args.st_seeds]
            summ.append({"dataset_version": args.dataset_version,
                         "method": label, "severity": sev, "N": n, "fit_failure_rate": 0.0,
                         "MAE_logM_failures_excluded": float(np.mean(per)),
                         "MAE_logM_failures_replaced_by_metadata": float(np.mean(per)),
                         "MAE_logM_sd_over_seeds": float(np.std(per, ddof=1))})
    sdf = pd.DataFrame(summ)
    sdf.to_csv(args.outdir / "summary_extra_variants.csv", index=False)
    print(sdf.to_string(index=False))

    (args.outdir / "run_sidecar_extra.json").write_text(json.dumps({
        "task": "A3b", "script": str(Path(__file__).relative_to(ROOT)),
        "dataset_version": args.dataset_version,
        "dataset_npz": str(Path(args.npz).resolve().relative_to(ROOT)),
        "master_seed": args.seed, "severities": list(args.severities),
        "n_stress_spectra": int(n), "window_meV": WINDOW_MEV,
        "track_half_width_meV": TRACK_HALF_WIDTH,
        "st_seeds": list(args.st_seeds), "st_source": args.st_source,
        "metadata_fallback_model": "mlp_l1 (A2, seed-averaged)",
        "git_commit": git_sha(), "date_utc": datetime.now(timezone.utc).isoformat(),
        "numpy": np.__version__, "pandas": pd.__version__}, indent=2))
    print(f"\nwrote {args.outdir}")


if __name__ == "__main__":
    main()
