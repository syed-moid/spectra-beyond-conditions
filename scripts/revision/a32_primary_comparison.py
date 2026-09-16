"""Round 48 A3/A4/B6: the W = 0 control and the all-case primary comparison.

Three things, on the 600-spectrum subset of `test_v10` at s = 1:

  (A4) The **W = 0 control** -- the scalar objective with the penalty switched
       off, using the same optimizer (L-BFGS-B), initialisation, bounds and
       tolerances as the regularized fit. Without it, "regularized vs
       unregularized" confounds the penalty with the change of solver, because
       the unregularized matched fit uses trust-region least squares.

  (A3) Failure rates recomputed under the **per-solver** success test:
       `least_squares` -> result.status > 0; L-BFGS-B -> result.success. Both
       plus the identical bound check. The before/after is reported.

  (B6) The **all-case** table: every method scored on all 600 cases, failures
       replaced by the conditions-only conditional median. Successful-fit-only
       errors are retained beside it as a secondary diagnostic.

Evaluation only; no training.
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import skewnorm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import a3_fitting_baselines as A3
import a3b_extra_baselines as A3B
import a4_train_v8 as A4
from a17_arch_sweep import build_variant
from sbc.data.dataset import InsSpectraDataset
from sbc.models.spectral_transformer import TargetStats, unstandardize_outputs
from sbc.models.nonlinear_conditions_mlp import normalize_conditions
from sbc.data.merit import merit
from sbc.data.latent_perturbations import (ALPHA_MU_LOG, ALPHA_SIGMA_LOG, BETA_XI2_HIGH,
    BETA_XI2_LOW, BETA_XI2_SKEW, GAMMA_FLOOR, _BETA_XI2_LOC, _BETA_XI2_SCALE)
from sbc.data.spectrum_generator import Gamma_Q, omega_Q
import dataset_paths as DS

METHODS = {
    "dho_matched":       (A3.fit_matched,      "matched-model fit (trust region, no penalty)"),
    "dho_matched_w0":    (lambda *a: A3.fit_matched_prior(*a, weight=0.0),
                                               "W = 0 control (L-BFGS-B, no penalty)"),
    "dho_matched_prior": (A3.fit_matched_prior, "prior-regularized forward-model fit"),
    # Round 49 item 5: these are two DIFFERENT implementations, and until now the
    # full-range fit was reported under the windowed fit's name. Both are run.
    "dho_single":        (A3.fit_single,        "single-mode fit, full range (±15 meV)"),
    "dho_single_win":    (A3B.fit_single_windowed, "windowed single-mode fit (±8 meV)"),
}
PARAM_NAMES = {
    "dho_matched":       ["omega0", "Gamma_soft", "Gamma_ac", "Gamma_op", "cp_width",
                          "energy_offset", "res_sigma", "bg_intercept", "bg_slope", "scale"],
    "dho_matched_w0":    ["omega0", "Gamma_soft", "Gamma_ac", "Gamma_op", "cp_width",
                          "energy_offset", "res_sigma", "bg_intercept", "bg_slope", "scale"],
    "dho_matched_prior": ["omega0", "Gamma_soft", "Gamma_ac", "Gamma_op", "cp_width",
                          "energy_offset", "res_sigma", "bg_intercept", "bg_slope", "scale"],
    "dho_single":        ["A", "omega0", "Gamma", "A_cp", "w_cp", "bg_intercept", "bg_slope"],
    "dho_single_win":    ["A", "omega0", "Gamma", "A_cp", "w_cp", "bg_intercept", "bg_slope"],
}


NETS = [("cnn_kernel45", "variant"), ("tf_patch60", "variant"), ("fusion", "base"),
        ("cnn", "base"), ("5a", "base"), ("5b", "base")]
DS_SEEDS, TR_SEEDS = (0, 1, 2), (42, 43, 44)


def _load(model, kind, ds, sd, root):
    ck = torch.load(root / f"ds{ds}" / f"{model}_v7_seed{sd}.pt", map_location="cpu",
                    weights_only=False)
    net = build_variant(model, sd) if kind == "variant" else A4.build_model(ck["arch"], sd)
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net, TargetStats.from_dict(ck["target_stats"])


@torch.no_grad()
def _pred(net, stats, X, cond=None, chunk=2048):
    o = []
    for i in range(0, len(X), chunk):
        xb = torch.from_numpy(X[i:i + chunk].astype(np.float32))
        o.append(net(xb).numpy() if cond is None else
                 net(xb, torch.from_numpy(cond[i:i + chunk].astype(np.float32))).numpy())
    return unstandardize_outputs(np.concatenate(o, 0), stats)


def network_per_spectrum(d, idx, cond, sev):
    """Per-spectrum predictions, averaged over the nine runs, for every architecture."""
    X = np.stack([d.get_augmented(int(i), sev) for i in idx]).astype(np.float32)
    root = DS.rev("A21") / "checkpoints"
    out = {}
    for model, kind in NETS:
        acc = {"M": [], "omega_Q": [], "Gamma_Q": []}
        for ds_ in DS_SEEDS:
            for sd in TR_SEEDS:
                net, st = _load(model, kind, ds_, sd, root)
                P = _pred(net, st, X, cond if model == "fusion" else None)
                for k in acc:
                    acc[k].append(P[k])
        out[model] = {k: np.mean(v, axis=0) for k, v in acc.items()}
        print(f"  network {model:14s} 9 runs", flush=True)
    return out


def logM_with_fallback(ok, logM_fit, logM_cond):
    """Failed fits take the conditional median of log M, not merit() at the medians.

    Round 51. `conditional_params` returns the median of log M over redrawn
    latents, which is the MAE-optimal conditions-only predictor for this target.
    Until this round the per-spectrum table replaced a failed fit's (omega0,
    Gamma) with their own conditional medians and then pushed that PAIR through
    merit(), giving log merit(median omega0, median Gamma). That is a different
    estimator, and it is the one round 49 established was wrong for the reference
    itself -- the reference was corrected, the failed-fit predictions were not.
    `a33_severity_map.py` has always used the form below.

    The two agree closely here (the all-case log-M MAE moves by at most 0.0002
    nats), which is why it survived three rounds of review. Being close is not
    being the same estimator.
    """
    ok = np.asarray(ok, dtype=bool)
    return np.where(ok, np.asarray(logM_fit, dtype=float),
                    np.asarray(logM_cond, dtype=float))


def conditional_params(T, c, E, n, rng):
    """Conditions-only conditional medians of omega0, Gamma AND log M.

    log M is the median of log M over the draws, NOT merit() evaluated at the
    median omega0 and Gamma -- those are different numbers, and the MAE-optimal
    conditions-only predictor for the log M target is the former. An earlier
    version of this script used the latter.
    """
    om0, gm0 = omega_Q(T, c / 100.0, E), Gamma_Q(T, c / 100.0, E)
    ax = rng.lognormal(ALPHA_MU_LOG, ALPHA_SIGMA_LOG, n) * rng.standard_normal(n)
    raw = skewnorm.rvs(a=BETA_XI2_SKEW, size=n, random_state=rng)
    bx2 = np.clip(raw * _BETA_XI2_SCALE + _BETA_XI2_LOC, BETA_XI2_LOW, BETA_XI2_HIGH)
    om = om0 * (1.0 + bx2)
    gm = gm0 * np.maximum(GAMMA_FLOOR, 1.0 + ax)
    lm = np.log(np.clip([merit(float(a), float(b), T, E) for a, b in zip(om, gm)], 1e-9, None))
    return float(np.median(om)), float(np.median(gm)), float(np.median(lm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--oracle-draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()
    out = args.outdir or DS.rev("A23"); out.mkdir(parents=True, exist_ok=True)

    d = InsSpectraDataset(DS.TEST, "test", severity=args.severity, as_torch=False)
    g = d._omega_grid
    idx = np.linspace(0, len(d) - 1, min(args.n, len(d))).astype(int)   # identical to A23
    om_t = d._omega_Q[idx].astype(float); gm_t = d._Gamma_Q[idx].astype(float)
    over = gm_t > om_t
    print(f"  {len(idx)} spectra, {over.mean():.2%} overdamped")

    rng = np.random.default_rng(args.seed)
    repl3 = np.array([conditional_params(float(d._T_K[i]), float(d._c_pct[i]),
                                         float(d._E_kVcm[i]), args.oracle_draws, rng)
                      for i in idx])
    repl = repl3[:, :2]
    logM_cond = repl3[:, 2]

    rows = []
    for name, (fn, label) in METHODS.items():
        t0 = time.time()
        for j, i in enumerate(idx):
            y = d.get_augmented(int(i), args.severity).astype(float)
            r = fn(y, g, float(d._T_K[i]), float(d._c_pct[i]), float(d._E_kVcm[i]))
            pv = {f"p{j}_{nm}": v for j, (nm, v) in
                  enumerate(zip(PARAM_NAMES[name], r.get("params", [])))}
            rows.append({**pv, "method": name, "label": label, "idx": int(i),
                         "omega0_true": om_t[j], "Gamma_true": gm_t[j],
                         "omega0_fit": r["omega0"], "Gamma_fit": r["Gamma"],
                         "ok": bool(r["ok"]), "reason": r["reason"],
                         "omega0_repl": repl[j, 0], "Gamma_repl": repl[j, 1],
                         "overdamped": bool(over[j])})
        print(f"  {name:20s} {len(idx)} fits in {time.time()-t0:.0f}s", flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(out / "primary_fit_results.csv", index=False)
    # Round 49 item 6: the complete optimized vector for every fit and every
    # case, so Figure 7 can be rebuilt through the same forward model used in
    # fitting rather than from a re-fit.
    for name, gdf in per.groupby("method", sort=False):
        # Select this method's OWN parameter columns. The frame holds the union
        # across methods (10-vector for the matched fits, 7 for the single-mode
        # ones), so a generic "starts with p<digit>" match picks up both sets.
        cols = ["idx", "ok", "reason"] + [f"p{j}_{nm}" for j, nm
                                          in enumerate(PARAM_NAMES[name])]
        gdf[[c for c in cols if c in gdf.columns]].to_csv(
            out / f"fit_parameters_{name}.csv", index=False)
    print(f"  wrote fit_parameters_*.csv for {per.method.nunique()} methods")

    summ = []
    for name, gdf in per.groupby("method", sort=False):
        ok = gdf.ok.to_numpy()
        gm_b = np.where(ok, gdf.Gamma_fit, gdf.Gamma_repl)
        om_b = np.where(ok, gdf.omega0_fit, gdf.omega0_repl)
        ov = gdf.overdamped.to_numpy()
        eg_all = np.abs(gm_b - gdf.Gamma_true.to_numpy())
        eo_all = np.abs(om_b - gdf.omega0_true.to_numpy())
        s_ok = gdf[ok]
        eg_ok = np.abs(s_ok.Gamma_fit - s_ok.Gamma_true)
        summ.append({
            "method": name, "label": gdf.label.iloc[0], "n": len(gdf),
            "failure_rate": float(1 - ok.mean()), "n_failed": int((~ok).sum()),
            # PRIMARY: all cases, failures replaced by the conditions-only median
            "MAE_omega0_all": float(eo_all.mean()), "MAE_Gamma_all": float(eg_all.mean()),
            "median_abs_err_Gamma_all": float(np.median(eg_all)),
            "MAE_Gamma_overdamped_all": float(eg_all[ov].mean()), "n_overdamped_all": int(ov.sum()),
            # SECONDARY: successful fits only
            "n_scored_ok": int(ok.sum()),
            "MAE_omega0_ok": float(np.abs(s_ok.omega0_fit - s_ok.omega0_true).mean()),
            "MAE_Gamma_ok": float(eg_ok.mean()),
            "median_abs_err_Gamma_ok": float(np.median(eg_ok)),
            "MAE_Gamma_overdamped_ok": float(
                np.abs(s_ok.Gamma_fit - s_ok.Gamma_true)[s_ok.overdamped.to_numpy()].mean()),
        })
    sm = pd.DataFrame(summ); sm.to_csv(out / "primary_fit_summary.csv", index=False)

    # Per-spectrum table for the figures: every fit and every network on one row
    # per spectrum, so fig1 and fig2 are drawn from one aligned population.
    cond = normalize_conditions(d._T_K[idx], d._c_pct[idx], d._E_kVcm[idx])
    nets = network_per_spectrum(d, idx, cond, args.severity)
    lm_true = np.log(np.clip(d._M[idx].astype(float), 1e-9, None))
    ps = pd.DataFrame({"idx": idx, "omega0_true": om_t, "Gamma_true": gm_t,
                       "logM_true": lm_true, "overdamped": over,
                       "omega0_cond": repl[:, 0], "Gamma_cond": repl[:, 1]})
    ps["logM_cond"] = logM_cond
    for model, P in nets.items():
        ps[f"omega0_{model}"] = P["omega_Q"]
        ps[f"Gamma_{model}"] = P["Gamma_Q"]
        ps[f"logM_{model}"] = np.log(np.clip(P["M"], 1e-9, None))
    for name, gdf in per.groupby("method", sort=False):
        gg = gdf.set_index("idx").loc[idx]
        ok = gg.ok.to_numpy()
        o_b = np.where(ok, gg.omega0_fit, gg.omega0_repl)
        g_b = np.where(ok, gg.Gamma_fit, gg.Gamma_repl)
        ps[f"omega0_{name}"] = o_b
        ps[f"Gamma_{name}"] = g_b
        ps[f"ok_{name}"] = ok
        lm_fit = [np.log(max(merit(float(a), float(b), float(t), float(e)), 1e-9))
                  for a, b, t, e in zip(o_b, g_b, d._T_K[idx], d._E_kVcm[idx])]
        ps[f"logM_{name}"] = logM_with_fallback(ok, lm_fit, logM_cond)
    ps.to_csv(out / "primary_per_spectrum.csv", index=False)
    # THE authoritative conditions-only reference on these 600 cases. Computed
    # once here; a33 and the figures read it rather than re-estimating, so one
    # number is used everywhere.
    ref = float(np.abs(logM_cond - lm_true).mean())
    (out / "conditions_only_reference.json").write_text(json.dumps({
        "MAE_logM": ref, "n": int(len(idx)), "draws_per_condition": int(args.oracle_draws),
        "seed": int(args.seed), "eval_set": str(DS.TEST.relative_to(ROOT)),
        "subset": "np.linspace(0, N-1, 600) of the test split",
        "predictor": "conditional median of log M over redrawn latents",
        "note": ("the median of log M over draws, not merit() at the median omega0 and "
                 "Gamma; those differ and only the former is the MAE-optimal "
                 "conditions-only predictor for this target"),
        "date_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"  authoritative conditions-only reference on the 600 cases: {ref:.6f}")
    print(f"  wrote primary_per_spectrum.csv ({len(ps)} rows, {len(ps.columns)} cols)")
    cols = ["method", "failure_rate", "MAE_Gamma_all", "MAE_Gamma_overdamped_all",
            "n_scored_ok", "MAE_Gamma_ok", "MAE_omega0_all"]
    print("\n" + sm[cols].round(4).to_string(index=False))

    # gap closure on the all-case numbers, against the tuned CNN
    net = pd.read_csv(out / "network_vs_fit_test.csv").set_index("method")
    cnn_g, cnn_go = float(net.loc["cnn_kernel45"].MAE_Gamma), float(net.loc["cnn_kernel45"].MAE_Gamma_overdamped)
    S = sm.set_index("method")
    def closure(baseline, tag, cnn_ref):
        """Fraction of the baseline-to-CNN gap that the penalty closes."""
        b = float(S.loc[baseline, tag]); pr = float(S.loc["dho_matched_prior", tag])
        gap = b - cnn_ref
        return {"baseline": baseline, "baseline_MAE": b, "regularized_MAE": pr,
                "cnn_MAE": cnn_ref, "gap_meV": gap, "closed_meV": b - pr,
                "fraction": (b - pr) / gap if gap else float("nan")}

    res = {}
    for tag, lbl, ref in (("MAE_Gamma_all", "overall", cnn_g),
                          ("MAE_Gamma_overdamped_all", "overdamped", cnn_go)):
        res[f"{lbl}_vs_trust_region"] = closure("dho_matched", tag, ref)
        res[f"{lbl}_vs_W0_control"] = closure("dho_matched_w0", tag, ref)
    # How much of the trust-region-vs-regularized difference is the optimizer alone?
    for tag, lbl in (("MAE_Gamma_all", "overall"), ("MAE_Gamma_overdamped_all", "overdamped")):
        res[f"{lbl}_optimizer_only"] = {
            "trust_region_MAE": float(S.loc["dho_matched", tag]),
            "W0_control_MAE": float(S.loc["dho_matched_w0", tag]),
            "difference_meV": float(S.loc["dho_matched", tag] - S.loc["dho_matched_w0", tag]),
            "note": ("same objective without the penalty, one solver each; this is the "
                     "optimizer's own contribution")}
    (out / "primary_gap_fractions.json").write_text(json.dumps(res, indent=2))
    print("\n  gap closures (all-case):")
    for k, v in res.items():
        if "fraction" in v:
            print(f"    {k:28s} closed {v['closed_meV']:+.4f} of {v['gap_meV']:.4f}"
                  f"  = {v['fraction']:.1%}")
        else:
            print(f"    {k:28s} optimizer alone: {v['difference_meV']:+.4f} meV")

    (out / "run_sidecar_primary.json").write_text(json.dumps({
        "task": "A32_primary_comparison", "dataset_version": DS.VERSION,
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "eval_set": str(DS.TEST.relative_to(ROOT)),
        "subset": "np.linspace(0, N-1, 600) of the test split; identical to A23",
        "severity": args.severity,
        "primary_policy": "all 600 cases; failures replaced by the conditions-only conditional median",
        "secondary_policy": "successful fits only",
        "success_test": {"least_squares": "result.status > 0",
                         "L-BFGS-B": "result.success (status 0 means converged)"},
        "bound_check": "soft-mode omega0 or Gamma within a relative 1e-3 of either bound",
        "W0_control": "same objective, optimizer, init, bounds and tolerances, penalty weight 0",
        "training": "none", "seed": args.seed,
        "date_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"\n  wrote {out}/primary_fit_summary.csv")


if __name__ == "__main__":
    main()
