"""A12 - consolidated results digest.

Assembly only: reads existing result files, emits the digest tables into
results/revision/A12/, and asserts (a) a single run_id per table where a run_id
exists, (b) every cited path exists. Resume-on-entry: tables already written are
skipped unless --force.
"""

from __future__ import annotations

import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import dataset_paths as DS

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "results" / "revision"
OUT = DS.rev("A12")
MISSING: list[str] = []
SKIPPED: list[tuple] = []



# Where a task's outputs moved between versions. The tuned architecture variants
# were trained into A19 on v9 but into A17 on v10, because the v10 sweep trained
# every variant into one directory. Applied only when the primary path is absent,
# so v9 resolution is unchanged.
_TASK_FALLBACK = {"A19": "A17"}


def _resolve(rel):
    """REV-relative path with the leading task segment mapped to the active version.

    Segments that are not declared tasks (A2, A2_v8, bare files) pass through
    unchanged: those records are version-invariant or deliberately historical.
    """
    head, _, tail = str(rel).partition("/")
    try:
        base = DS.rev(head)
    except KeyError:
        return REV / rel
    p = base / tail if tail else base
    if not p.exists() and head in _TASK_FALLBACK:
        try:
            alt = DS.rev(_TASK_FALLBACK[head])
        except KeyError:
            return p
        q = alt / tail if tail else alt
        if q.exists():
            return q
    return p


def rd(rel, **kw):
    p = _resolve(rel)
    if not p.exists():
        MISSING.append(str(p.relative_to(ROOT)))
        return None
    try:
        return pd.read_csv(p, **kw)
    except pd.errors.EmptyDataError:
        # A table with no rows and no header: legitimate when the quantity has
        # no instances (e.g. no repeated runs to compare). Record it and carry
        # on rather than failing the whole digest.
        MISSING.append(f"{p.relative_to(ROOT)} (empty)")
        return None


def path_of(rel):
    return str(_resolve(rel).relative_to(ROOT))


# ---------------------------------------------------------------------------
# Controlled vocabulary (round-29 repair). No free-text model strings.
# ---------------------------------------------------------------------------

MODEL_IDS = {
    # spectral, base family
    "5a": ("tf_learned_patch", "learned-patch transformer"),
    "5b": ("tf_fourier", "Fourier-feature transformer"),
    "fusion": ("tf_fused", "fused transformer"),
    "cnn": ("cnn_base", "1D CNN"),
    "tf_base": ("tf_learned_patch", "learned-patch transformer"),
    "cnn_base": ("cnn_base", "1D CNN"),
    "ST-5a": ("tf_learned_patch", "learned-patch transformer"),
    "ST-5b": ("tf_fourier", "Fourier-feature transformer"),
    # tuned variants
    "cnn_kernel45": ("cnn_k45", "tuned 1D CNN (kernel 45)"),
    "tf_patch60": ("tf_p60", "tuned transformer (patch 60)"),
    "cnn_kernel15": ("cnn_k15", "1D CNN (kernel 15)"),
    "cnn_kernel21": ("cnn_k21", "1D CNN (kernel 21)"),
    "cnn_kernel31": ("cnn_k31", "1D CNN (kernel 31)"),
    "cnn_param_matched": ("cnn_pm", "1D CNN (parameter-matched)"),
    "cnn_half_depth": ("cnn_hd", "1D CNN (half depth)"),
    "cnn_wide": ("cnn_wide", "1D CNN (wide)"),
    "tf_patch5": ("tf_p5", "transformer (patch 5)"),
    "tf_patch20": ("tf_p20", "transformer (patch 20)"),
    "tf_patch30": ("tf_p30", "transformer (patch 30)"),
    "tf_patch40": ("tf_p40", "transformer (patch 40)"),
    "tf_depth8": ("tf_d8", "transformer (8 layers)"),
    # A16 protocol variants
    "5a_fixed1": ("tf_learned_patch_fixed1", "learned-patch transformer (fixed s=1)"),
    "5a_low": ("tf_learned_patch_low", "learned-patch transformer (s<=1)"),
    "fusion_fixed1": ("tf_fused_fixed1", "fused transformer (fixed s=1)"),
    # metadata
    "metadata_mlp_l1": ("metadata_mlp_l1", "metadata MLP (L1)"),
    "metadata mlp_l1": ("metadata_mlp_l1", "metadata MLP (L1)"),
    "mlp_l1": ("metadata_mlp_l1", "metadata MLP (L1)"),
    "metadata_mlp_mse": ("metadata_mlp_mse", "metadata MLP (MSE)"),
    "mlp_mse": ("metadata_mlp_mse", "metadata MLP (MSE)"),
    "metadata_mlp_mse_innerval": ("metadata_mlp_mse_inner", "metadata MLP (MSE, inner-split)"),
    "mlp_mse_innerval": ("metadata_mlp_mse_inner", "metadata MLP (MSE, inner-split)"),
    "metadata_hgb": ("metadata_hgb", "metadata GBT"),
    "metadata hgb": ("metadata_hgb", "metadata GBT"),
    "hgb": ("metadata_hgb", "metadata GBT"),
    "metadata_knn": ("metadata_knn", "metadata k-NN"),
    "knn": ("metadata_knn", "metadata k-NN"),
    "metadata_oracle": ("metadata_oracle", "metadata oracle (bound)"),
    "metadata oracle (bound)": ("metadata_oracle", "metadata oracle (bound)"),
    # fitting
    "dho_matched": ("fit_matched", "matched-model fit"),
    "dho_matched_prior": ("fit_matched_prior", "prior-regularized forward-model fit"),
    "dho_two_mode": ("fit_two_mode", "two-mode fit"),
    "dho_single_win": ("fit_single_win", "windowed single-mode fit"),
    "dho_single_track": ("fit_single_track", "dispersion-tracking single-mode fit"),
    "dho_single": ("fit_single_naive", "naive single-mode fit"),
    "feat_gbm": ("feat_gbm", "feature regression"),
}
METADATA_IDS = {"metadata_mlp_l1", "metadata_mlp_mse", "metadata_mlp_mse_inner",
                "metadata_hgb", "metadata_knn", "metadata_oracle"}
FIT_IDS = {"fit_matched", "fit_matched_prior", "fit_two_mode", "fit_single_win",
           "fit_single_track", "fit_single_naive"}

# ---------------------------------------------------------------------------
# Metric applicability (round-38 repair 2). Some metrics only mean anything for
# a fitting method: a learned model has no fit to fail, so "MAE with failures
# replaced by the metadata prediction" is its plain MAE under a name that
# claims otherwise. Metrics absent from this map apply to every class.
# ---------------------------------------------------------------------------
METRIC_APPLICABILITY = {
    "MAE_logM_failures_replaced": {"fit"},
    "MAE_logM_failures_excluded": {"fit"},
    "fit_failure_rate": {"fit"},
    "fail_no_convergence": {"fit"},
    "fail_Gamma_at_bound": {"fit"},
    "fail_omega0_at_bound": {"fit"},
}


def model_class(model_id):
    if model_id in METADATA_IDS:
        return "metadata"
    if model_id in FIT_IDS:
        return "fit"
    return "learned"


def applicable(model_id, metric):
    allowed = METRIC_APPLICABILITY.get(metric)
    return allowed is None or model_class(model_id) in allowed


def canon(raw):
    if raw not in MODEL_IDS:
        raise KeyError(f"model string not in controlled vocabulary: {raw!r}")
    return MODEL_IDS[raw]


def run_id_for(rel):
    """Source run_id where the file carries one, else a deterministic derived id."""
    import hashlib
    p = _resolve(rel)
    try:
        t = pd.read_csv(p, nrows=1)
        if "run_id" in t.columns and pd.notna(t.run_id.iloc[0]):
            return str(t.run_id.iloc[0])
    except Exception:
        pass
    h = hashlib.sha256(f"{rel}:{p.stat().st_mtime_ns}".encode()).hexdigest()[:12]
    return f"derived:{h}"


def consolidated():
    rows = []

    def add(raw_model, metric, evalset, severity, value, sd, n_seeds, proto, src,
            statistic, dv=None):
        mid, disp = canon(raw_model)
        if not applicable(mid, metric):
            SKIPPED.append((mid, model_class(mid), metric))
            return
        if mid in METADATA_IDS or mid in FIT_IDS:
            proto = None
        rows.append({"model_id": mid, "display_name": disp, "metric": metric,
                     "eval_set": evalset, "severity": severity, "statistic": statistic,
                     "value": value, "sd": sd, "n_seeds": n_seeds, "protocol": proto,
                     "dataset_version": dv or DS.VERSION, "run_id": run_id_for(src),
                     "source_file": path_of(src)})

    g = rd("A4_extended/gate_and_improvement.csv")
    if g is not None:
        for r in g.itertuples():
            proto = f"extended/{r.train_severity_protocol}"
            key = r.arch if r.train_severity_protocol == "v7" else f"{r.arch}_{r.train_severity_protocol}"
            add(key, "MAE_logM", "val", 1.0, r.MAE_logM_mean, r.MAE_logM_sd, r.n_seeds,
                proto, "A4_extended/gate_and_improvement.csv", "per_seed_mean")
            add(key, "MAE_logM", "val", 1.0, r.MAE_logM_ensemble, None, r.n_seeds,
                proto, "A4_extended/gate_and_improvement.csv", "ensemble")
            add(key, "improvement_vs_metadata_reference", "val", 1.0,
                r.improvement_vs_reference, None, r.n_seeds, proto,
                "A4_extended/gate_and_improvement.csv", "ensemble")

    a17 = rd("A17/variant_paired_vs_base.csv")
    if a17 is not None:
        for r in a17.itertuples():
            add(r.variant, "MAE_logM", "val", 1.0, r.MAE_logM_ensemble, None, 3,
                "extended/v7", "A17/variant_paired_vs_base.csv", "ensemble")

    for f, dv in (("A2/metadata_family_metrics.csv", "v7"),
                  ("A2_v8/metadata_family_metrics_v8_newsets.csv", "v8")):
        m = rd(f)
        if m is not None:
            for r in m.itertuples():
                add(r.model, "MAE_logM", r.split, 1.0, r.MAE_logM_mean, r.MAE_logM_sd,
                    r.n_seeds, None, f, "per_seed_mean", dv)

    pj = DS.rev("A1") / "oracle_floor.json"
    if pj.exists():
        for k, v in json.loads(pj.read_text())["summary"].items():
            add("metadata_oracle", "MAE_logM", k, 1.0,
                v["MAE_logM_conditional_median"], None, None, None,
                "A1/oracle_floor.json", "bound")
    else:
        MISSING.append("A1/oracle_floor.json")

    s8 = rd("A8/dense_sweep_stress.csv")
    if s8 is not None:
        for r in s8.itertuples():
            if pd.notna(r.MAE_mean):
                add(r.model, "MAE_logM", "stress_base", r.severity, r.MAE_mean, r.MAE_sd, 3,
                    "extended/v7", "A8/dense_sweep_stress.csv", "per_seed_mean")
            add(r.model, "MAE_logM", "stress_base", r.severity, r.MAE_ensemble, None, 3,
                "extended/v7", "A8/dense_sweep_stress.csv", "ensemble")

    rho = rd("A14/extended_rho_replicate.csv")
    if rho is not None:
        for r in rho.itertuples():
            add(r.model, "within_condition_rho", "replicate_eval", r.severity, r.rho_mean,
                None, 3, "extended/v7", "A14/extended_rho_replicate.csv", "ensemble")
            add(r.model, "MAE_logM", "replicate_eval", r.severity, r.MAE_ensemble, None, 3,
                "extended/v7", "A14/extended_rho_replicate.csv", "ensemble")

    h = rd("A10/holdout_table.csv")
    if h is not None:
        for r in h[~h.model.str.startswith("PAIRED")].itertuples():
            add(r.model, "MAE_logM", r.split, 1.0, r.MAE_logM, None, 3, "extended/v7",
                "A10/holdout_table.csv", "ensemble")

    a6 = rd("A6/aux_head_metrics.csv")
    if a6 is not None:
        for r in a6.itertuples():
            for met in ("MAE_omega0_meV", "MAE_Gamma_meV", "MAE_Gamma_over_omega0"):
                add(r.model, met, "stress_base", r.severity, getattr(r, met), None, 3,
                    "extended/v7", "A6/aux_head_metrics.csv", "ensemble")

    q = rd("A6/logM_error_by_gamma_error.csv")
    if q is not None:
        for r in q.itertuples():
            add(r.model, f"MAE_logM_gamma_err_q{r.gamma_err_bin}", "stress_base", r.severity,
                r.MAE_logM_in_bin, None, 3, "extended/v7",
                "A6/logM_error_by_gamma_error.csv", "ensemble")

    for f in ("A3/summary_by_severity.csv", "A3/summary_extra_variants.csv"):
        t = rd(f)
        if t is not None:
            for r in t.itertuples():
                stat = "per_seed_mean" if r.method == "feat_gbm" else "single"
                sd = getattr(r, "MAE_logM_sd_over_seeds", None)
                add(r.method, "MAE_logM_failures_replaced", "stress_base", r.severity,
                    r.MAE_logM_failures_replaced_by_metadata,
                    sd if stat == "per_seed_mean" else None,
                    5 if stat == "per_seed_mean" else None, None, f, stat)

    df = pd.DataFrame(rows).drop_duplicates(
        subset=["model_id", "eval_set", "metric", "severity", "statistic", "protocol"],
        keep="first")
    df.to_csv(OUT / "consolidated_results.csv", index=False)
    return df


def protocol_health():
    frames = []
    for f in ("A4_extended/training_runs.csv", "A17/variant_runs.csv",
              "A19/variant_runs.csv", "A7/training_runs_dedup.csv"):
        t = rd(f)
        if t is not None:
            t = t.copy(); t["source"] = f
            frames.append(t)
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    key = "arch" if "arch" in d.columns else "variant"
    rows = []
    for a, g in d.groupby(d.get("variant", d.get("arch")).fillna(d.get("arch"))):
        rows.append({"architecture": a, "n_runs": len(g),
                     "stopped_before_100": int((g.stop_epoch < 100).sum()),
                     "hit_cap": int(g.cap_hit.sum()),
                     "best_within_5_of_stop": int(((g.stop_epoch - g.best_epoch) <= 5).sum()),
                     "best_epoch_median": float(g.best_epoch.median()),
                     "best_epoch_p10": float(g.best_epoch.quantile(0.1)),
                     "best_epoch_p90": float(g.best_epoch.quantile(0.9))})
    h = pd.DataFrame(rows).sort_values("architecture")
    h.to_csv(OUT / "protocol_health.csv", index=False)
    return h



# ---------------------------------------------------------------------------
# Figure-input registry. Files the figures read that are not digest cells
# (per-spectrum scatters, sweeps, per-seed run tables). The v8 registry was
# hand-maintained and unscripted; the descriptions are carried over verbatim,
# the paths are resolved to the active dataset version, and the run ids are
# recomputed from the files actually on disk.
# ---------------------------------------------------------------------------
FIGURE_INPUTS = [
    ("A6/predicted_vs_reference_scatter.csv", "per-spectrum predicted vs reference omega0/Gamma/kappa, s=1", "fig1"),
    ("A6/logM_error_by_gamma_error.csv", "log M error by Gamma-error quintile", "fig1 inset"),
    ("A7/improvement_over_oracle.csv", "channel ablation: improvement over per-setting oracle", "fig5"),
    ("A7/rho_per_setting.csv", "channel ablation: within-condition rho per setting", "fig5"),
    ("A8/one_factor_sweeps.csv", "one-factor sweeps (resolution, counts)", "figS2"),
    ("A8/one_factor_background_shaped.csv", "one-factor sweep, shaped background", "figS2"),
    ("A14/extended_sweep_stress.csv", "extended severities s in {6,8,12}, stress base", "fig3"),
    ("A14/boundaries_threshold_sensitivity.csv", "ranking boundaries at rho 0.3/0.5/0.7", "fig3"),
    ("A17/variant_paired_vs_base.csv", "architecture variants vs family base", "figS1"),
    ("A3/summary_by_severity.csv", "fitting baselines by severity", "fig2"),
    ("A3/summary_extra_variants.csv", "extra fitting variants by severity", "fig2"),
    ("A9/occlusion_sensitivity.csv", "occlusion delta-MAE vs energy, with T-split", "figS3"),
    ("A1/conditional_spread_vs_conditions.csv", "conditional spread of log M vs T,c,E", "figS4"),
    ("A4/training_runs.csv", "v7-protocol training runs (protocol sensitivity)", "figS5"),
    ("A4_extended/training_runs.csv", "extended-protocol training runs", "figS5"),
    ("A4_control_buggy_bose/control_runs.csv", "Bose control runs", "figS6"),
    ("A3/feature_set_ablation.csv", "feature-set ablation at s=1", "figS7"),
    ("A14/replicate_vs_mc_conditional.csv", "replicate empirical vs Monte-Carlo conditional median", "figS8"),
    ("A18b/energy_tilt_sensitivity.csv", "energy-tilt sensitivity", "figS9"),
    ("A18b_tilt_trained/tilt_trained_beta_grid.csv", "tilt-trained CNN across beta", "figS9"),
    ("A3/fit_results_extra_variants.csv", "per-spectrum fitted omega0/Gamma", "fig7"),
    ("A4_extended/gate_and_improvement.csv", "gate and improvement with CIs", "fig8"),
    ("A19/variant_runs.csv", "A19 receptive-field variant runs (per-seed)", "fig8"),
]


def write_figure_inputs():
    """Emit figure_inputs.csv; report any entry whose file is not on disk."""
    rows, absent = [], []
    for rel, desc, used_by in FIGURE_INPUTS:
        p = _resolve(rel)
        if not p.exists():
            absent.append(str(p.relative_to(ROOT)))
            continue
        rows.append({"source_file": str(p.relative_to(ROOT)), "relative": rel,
                     "description": desc, "used_by": used_by,
                     "run_id": run_id_for(rel)})
    pd.DataFrame(rows).to_csv(OUT / "figure_inputs.csv", index=False)
    return rows, absent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    df = consolidated()
    print(f"consolidated: {len(df)} cells, {df.model_id.nunique()} models, "
          f"{df.eval_set.nunique()} evaluation sets")
    h = protocol_health()
    if h is not None:
        print("\n=== protocol health ===")
        print(h.to_string(index=False, float_format=lambda x: f"{x:.0f}"))

    # copy through the tables that already exist in final form
    copies = {"three_boundaries.csv": "A14/boundaries_threshold_sensitivity.csv",
              "comparison_robustness.csv": "A12_comparison_robustness.csv",
              "channel_ablation_improvement.csv": "A7/improvement_over_oracle.csv",
              "channel_ablation_rho.csv": "A7/rho_per_setting.csv",
              "variance_decomposition.csv": "A7/variance_decomposition.csv",
              "mps_reproducibility.csv": "A7/mps_reproducibility_pairs.csv"}
    for dst, src in copies.items():
        t = rd(src)
        if t is not None:
            t.to_csv(OUT / dst, index=False)

    fig_rows, fig_absent = write_figure_inputs()
    print(f"figure inputs registered: {len(fig_rows)}")
    for a in fig_absent:
        MISSING.append(f"{a} (figure input)")
        print(f"  MISSING figure input {a}")

    # assertions
    problems = []
    for mid, cls, metric in {tuple(x) for x in SKIPPED}:
        print(f"  not applicable, dropped: {metric} for {mid} ({cls})")
    bad = [(r.model_id, r.metric) for r in df.itertuples()
           if not applicable(r.model_id, r.metric)]
    if bad:
        problems.append(f"metric applicability: {len(bad)} rows outside their declared "
                        f"model classes, e.g. {bad[:3]}")
    key = ["model_id", "eval_set", "metric", "severity", "statistic", "protocol"]
    dup = df.groupby(key, dropna=False).size()
    if (dup > 1).any():
        problems.append(f"uniqueness: {int((dup > 1).sum())} duplicated key tuples")
    if df.run_id.isna().any():
        problems.append(f"run_id null in {int(df.run_id.isna().sum())} rows")
    for f in OUT.glob("*.csv"):
        t = pd.read_csv(f)
        if "run_id" in t.columns:
            ids = t.run_id.dropna().unique()
            if len(ids) > 1 and f.name not in ("consolidated_results.csv",
                                               "figure_inputs.csv"):
                problems.append(f"{f.name}: {len(ids)} run_ids")
    for p in df.source_file.dropna().unique():
        if not (ROOT / p).exists():
            problems.append(f"missing source: {p}")

    # Round-36 producing-script rule: every results file the digest cites must
    # name the script and repo SHA that produced it, in its directory's
    # producing_scripts.json (written by backfill_sidecars.py).
    for p in df.source_file.dropna().unique():
        f = ROOT / p
        if not f.exists():
            continue
        reg = f.parent / "producing_scripts.json"
        if not reg.exists():
            problems.append(f"no producing_scripts.json for {p}")
            continue
        try:
            files = json.loads(reg.read_text()).get("files", {})
        except Exception as e:
            problems.append(f"unreadable producing_scripts.json for {p}: {e}")
            continue
        rec = files.get(f.name)
        if rec is None:
            problems.append(f"unattributed results file: {p}")
        elif not rec.get("git_commit"):
            problems.append(f"no producing git SHA for {p} (script: {rec.get('script')})")
    (OUT / "assertions.json").write_text(json.dumps(
        {"missing_inputs": MISSING, "problems": problems,
         "date_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"\nmissing inputs: {len(MISSING)}")
    for m in MISSING:
        print(f"  MISSING {m}")
    print(f"assertion problems: {len(problems)}")
    for p in problems:
        print(f"  PROBLEM {p}")


if __name__ == "__main__":
    main()
