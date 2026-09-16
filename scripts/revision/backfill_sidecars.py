"""Producing-script provenance for every file under results/revision (round 36, item 2.4).

Writes `producing_scripts.json` into each result directory: for every non-checkpoint
file it records the script that produced it and the repo SHA at the time of writing.
Files whose producer no longer exists are marked ad hoc and superseded, rather than
left blank -- an unattributed results file is the thing A12 now refuses.

Run with --check to report coverage without writing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import dataset_paths as DS

ROOT = DS.ROOT
REV = DS.REV

# filename -> producing script, taken from each script's write targets.
BY_FILE = {
    "oracle_floor.json": "a1_oracle_floor.py",
    "per_tuple_conditional_logM.csv": "a1_oracle_floor.py",
    "conditional_spread_vs_conditions.csv": "a1_oracle_floor.py",
    "metadata_family_metrics.csv": "a2_metadata_family.py",
    "predictions_seedavg.csv": "a2_metadata_family.py",
    "metadata_family_metrics_v8_newsets.csv": "a2b_new_sets.py",
    "predictions_seedavg_v8_newsets.csv": "a2b_new_sets.py",
    "fit_results_per_spectrum.csv": "a3_fitting_baselines.py",
    "summary_by_severity.csv": "a3_fitting_baselines.py",
    "feature_set_ablation.csv": "a3_fitting_baselines.py",
    "feature_gbm_predictions.csv": "a3_fitting_baselines.py",
    "fit_results_extra_variants.csv": "a3b_extra_baselines.py",
    "summary_extra_variants.csv": "a3b_extra_baselines.py",
    "st_predictions_stress_base.csv": "a3b_extra_baselines.py",
    "paired_comparison_stress_base.csv": "a3c_paired_stats.py",
    "training_runs.csv": "a4_train_v8.py",
    "training_summary.csv": "a4_train_v8.py",
    "gate_and_improvement.csv": "a4_gate_eval.py",
    "pairwise_model_comparisons.csv": "a4_gate_eval.py",
    "val_predictions_seedavg.csv": "a4_gate_eval.py",
    "label_permutation.csv": "a5_label_permutation.py",
    "aux_head_metrics.csv": "a6_a10_eval.py",
    "logM_error_by_gamma_error.csv": "a6_a10_eval.py",
    "holdout_table.csv": "a6_a10_eval.py",
    "gamma_by_damping_regime.csv": "a6b_completeness.py",
    "tuned_cnn_params_by_severity.csv": "a6b_completeness.py",
    "predicted_vs_reference_scatter.csv": "a6b_completeness.py",
    "identifiability_kappa.csv": "a6c_identifiability.py",
    "figure1_reference_lines.csv": "a6c_identifiability.py",
    "parameter_recovery_vs_metadata_oracle.csv": "a6c_identifiability.py",
    "training_runs_dedup.csv": "a7_postprocess.py",
    "mps_reproducibility_pairs.csv": "a7_postprocess.py",
    "marginal_check.csv": "a7_postprocess.py",
    "cross_mode_correlation.csv": "a7_postprocess.py",
    "oracle_per_setting.csv": "a7_postprocess.py",
    "improvement_over_oracle.csv": "a7_postprocess.py",
    "rho_per_setting.csv": "a7_postprocess.py",
    "variance_decomposition.csv": "a7_postprocess.py",
    "dense_sweep_stress.csv": "a8_dense_sweep.py",
    "dense_sweep_replicate.csv": "a8_dense_sweep.py",
    "one_factor_sweeps.csv": "a8_dense_sweep.py",
    "value_crossover.csv": "a8_dense_sweep.py",
    "occlusion_sensitivity.csv": "a9_attribution.py",
    "region_shares.csv": "a9_attribution.py",
    "s12_central_peak_mask.csv": "a9_attribution.py",
    "replicate_vs_mc_conditional.csv": "a14_data_check.py",
    "replicate_vs_mc_summary.json": "a14_data_check.py",
    "two_boundaries.csv": "a14b_boundaries.py",
    "extended_sweep_stress.csv": "a14b_boundaries.py",
    "extended_rho_replicate.csv": "a14b_boundaries.py",
    "boundaries_threshold_sensitivity.csv": "a14b_boundaries.py",
    "rho_fitting_vs_learning_subsample.csv": "a14c_fitting_rho.py",
    "a5_shuffle_and_a14_rho.csv": "a5_a14_replicate.py",
    "a14_within_tuple_rho_by_target.csv": "a5_a14_replicate.py",
    "variant_runs.csv": "a17_arch_sweep.py",
    "variant_summary.csv": "a17_arch_sweep.py",
    "variant_paired_vs_base.csv": "a17_paired_stats.py",
    "best_of_family.json": "a17_paired_stats.py",
    "intensity_invariance.csv": "a18_intensity_invariance.py",
    "energy_tilt_sensitivity.csv": "a18b_energy_tilt.py",
    "tilt_asymmetry_cis.csv": "a18c_tilt_cis.py",
    "tilt_rho_cis.csv": "a18c_tilt_cis.py",
    "confirmation_heldout.csv": "a19_confirm.py",
    "tilt_trained_summary.csv": "a19_confirm.py",
    "tilt_trained_beta_grid.csv": "a19_confirm.py",
    "consolidated_results.csv": "a12_digest.py",
    "figure_inputs.csv": "a12_digest.py",
    "assertions.json": "a12_digest.py",
    "protocol_health.csv": "a12_digest.py",
    "three_boundaries.csv": "a12_digest.py",
    "comparison_robustness.csv": "a12_digest.py",
    "channel_ablation_improvement.csv": "a12_digest.py",
    "channel_ablation_rho.csv": "a12_digest.py",
    "mps_reproducibility.csv": "a12_digest.py",
    "metadata_family_per_seed.csv": "a2_metadata_family.py",
    "metadata_family_per_seed_v8_newsets.csv": "a2b_new_sets.py",
    "within_condition_shuffle.csv": "a5_a14_replicate.py",
    "identifiability_kappa_robust.csv": "a6c_identifiability.py",
    "one_factor_background_shaped.csv": "a8_dense_sweep.py",
    "central_peak_mask_severities.csv": "a9_attribution.py",
    "control_runs.csv": "a4_control_buggy_bose.py",
    "control_summary.csv": "a4_control_buggy_bose.py",
}
AD_HOC = "ad hoc (v8), superseded by v9"
SCRIPT_DIR = Path(__file__).resolve().parent


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def producer(name: str) -> str | None:
    if name in BY_FILE:
        return BY_FILE[name]
    if name.endswith("_history.csv"):
        return "a4_train_v8.py"        # per-epoch history, written by the trainer
    return None


def is_sidecar(name: str) -> bool:
    """Sidecars describe their own run; they are not results needing attribution."""
    return name.startswith("run_sidecar") or name == "producing_scripts.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report coverage, write nothing")
    args = ap.parse_args()

    sha = git_sha()
    total = attributed = adhoc = 0
    for d in sorted(p for p in REV.iterdir() if p.is_dir()):
        files = [p for p in sorted(d.rglob("*"))
                 if p.is_file() and "checkpoints" not in p.parts
                 and p.suffix in (".csv", ".json") and not is_sidecar(p.name)]
        if not files:
            continue
        entry = {}
        for f in files:
            s = producer(f.name)
            if s is None:
                entry[str(f.relative_to(d))] = {"script": AD_HOC}
                adhoc += 1
            else:
                entry[str(f.relative_to(d))] = {
                    "script": str((SCRIPT_DIR / s).relative_to(ROOT)),
                    "git_commit": sha,
                }
                attributed += 1
            total += 1
        if not args.check:
            (d / "producing_scripts.json").write_text(json.dumps(
                {"directory": str(d.relative_to(ROOT)),
                 "dataset_version": DS.VERSION,
                 "generated_utc": datetime.now(timezone.utc).isoformat(),
                 "files": entry}, indent=2))
    verb = "would attribute" if args.check else "attributed"
    print(f"  {verb} {attributed}/{total} files; {adhoc} marked {AD_HOC!r}")


if __name__ == "__main__":
    main()
