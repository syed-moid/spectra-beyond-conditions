#!/usr/bin/env python3
"""Ordered, resumable, deadline-aware driver for the v9 rerun (round-32 step 3).

Run it in sessions. It records what finished, stops cleanly at a wall-clock
deadline, and picks up where it left off next time:

    # this evening, stop by 08:00 tomorrow
    python scripts/revision/run_pipeline.py --until 08:00 --caffeinate

    # what is left
    python scripts/revision/run_pipeline.py --status

    # continue tomorrow evening
    python scripts/revision/run_pipeline.py --until 08:00 --caffeinate

Two kinds of stage:

  RESUMABLE  the underlying script keeps its own per-cell record and skips work
             already on disk. Safe to kill at any instant; restarting continues
             from the last completed cell. These may be started even when less
             time remains than they need.
  ATOMIC     the script writes its outputs in one pass and does not deduplicate
             on re-entry. A partial run is discarded: the driver deletes that
             stage's outputs before every attempt, so a restart is clean rather
             than double-counted. These are only started when the estimated
             duration fits inside the remaining time.

Failure (OOM, crash, non-zero exit) stops the pipeline rather than cascading
into stages that would read half-written inputs. The failed stage is retried on
the next invocation.

State: .pipeline_state_<version>.json   Logs: logs/
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import dataset_paths as DS

ROOT = DS.ROOT
SCRIPTS = Path(__file__).resolve().parent
# State is scoped to the dataset version. A single shared file let a v9 run's
# "done" marks be read as v10 progress, which would silently skip every stage.
LEGACY_STATE = ROOT / ".pipeline_state.json"
STATE = ROOT / f".pipeline_state_{DS.VERSION}.json"
LOGS = ROOT / "logs"
FIGDIR = ROOT / "figures"
MSDIR = Path(os.environ.get("SBC_MANUSCRIPT_DIR", ROOT / "manuscript"))
LOCK = ROOT / ".locks" / "pipeline.lock"
PY = sys.executable

RESUMABLE, ATOMIC = "resumable", "atomic"
GRACE_SEC = 120          # SIGTERM -> SIGKILL window at the deadline


@dataclass
class Stage:
    name: str
    argv: list            # built lazily: callables are resolved at run time
    est_min: float
    mode: str = ATOMIC
    wipe: list = field(default_factory=list)   # paths cleared before an attempt
    outputs: list = field(default_factory=list)  # declared; defaults to `wipe`
    inputs: list = field(default_factory=list)   # mtime sources for the freshness check
    note: str = ""
    blocked: str = ""     # non-empty => cannot run yet, and why

    def cmd(self):
        return [str(x() if callable(x) else x) for x in self.argv]

    def declared_outputs(self):
        return [Path(x() if callable(x) else x) for x in (self.outputs or self.wipe)]

    def declared_inputs(self):
        return [Path(x() if callable(x) else x) for x in self.inputs]


def s(name):
    return str(SCRIPTS / f"{name}.py")


def rel(p):
    """Repo-relative display path, falling back to the absolute one."""
    try:
        return Path(p).relative_to(ROOT)
    except ValueError:
        return Path(p)


def rev(task, *parts):
    return lambda: DS.rev(task).joinpath(*parts)


# --------------------------------------------------------------------------
# The pipeline, in dependency order (round-32 step 3, continued from A18b).
# est_min values are measured v8 durations scaled by the 1.42x v9 factor
# observed on A4; the driver records actuals and reports drift.
# --------------------------------------------------------------------------
def build_stages():
    A7_SETTINGS = ("xi1_only", "xi2_only", "both", "alpha_per_mode")
    A7_DS_SEEDS = (0, 1, 2)

    # a17_paired_stats and a19_confirm compare against the A4 "family base"
    # checkpoints. Those exist on v9 but were never trained on v10, so on v10
    # both stages die with FileNotFoundError after the driver has spent a slot
    # on them. Declare the dependency instead of discovering it at run time.
    _a4e = DS.rev("A4_extended") / "checkpoints"
    _no_base = (not _a4e.exists()) or not any(_a4e.glob("*.pt"))
    BASE_MISSING = (
        f"needs the A4 family-base checkpoints at {rel(_a4e)}, which do not "
        f"exist for {DS.VERSION}; they are 29 training runs that this revision "
        f"did not do. Train them, or keep the {DS.VERSION} result on the A21 "
        f"checkpoints instead." if _no_base else "")

    st = [
        Stage("a18b_tilt_train", [
            PY, s("a4_train_v8"), "--outdir", rev("A18b_tilt_trained"),
            "--stopping", "extended", "--train-tilt", "0.2", "--tag-suffix", "_tilt",
            "--runs", "cnn:v7:42", "cnn:v7:43", "cnn:v7:44"],
            est_min=6, mode=RESUMABLE,
            outputs=[rev("A18b_tilt_trained", "training_runs.csv")],
            inputs=[lambda: DS.FULL],
            note="seed 42 already done; resume skips it"),

        # Round 47 item 5: the A4 family-base grid on the current dataset. Until
        # this exists on v10, a17_paired_stats had to borrow bases and a19_confirm
        # could not run at all. PLAN in a4_train_v8.py is exactly these 29 runs;
        # the script appends per cell and skips what it has.
        Stage("a4_extended_train", [
            PY, s("a4_train_v8"), "--stopping", "extended",
            "--outdir", rev("A4_extended")],
            est_min=160, mode=RESUMABLE,
            outputs=[rev("A4_extended", "training_runs.csv")],
            inputs=[lambda: DS.FULL],
            note="29 runs: 5a/5b/fusion/cnn on the v7 severity protocol plus the "
                 "fixed1 and low protocol arms; unblocks a19_confirm, figS1, fig8"),

        # figS5 contrasts the retired v7 stopping rule with the extended one. The
        # extended arm is on v10; the v7 arm existed only on v8, so the figure was
        # comparing two protocols across two generator versions at once. This
        # trains the v7 arm on v10 so the comparison isolates the protocol.
        Stage("a4_v7_protocol_train", [
            PY, s("a4_train_v8"), "--stopping", "v7", "--outdir", rev("A4")],
            est_min=140, mode=RESUMABLE,
            outputs=[rev("A4", "training_runs.csv")],
            inputs=[lambda: DS.FULL],
            note="the retired stopping rule on v10, so figS5 compares protocols "
                 "within one generator version"),

        Stage("a4_gate_eval", [PY, s("a4_gate_eval")], est_min=3, mode=ATOMIC,
              wipe=[rev("A4_extended", "gate_and_improvement.csv"),
                    rev("A4_extended", "pairwise_model_comparisons.csv"),
                    rev("A4_extended", "val_predictions_seedavg.csv"),
                    rev("A4_extended", "run_sidecar_gate.json")],
              inputs=[rev("A4_extended", "training_runs.csv")],
              note="was never a declared stage; its outputs kept a stale v8 label "
                   "until round 41"),

        Stage("a5_permutation", [PY, s("a5_label_permutation")],
              est_min=20, mode=ATOMIC,
              wipe=[rev("A5", "label_permutation.csv"),
                    rev("A5", "run_sidecar_label_perm.json"),
                    rev("A5", "checkpoints_label_perm")]),

        # Round 47 item 5: the A3 severity sweep, now including the corrected
        # prior-regularized fit alongside the unregularized pair. a3b, a3c, a6b
        # and the figures all read its outputs.
        Stage("a3_sweep", [PY, s("a3_fitting_baselines")], est_min=45, mode=ATOMIC,
              wipe=[rev("A3", "fit_results_per_spectrum.csv"),
                    rev("A3", "feature_gbm_predictions.csv"),
                    rev("A3", "summary_by_severity.csv"),
                    rev("A3", "feature_set_ablation.csv")],
              inputs=[lambda: DS.FULL],
              note="dho_matched + dho_matched_prior + dho_single over 5 severities "
                   "x 500 stress spectra, then the feature GBM"),

        Stage("a3b_extra_baselines", [PY, s("a3b_extra_baselines")],
              est_min=180, mode=ATOMIC,
              wipe=[rev("A3", "fit_results_extra_variants.csv"),
                    rev("A3", "summary_extra_variants.csv"),
                    rev("A3", "st_predictions_stress_base.csv"),
                    rev("A3", "run_sidecar_extra.json")],
              note="A3_v9 is missing these; a6b/a6c/a12 all read them"),

        Stage("a3c_paired_stats", [PY, s("a3c_paired_stats")],
              est_min=10, mode=ATOMIC,
              wipe=[rev("A3", "paired_comparison_stress_base.csv"),
                    rev("A3", "run_sidecar_paired.json")]),
    ]

    for setting in A7_SETTINGS:
        for ds_seed in A7_DS_SEEDS:
            st.append(Stage(
                f"a7_{setting}_ds{ds_seed}",
                [PY, s("a7_train_ablation"), "--setting", setting,
                 "--dataset-seed", str(ds_seed)],
                est_min=38, mode=RESUMABLE,
                outputs=[rev("A7", "training_runs.csv")],
                inputs=[(lambda st=setting, sd=ds_seed: DS.ablation(st, sd))]))

    st += [
        Stage("a7_postprocess", [PY, s("a7_postprocess")], est_min=25, mode=ATOMIC,
              wipe=[rev("A7", "training_runs_dedup.csv"),
                    rev("A7", "mps_reproducibility_pairs.csv"),
                    rev("A7", "marginal_check.csv"),
                    rev("A7", "cross_mode_correlation.csv"),
                    rev("A7", "oracle_per_setting.csv"),
                    rev("A7", "improvement_over_oracle.csv"),
                    rev("A7", "rho_per_setting.csv"),
                    rev("A7", "variance_decomposition.csv"),
                    rev("A7", "run_sidecar_postprocess.json")],
              note="reconstructed round 36; four of its eight outputs reproduce "
                   "the v8 files exactly (see reports/A7_postprocess.md)"),

        Stage("a6_a10_eval", [PY, s("a6_a10_eval")], est_min=70, mode=ATOMIC,
              wipe=[rev("A6", "aux_head_metrics.csv"),
                    rev("A6", "logM_error_by_gamma_error.csv"),
                    rev("A6", "run_sidecar.json"),
                    rev("A10", "holdout_table.csv"),
                    rev("A10", "run_sidecar.json")]),

        Stage("a6b_completeness", [PY, s("a6b_completeness")], est_min=12, mode=ATOMIC,
              wipe=[rev("A6", "gamma_by_damping_regime.csv"),
                    rev("A6", "predicted_vs_reference_scatter.csv"),
                    rev("A6", "tuned_cnn_params_by_severity.csv"),
                    rev("A6", "run_sidecar_completeness.json")]),

        Stage("a6c_identifiability", [PY, s("a6c_identifiability")], est_min=12, mode=ATOMIC,
              wipe=[rev("A6", "figure1_reference_lines.csv"),
                    rev("A6", "identifiability_kappa.csv"),
                    rev("A6", "parameter_recovery_vs_metadata_oracle.csv"),
                    rev("A6", "run_sidecar_identifiability.json")]),

        Stage("a8_dense_sweep", [PY, s("a8_dense_sweep")], est_min=36, mode=ATOMIC,
              wipe=[rev("A8", "one_factor_background_shaped.csv"),
                    rev("A8", "dense_sweep_stress.csv"),
                    rev("A8", "dense_sweep_replicate.csv"),
                    rev("A8", "one_factor_sweeps.csv"),
                    rev("A8", "value_crossover.csv"),
                    rev("A8", "run_sidecar.json")]),

        Stage("a9_attribution", [PY, s("a9_attribution")], est_min=420, mode=RESUMABLE,
              outputs=[rev("A9", "occlusion_sensitivity.csv"),
                       rev("A9", "region_shares.csv"),
                       rev("A9", "s12_central_peak_mask.csv")],
              inputs=[lambda: DS.FULL],
              note="longest stage; appends per cell and skips what it has"),

        Stage("a14b_boundaries", [PY, s("a14b_boundaries")], est_min=90, mode=ATOMIC,
              wipe=[rev("A14", "two_boundaries.csv"),
                    rev("A14", "extended_sweep_stress.csv"),
                    rev("A14", "extended_rho_replicate.csv"),
                    rev("A14", "rho_fitting_vs_learning_subsample.csv"),
                    rev("A14", "run_sidecar_boundaries.json")]),

        # Produces the threshold-sensitivity table fig3 reads. Like a3_sweep it was
        # never a declared stage, so the figure input it feeds went missing on a
        # clean version even though a14b had run.
        Stage("a14d_boundary_thresholds", [PY, s("a14d_boundary_thresholds")],
              est_min=5, mode=ATOMIC,
              wipe=[rev("A14", "boundaries_threshold_sensitivity.csv")],
              inputs=[rev("A14", "extended_rho_replicate.csv"),
                      rev("A14", "two_boundaries.csv")]),

        Stage("a14c_fitting_rho", [PY, s("a14c_fitting_rho")], est_min=60, mode=ATOMIC,
              wipe=[rev("A14", "rho_fitting_vs_learning_subsample.csv"),
                    rev("A14", "run_sidecar_fitting_rho.json"),
                    rev("A14", "run_sidecar_fitting_subsample.json")],
              note="appends without dedup, so the wipe is load-bearing; must "
                   "follow a14b, which writes the same filename"),

        Stage("a5_a14_replicate", [PY, s("a5_a14_replicate")], est_min=20, mode=ATOMIC,
              wipe=[rev("A14", "a5_shuffle_and_a14_rho.csv"),
                    rev("A14", "a14_within_tuple_rho_by_target.csv"),
                    rev("A14", "run_sidecar_a5_a14.json")]),

        Stage("a14_data_check", [PY, s("a14_data_check")], est_min=8, mode=ATOMIC,
              wipe=[rev("A14", "replicate_vs_mc_conditional.csv"),
                    rev("A14", "replicate_vs_mc_summary.json"),
                    rev("A14", "run_sidecar.json")]),

        Stage("a18_intensity_invariance", [PY, s("a18_intensity_invariance")],
              est_min=8, mode=ATOMIC,
              wipe=[rev("A18", "intensity_invariance.csv"), rev("A18", "run_sidecar.json")]),

        Stage("a18b_energy_tilt", [PY, s("a18b_energy_tilt")], est_min=30, mode=ATOMIC,
              wipe=[rev("A18b", "energy_tilt_sensitivity.csv"), rev("A18b", "run_sidecar.json")]),

        Stage("a18c_tilt_cis", [PY, s("a18c_tilt_cis")], est_min=20, mode=ATOMIC,
              wipe=[rev("A18b", "tilt_asymmetry_cis.csv"),
                    rev("A18b", "tilt_rho_cis.csv"),
                    rev("A18b", "run_sidecar_cis.json")]),

        Stage("a19_confirm", [PY, s("a19_confirm")], est_min=60, mode=ATOMIC,
              blocked=BASE_MISSING,
              wipe=[rev("A19", "confirmation_heldout.csv"),
                    rev("A18b_tilt_trained", "tilt_trained_summary.csv"),
                    rev("A18b_tilt_trained", "tilt_trained_beta_grid.csv"),
                    rev("A18b_tilt_trained", "run_sidecar_eval.json")]),

        # The A17 architecture sweep trains its own variants and was never a
        # declared stage, so a17_paired_stats had an undeclared dependency: on a
        # fresh dataset version it would read whatever variant_runs.csv happened
        # to be on disk. It appends per cell and skips what it has.
        Stage("a17_arch_sweep", [PY, s("a17_arch_sweep")], est_min=200, mode=RESUMABLE,
              outputs=[rev("A17", "variant_runs.csv")],
              inputs=[lambda: DS.FULL],
              note="21 variant runs (patch and kernel width); resumable per cell"),

        Stage("a17_paired_stats", [PY, s("a17_paired_stats")], est_min=10, mode=ATOMIC,
              inputs=[rev("A17", "variant_runs.csv")],
              note="round 46: family bases now come from the sweep's own cnn_base/tf_base "
                   "checkpoints, so this no longer needs A4_extended",
              wipe=[rev("A17", "variant_paired_vs_base.csv"),
                    rev("A17", "best_of_family.json")]),

        # backfill writes producing_scripts.json into every result directory that
        # EXISTS when it runs. A12 is created by the digest, which runs after this,
        # so A12's file cannot be a declared output here -- declaring it made the
        # stage fail its own freshness check on a clean version. The digest needs
        # the attribution for the directories it CITES, which this pass covers; a
        # second pass after the digest attributes A12 itself.
        Stage("backfill_sidecars", [PY, s("backfill_sidecars")], est_min=1, mode=ATOMIC,
              outputs=[rev("A7", "producing_scripts.json"),
                       rev("A14", "producing_scripts.json")],
              note="attributes every results file to its producing script; the "
                   "digest asserts on this"),

        Stage("a12_digest", [PY, s("a12_digest"), "--force"], est_min=30, mode=ATOMIC,
              # Wipe the digest's own outputs, not the whole directory: the
              # attribution file backfill_sidecars writes lives there too, and
              # removing the directory destroyed it every run.
              wipe=[rev("A12", "consolidated_results.csv"),
                    rev("A12", "figure_inputs.csv"),
                    rev("A12", "assertions.json"),
                    rev("A12", "protocol_health.csv"),
                    rev("A12", "comparison_robustness.csv"),
                    rev("A12", "three_boundaries.csv"),
                    rev("A12", "channel_ablation_improvement.csv"),
                    rev("A12", "channel_ablation_rho.csv"),
                    rev("A12", "variance_decomposition.csv"),
                    rev("A12", "mps_reproducibility.csv")],
              outputs=[rev("A12", "consolidated_results.csv"),
                       rev("A12", "figure_inputs.csv"),
                       rev("A12", "assertions.json")]),

        # ------------------------------------------------------------------
        # Round 45: evaluation-only recomputes on v10, then the manuscript
        # assembly. None of these train; all declare their outputs so a kill
        # mid-stage is retried cleanly rather than half-counted.
        # ------------------------------------------------------------------
        Stage("a24_recompute", [PY, s("a24_recompute_v10")], est_min=32, mode=ATOMIC,
              wipe=[rev("A24", "parameter_recovery_v10.csv"),
                    rev("A24", "ranking_by_severity_v10.csv"),
                    rev("A24", "run_sidecar.json")],
              inputs=[lambda: DS.TEST, lambda: DS.REPLICATE_TEST,
                      rev("A21", "training_runs.csv")],
              note="kappa_slow, 200+200 shuffle nulls, pairwise-accuracy "
                   "boundaries over 11 severities, conditions-only comparator"),

        Stage("a25_network_vs_fit", [PY, s("a25_network_vs_fit")], est_min=3, mode=ATOMIC,
              wipe=[rev("A23", "network_vs_fit_test.csv"),
                    rev("A23", "network_runs_test.csv"),
                    rev("A23", "prior_gap_fractions.json"),
                    rev("A23", "run_sidecar_network.json")],
              inputs=[lambda: DS.TEST, rev("A23", "fit_summary_test.csv"),
                      rev("A21", "training_runs.csv")],
              note="networks on the identical 600 spectra A23 fits, so the "
                   "§3.6 table is readable from one file"),

        Stage("a26_holdouts", [PY, s("a26_holdouts_v10")], est_min=6, mode=ATOMIC,
              wipe=[rev("A10", "holdout_table_v10.csv"),
                    rev("A10", "holdout_runs_v10.csv"),
                    rev("A10", "run_sidecar_v10.json")],
              inputs=[lambda: DS.FULL, lambda: DS.HOLDOUT_C_EXTRAP,
                      rev("A21", "training_runs.csv")],
              note="nine runs per model, both metadata baselines in every row"),

        Stage("a27_feature_ablation", [PY, s("a27_feature_ablation_v10")],
              est_min=10, mode=ATOMIC,
              wipe=[rev("A3", "feature_set_ablation.csv"),
                    rev("A3", "run_sidecar_feature_ablation.json")],
              inputs=[lambda: DS.FULL],
              note="DRAFT_v8_2 cited A3_v10/feature_set_ablation.csv, which did "
                   "not exist; the numbers came from A3_v9"),

        Stage("a28_fill_tables", [PY, s("a28_fill_a24_tables")], est_min=1, mode=ATOMIC,
              outputs=[MSDIR / "DRAFT_v8_9_main.md",
                       MSDIR / "DRAFT_v8_9_supplement.md"],
              inputs=[rev("A24", "parameter_recovery_v10.csv"),
                      rev("A24", "ranking_by_severity_v10.csv")],
              note="renders the five A24 tables into the manuscript placeholders; "
                   "idempotent, replaces between marker and closing marker"),

        Stage("main_text_provenance", [PY, s("a12_main_text_provenance")],
              est_min=1, mode=ATOMIC,
              outputs=[],
              inputs=[MSDIR / "DRAFT_v8_9_main.md"],
              note="assertion: no main-text number may cite a pre-v10 run. "
                   "Exit status is the assertion; it fails the stage."),

        Stage("backfill_sidecars_post", [PY, s("backfill_sidecars")], est_min=1, mode=ATOMIC,
              outputs=[rev("A12", "producing_scripts.json")],
              inputs=[rev("A12", "consolidated_results.csv")],
              note="second pass, so the digest's own directory is attributed too"),

        Stage("figures", [PY, s("generate_figures_v10")],
              outputs=[FIGDIR / "fig1_parameter_recovery.pdf",
                       FIGDIR / "fig2_recoverability_map.pdf",
                       FIGDIR / "fig3_two_boundaries.pdf",
                       FIGDIR / "fig4_holdout_heatmap.pdf",
                       FIGDIR / "fig5_channel_ablation.pdf",
                       FIGDIR / "fig6_example_spectra.pdf",
                       FIGDIR / "fig7_fit_overlay.pdf",
                       FIGDIR / "fig8_model_family_bars.pdf",
                       FIGDIR / "figS1_receptive_field.pdf",
                       FIGDIR / "figS2_one_factor.pdf",
                       FIGDIR / "figS3_occlusion.pdf",
                       FIGDIR / "figS4_conditional_spread.pdf",
                       FIGDIR / "figS5_protocol_sensitivity.pdf",
                       FIGDIR / "figS6_bose_control.pdf",
                       FIGDIR / "figS7_feature_ablation.pdf",
                       FIGDIR / "figS8_replicate_vs_mc.pdf",
                       FIGDIR / "figS9_tilt_sensitivity.pdf"],
              inputs=[lambda: DS.rev("A12") / "consolidated_results.csv"],
              est_min=60, mode=ATOMIC,
              note="all 17 declared; the freshness check fails the stage if any "
                   "is missing or left over from an earlier run"),
    ]
    return st


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------
def load_state():
    """Per-version state, with a one-time migration of the old shared file.

    Refuses to run against state written for a different dataset version: a
    stage marked done on v9 has produced nothing on v10, and treating it as
    complete would skip real work and leave v9 outputs wearing a v10 label.
    """
    if not STATE.exists() and LEGACY_STATE.exists():
        legacy = json.loads(LEGACY_STATE.read_text())
        lv = legacy.get("version", "v9")
        dest = ROOT / f".pipeline_state_{lv}.json"
        if not dest.exists():
            dest.write_text(json.dumps(legacy, indent=2))
        LEGACY_STATE.unlink()
        print(f"migrated {LEGACY_STATE.name} -> {dest.name} (version {lv})")
    if STATE.exists():
        st = json.loads(STATE.read_text())
        sv = st.get("version", DS.VERSION)
        if sv != DS.VERSION:
            raise SystemExit(
                f"state file {STATE.name} was written for {sv}, but "
                f"INS_DATASET_VERSION is {DS.VERSION}. Refusing to run: a stage "
                f"marked done on {sv} has produced nothing on {DS.VERSION}. "
                f"Move or delete {rel(STATE)} to start a {DS.VERSION} run.")
        return st
    return {"version": DS.VERSION, "stages": {}}


def save_state(st):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2))
    tmp.replace(STATE)


def rec(st, name):
    return st["stages"].setdefault(name, {"status": "pending", "attempts": 0})


# --------------------------------------------------------------------------
# deadline
# --------------------------------------------------------------------------
def parse_deadline(text):
    """'08:00' -> next occurrence; '2026-09-10T08:00' -> that instant; '9h' -> from now."""
    if not text:
        return None
    now = datetime.now()
    t = text.strip()
    if t.endswith(("h", "m")) and t[:-1].replace(".", "").isdigit():
        n = float(t[:-1])
        return now + timedelta(hours=n) if t.endswith("h") else now + timedelta(minutes=n)
    if "T" in t or " " in t.strip():
        return datetime.fromisoformat(t.replace(" ", "T"))
    hh, _, mm = t.partition(":")
    d = now.replace(hour=int(hh), minute=int(mm or 0), second=0, microsecond=0)
    return d if d > now else d + timedelta(days=1)


def hm(minutes):
    minutes = max(0, int(round(minutes)))
    return f"{minutes // 60}h{minutes % 60:02d}m"


# --------------------------------------------------------------------------
# lock
# --------------------------------------------------------------------------
def take_lock():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            pid = int(LOCK.read_text().split()[0])
            os.kill(pid, 0)
            print(f"another pipeline is running (pid {pid}); refusing to start")
            return False
        except (ProcessLookupError, ValueError, IndexError):
            print(f"clearing stale lock {LOCK.name}")
    LOCK.write_text(f"{os.getpid()} {datetime.now().isoformat()}\n")
    return True


def free_lock():
    try:
        LOCK.unlink()
    except FileNotFoundError:
        pass


# --------------------------------------------------------------------------
# running one stage
# --------------------------------------------------------------------------
def clear(paths):
    import shutil
    for p in paths:
        p = Path(p() if callable(p) else p)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()


def freshness(stage, started_at):
    """Did the stage actually produce what it declared, and is it newer than its inputs?

    Two failure modes this catches that an exit code does not: a stage that
    writes nothing and returns 0 (the figures stage did exactly this), and a
    stage whose outputs are left over from an earlier dataset version.
    Returns (ok, [problems]).
    """
    outs = stage.declared_outputs()
    if not outs:
        return True, []
    problems, newest_in = [], None
    for p in stage.declared_inputs():
        if p.exists():
            t = p.stat().st_mtime
            newest_in = t if newest_in is None else max(newest_in, t)
    missing = [p for p in outs if not p.exists()]
    if missing:
        problems.append(f"{len(missing)}/{len(outs)} declared outputs absent: "
                        + ", ".join(p.name for p in missing[:4]))
    for p in outs:
        if not p.exists():
            continue
        mt = p.stat().st_mtime
        if mt < started_at - 1:
            problems.append(f"{p.name} not rewritten by this run")
        elif newest_in is not None and mt < newest_in:
            problems.append(f"{p.name} older than its inputs")
    return not problems, problems


def run_stage(stage, state, deadline, dry):
    r = rec(state, stage.name)
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{stage.name}.log"

    if stage.mode == ATOMIC and stage.wipe:
        if dry:
            print(f"    would clear {len(stage.wipe)} output path(s)")
        else:
            clear(stage.wipe)
    if not dry:
        # A stage's result directory may not exist yet on a fresh dataset
        # version. Create the parent of every declared output so a stage
        # cannot die at its final write after doing all the work.
        for w in stage.wipe:
            Path(w() if callable(w) else w).parent.mkdir(parents=True, exist_ok=True)

    cmd = stage.cmd()
    if dry:
        print(f"    would run: {' '.join(Path(c).name if '/' in c else c for c in cmd)}")
        return "done"

    r["attempts"] += 1
    r["status"] = "running"
    r["started"] = datetime.now().isoformat(timespec="seconds")
    save_state(state)

    t0 = time.time()
    with log.open("a") as fh:
        fh.write(f"\n{'='*70}\n{datetime.now().isoformat(timespec='seconds')}  "
                 f"attempt {r['attempts']}\n{' '.join(cmd)}\n{'='*70}\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
        killed = False
        while True:
            try:
                proc.wait(timeout=15)
                break
            except subprocess.TimeoutExpired:
                if deadline and datetime.now() >= deadline and not killed:
                    print(f"    deadline reached — stopping {stage.name} "
                          f"(SIGTERM, {GRACE_SEC}s grace)")
                    fh.write(f"\n[driver] deadline reached, SIGTERM\n"); fh.flush()
                    killed = True
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        proc.wait(timeout=GRACE_SEC)
                    except subprocess.TimeoutExpired:
                        fh.write("[driver] grace expired, SIGKILL\n"); fh.flush()
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        proc.wait()
                    break

    secs = time.time() - t0
    r["seconds"] = round(secs, 1)
    r["ended"] = datetime.now().isoformat(timespec="seconds")
    r["exit_code"] = proc.returncode

    if killed:
        r["status"] = "interrupted"
        outcome = "interrupted"
    elif proc.returncode == 0:
        ok, problems = freshness(stage, t0)
        r["freshness"] = "ok" if ok else "stale"
        r["freshness_problems"] = problems
        if ok:
            r["status"] = "done"
            outcome = "done"
        else:
            r["status"] = "failed"
            outcome = "failed"
            print("    freshness check failed:")
            for pb in problems:
                print(f"      - {pb}")
    else:
        r["status"] = "failed"
        outcome = "failed"
    save_state(state)

    drift = ""
    if outcome == "done" and stage.est_min:
        drift = f"  (est {hm(stage.est_min)}, {secs/60/stage.est_min:.2f}x)"
    print(f"    {outcome} in {hm(secs/60)}{drift}   log: {rel(log)}")
    return outcome


# --------------------------------------------------------------------------
def static_fresh(stage):
    """Outputs all present and none older than any declared input.

    The "rewritten by this run" arm of freshness() needs a run, so it is not
    part of this check; this is the question --adopt-fresh can answer about
    work that happened outside the driver.
    """
    outs = stage.declared_outputs()
    if not outs:
        return False                       # nothing declared: never adopt
    if any(not p.exists() for p in outs):
        return False
    ins = stage.declared_inputs()
    # A declared input that is not on disk means the stage cannot have produced
    # these outputs from it. Outputs that predate the current dataset version
    # would otherwise be adopted as done -- which is how the 17 v9 figures were
    # briefly adopted as v10. Refuse rather than guess.
    if any(not p.exists() for p in ins):
        return False
    if not ins:
        return True
    newest_in = max(p.stat().st_mtime for p in ins)
    return all(p.stat().st_mtime >= newest_in for p in outs)


def cmd_freshness(stages, state):
    """Evaluate the freshness check for every stage against the files on disk.

    A stage that ran under the check carries its verdict in the state file; the
    rest are evaluated statically here (outputs present, and newer than their
    inputs). The "rewritten by this run" arm needs a run and is marked n/a.
    """
    print(f"{'stage':26s} {'outputs':>9s} {'verdict':10s} {'source':9s}  detail")
    print("-" * 92)
    tally = {"ok": 0, "stale": 0}
    for st in stages:
        outs = st.declared_outputs()
        present = [p for p in outs if p.exists()]
        rec = state["stages"].get(st.name, {})
        problems = []
        missing = [p for p in outs if not p.exists()]
        if missing:
            problems.append("absent: " + ", ".join(p.name for p in missing[:3]))
        newest_in = None
        for p in st.declared_inputs():
            if p.exists():
                t = p.stat().st_mtime
                newest_in = t if newest_in is None else max(newest_in, t)
        if newest_in is not None:
            older = [p.name for p in present if p.stat().st_mtime < newest_in]
            if older:
                problems.append("older than inputs: " + ", ".join(older[:3]))
        verdict = "ok" if not problems else "STALE"
        source = "run" if rec.get("freshness") else "static"
        if rec.get("freshness") == "stale":
            verdict, problems = "STALE", rec.get("freshness_problems", problems)
        tally["ok" if verdict == "ok" else "stale"] += 1
        print(f"{st.name:26s} {len(present):4d}/{len(outs):<4d} {verdict:10s} {source:9s}  "
              f"{'; '.join(problems)[:44]}")
    print("-" * 92)
    print(f"  {tally['ok']} ok, {tally['stale']} stale, {len(stages)} stages")
    return 1 if tally["stale"] else 0


def cmd_status(stages, state):
    print(f"dataset version: {state.get('version', DS.VERSION)}\n")
    print(f"{'stage':26s} {'status':12s} {'est':>7s} {'actual':>8s}  note")
    print("-" * 88)
    left = 0.0
    for st in stages:
        r = state["stages"].get(st.name, {})
        status = r.get("status", "pending")
        if st.blocked and status != "done":
            status = "BLOCKED"
        actual = hm(r["seconds"] / 60) if r.get("seconds") else ""
        if status not in ("done",):
            left += st.est_min
        note = st.note if status != "BLOCKED" else st.blocked[:44] + "..."
        print(f"{st.name:26s} {status:12s} {hm(st.est_min):>7s} {actual:>8s}  {note[:40]}")
    print("-" * 88)
    print(f"estimated work remaining: {hm(left)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--until", type=str, default=None,
                    help="stop by this wall clock: '08:00', '2026-09-10T08:00', or '14h'")
    ap.add_argument("--start", type=str, default=None,
                    help="dry-run only: pretend the session begins at this time "
                         "(e.g. 17:00), to plan before you start")
    ap.add_argument("--status", action="store_true", help="print progress and exit")
    ap.add_argument("--freshness-report", action="store_true",
                    help="evaluate the freshness check for every stage and exit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--from", dest="start_at", type=str, default=None,
                    help="skip ahead to this stage")
    ap.add_argument("--only", type=str, nargs="+", default=None)
    ap.add_argument("--reset", type=str, nargs="+", default=None,
                    help="mark stages pending so they run again")
    ap.add_argument("--adopt-fresh", action="store_true",
                    help="mark every stage whose freshness check already passes as "
                         "done, without running it; use after work done outside "
                         "the driver, so a session does not redo it")
    ap.add_argument("--skip-blocked", action="store_true",
                    help="continue past a blocked stage instead of stopping")
    ap.add_argument("--keep-going", action="store_true",
                    help="on a stage failure, record it and continue to the next stage "
                         "instead of stopping the run; failures are listed at the end")
    ap.add_argument("--caffeinate", action="store_true",
                    help="re-exec under caffeinate so the Mac cannot sleep mid-run")
    args = ap.parse_args()

    stages = build_stages()
    state = load_state()

    if args.reset:
        for n in args.reset:
            state["stages"].pop(n, None)
        save_state(state)
        print(f"reset: {', '.join(args.reset)}")
        return 0

    if args.adopt_fresh:
        adopted = []
        for st_ in stages:
            if rec(state, st_.name).get("status") == "done":
                continue
            if static_fresh(st_):
                rec(state, st_.name).update(
                    status="done", adopted=True,
                    ended=datetime.now().isoformat(timespec="seconds"))
                adopted.append(st_.name)
        save_state(state)
        print(f"adopted {len(adopted)} stage(s) as done: {', '.join(adopted) or '(none)'}")
        return 0

    if args.freshness_report:
        return cmd_freshness(stages, state)

    if args.status:
        cmd_status(stages, state)
        return 0

    if args.caffeinate and not os.environ.get("_PIPELINE_CAFFEINATED"):
        env = dict(os.environ, _PIPELINE_CAFFEINATED="1")
        argv = [a for a in sys.argv if a != "--caffeinate"]
        os.execvpe("caffeinate", ["caffeinate", "-i", PY, *argv], env)

    deadline = parse_deadline(args.until)
    if args.only:
        stages = [s_ for s_ in stages if s_.name in set(args.only)]
    elif args.start_at:
        names = [s_.name for s_ in stages]
        if args.start_at not in names:
            print(f"unknown stage {args.start_at!r}")
            return 2
        stages = stages[names.index(args.start_at):]

    if not args.dry_run and not take_lock():
        return 1

    print(f"{DS.VERSION} pipeline — {len(stages)} stage(s) in scope")
    if deadline:
        print(f"deadline: {deadline:%Y-%m-%d %H:%M}  "
              f"({hm((deadline - datetime.now()).total_seconds()/60)} from now)")
    print()

    sim = parse_deadline(args.start) if args.start else datetime.now()
    failed = []

    try:
        for st in stages:
            r = rec(state, st.name)
            if r.get("status") == "done":
                continue

            if st.blocked:
                print(f"[{st.name}] BLOCKED: {st.blocked}")
                if args.skip_blocked:
                    print("  --skip-blocked: continuing\n")
                    continue
                print("  stopping. Re-run with --skip-blocked to continue past it.\n")
                break

            now = sim if args.dry_run else datetime.now()
            remaining = ((deadline - now).total_seconds() / 60
                         if deadline else float("inf"))
            if remaining <= 0:
                print("deadline reached — stopping cleanly")
                break
            if st.mode == ATOMIC and st.est_min > remaining:
                print(f"[{st.name}] needs ~{hm(st.est_min)}, only {hm(remaining)} left "
                      f"and it cannot resume — stopping cleanly.")
                break

            was = r.get("status", "pending")
            tag = {"interrupted": " (resuming)", "failed": " (retry)"}.get(was, "")
            if args.dry_run:
                end = sim + timedelta(minutes=st.est_min)
                if deadline and end > deadline:      # resumable, runs into the wall
                    end = deadline
                print(f"[{st.name}]{tag} {sim:%a %H:%M} -> {end:%a %H:%M}"
                      f"  (est {hm(st.est_min)})")
                sim = end
            else:
                print(f"[{st.name}]{tag} est {hm(st.est_min)}, {hm(remaining)} left"
                      if deadline else f"[{st.name}]{tag} est {hm(st.est_min)}")
            if st.note:
                print(f"    note: {st.note}")

            outcome = run_stage(st, state, deadline, args.dry_run)
            print()
            if args.dry_run and deadline and sim >= deadline:
                print(f"--- deadline {deadline:%a %H:%M} falls here; "
                      f"{st.name} would be interrupted and resumed next session ---\n")
                break
            if outcome == "failed":
                msg = (f"stage {st.name} failed (exit {rec(state, st.name)['exit_code']}). "
                       f"Inspect: {rel(LOGS / (st.name + '.log'))}")
                if args.keep_going:
                    # Independent stages should not lose a whole overnight window to
                    # one broken dependency. Stages that read this one's outputs will
                    # fail their own freshness check, so nothing silently consumes a
                    # partial result; the run summary lists every failure.
                    failed.append(st.name)
                    print(f"{msg}\n  --keep-going: continuing to the next stage.\n")
                    continue
                print(f"{msg}\nStopping so nothing downstream reads a partial result.\n"
                      f"Then re-run the same command to retry from here, or pass "
                      f"--keep-going to run the independent stages anyway.")
                break
            if outcome == "interrupted":
                print("stopped at the deadline; progress is on disk. "
                      "Re-run the same command to continue.")
                break
        if failed:
            print(f"\n{len(failed)} stage(s) failed and were skipped: {', '.join(failed)}")
            print("Re-run the same command to retry them once their cause is fixed.")
    finally:
        if not args.dry_run:
            free_lock()

    print()
    cmd_status(build_stages(), state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
