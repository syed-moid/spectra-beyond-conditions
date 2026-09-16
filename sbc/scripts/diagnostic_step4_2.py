"""Step 4.2 diagnostic (Session 7 reframe).

Runs two short trainings and reports per-severity val_MAE_logM for each:

  (A) Spectrum-only ST-5b seed=42, 25 epochs.
      Uses the full Step 4.2 training pipeline (uniform-severity training,
      full-spectrum standardization, sigma=1 RFF). NO metadata token --
      tests how much the spectrum alone carries.

  (B) Nonlinear-conditions MLP seed=42 trained to convergence.
      Sees only (T_n, c_n, E_n). Establishes the nonlinear-conditions floor.
      Sanity check: per-severity val_MAE_logM should be flat (no spectrum
      dependence). If it isn't, the M target is not invariant to augmentation
      severity, which would be a data-pipeline bug.

Outputs:
  outputs/eval/diagnostic_step4.2_<date>/diagnostic_report.json
  + per-model per-severity tables printed to stdout.

No overnight runs are launched from this script. The user reviews the
diagnostic and decides which configuration to commit to.
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message=".*enable_nested_tensor is True.*")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.data.dataset import InsSpectraDataset  # noqa: E402
from src.models.nonlinear_conditions_mlp import NonlinearConditionsMLPModel  # noqa: E402
from src.models.spectral_transformer import SpectralTransformerModel  # noqa: E402


# Defaults to the Phase 1 dataset bundled in this package. For the historical
# Phase 0 conditions-only floor (0.0048), use
# legacy_phase0/original_deterministic_task/floor_reproduction/reproduce_phase0_floor.py
# or pass --data-path ../../legacy_phase0/original_deterministic_task/data/full_dataset/dataset.npz
DATASET_PATH = ROOT / "data" / "full_dataset_phase1" / "dataset.npz"
OUTPUT_ROOT = ROOT / "results" / "diagnostic_runs"
SEVERITIES_TO_REPORT: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)


def _stress_collate_at_severity(severity: float) -> dict[str, np.ndarray]:
    """Materialize the stress_base split at the given severity into a batch
    dict (the format the eval harness / BaseModel.predict_M expects)."""
    ds = InsSpectraDataset(DATASET_PATH, "stress_base", severity=severity, as_torch=False)
    n = len(ds)
    spectra = np.empty((n, 600), dtype=np.float32)
    T_K = np.empty(n, dtype=np.float32)
    c_pct = np.empty(n, dtype=np.float32)
    E_kVcm = np.empty(n, dtype=np.float32)
    M = np.empty(n, dtype=np.float32)
    for i in range(n):
        it = ds[i]
        s = it["spectrum"]
        if hasattr(s, "numpy"):
            s = s.numpy()
        spectra[i] = s.astype(np.float32)
        T_K[i] = float(it["conditions"]["T_K"])
        c_pct[i] = float(it["conditions"]["c_pct"])
        E_kVcm[i] = float(it["conditions"]["E_kVcm"])
        M[i] = float(it["targets"]["M"])
    return {
        "spectrum": spectra,
        "T_K": T_K, "c_pct": c_pct, "E_kVcm": E_kVcm,
        "M": M,
    }


def _val_collate() -> dict[str, np.ndarray]:
    ds = InsSpectraDataset(DATASET_PATH, "val", severity=1.0, as_torch=False)
    n = len(ds)
    spectra = np.empty((n, 600), dtype=np.float32)
    T_K = np.empty(n, dtype=np.float32)
    c_pct = np.empty(n, dtype=np.float32)
    E_kVcm = np.empty(n, dtype=np.float32)
    M = np.empty(n, dtype=np.float32)
    for i in range(n):
        it = ds[i]
        s = it["spectrum"]
        if hasattr(s, "numpy"):
            s = s.numpy()
        spectra[i] = s.astype(np.float32)
        T_K[i] = float(it["conditions"]["T_K"])
        c_pct[i] = float(it["conditions"]["c_pct"])
        E_kVcm[i] = float(it["conditions"]["E_kVcm"])
        M[i] = float(it["targets"]["M"])
    return {
        "spectrum": spectra,
        "T_K": T_K, "c_pct": c_pct, "E_kVcm": E_kVcm,
        "M": M,
    }


def _mae_logM(model, batch: dict[str, np.ndarray]) -> float:
    pred = np.asarray(model.predict_M(batch), dtype=np.float64)
    true = np.asarray(batch["M"], dtype=np.float64)
    log_pred = np.log(np.clip(pred, 1e-9, None))
    log_true = np.log(np.clip(true, 1e-9, None))
    return float(np.mean(np.abs(log_pred - log_true)))


def per_severity_eval(model, severities=SEVERITIES_TO_REPORT) -> dict[str, float]:
    out: dict[str, float] = {}
    for sev in severities:
        batch = _stress_collate_at_severity(sev)
        out[f"stress_severity_{sev}"] = _mae_logM(model, batch)
    # In-distribution val at severity=1 too, as a cross-check.
    val_batch = _val_collate()
    out["in_dist_val_severity_1"] = _mae_logM(model, val_batch)
    return out


def run_spectrum_only_st_5b() -> tuple[dict[str, float], dict[str, float]]:
    """Invoke the existing training script as a subprocess, then load
    best.pt and run per-severity eval."""
    import subprocess
    today = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_tag = f"diagnostic_{today}"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_spectral_transformer.py"),
        "--arch", "5b",
        "--seeds", "42",
        "--limit-epochs", "25",
        "--out-tag", out_tag,
        "--data-path", str(DATASET_PATH),
        # No --use-metadata -> spectrum-only (the headline configuration).
    ]
    print(">>> launching:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print("--- subprocess stdout tail ---")
    print(proc.stdout[-3000:])
    if proc.returncode != 0:
        print("--- subprocess stderr tail ---")
        print(proc.stderr[-3000:])
        raise RuntimeError(f"ST-5b training subprocess exited with code {proc.returncode}")
    # Parse final-epoch train/val info from run_meta.json.
    run_dir = OUTPUT_ROOT / f"spectral_transformer_step4.2_{out_tag}" / "5b_seed42"
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        raise RuntimeError(f"missing run_meta at {meta_path}")
    meta = json.loads(meta_path.read_text())

    ckpt = run_dir / "checkpoints" / "best.pt"
    if not ckpt.exists():
        raise RuntimeError(f"missing best.pt at {ckpt}")
    print(f"\n>>> loading {ckpt} for per-severity eval")
    model = SpectralTransformerModel.from_checkpoint(ckpt, device="cpu")
    per_sev = per_severity_eval(model)
    return per_sev, meta


def run_nonlinear_conditions_mlp() -> dict[str, float]:
    print("\n>>> training nonlinear-conditions MLP (3 -> 64 -> 64 -> 1) ...")
    train_ds = InsSpectraDataset(DATASET_PATH, "train", severity=1.0, as_torch=False)
    val_ds = InsSpectraDataset(DATASET_PATH, "val", severity=1.0, as_torch=False)
    model = NonlinearConditionsMLPModel(lr=1e-3, n_epochs=1500, batch_size=256,
                                         weight_decay=1e-4, seed=42)
    model.fit(train_ds, val_ds)
    print(f"    trained {len(model.history)} epochs; final val_MAE_logM "
          f"= {model.history[-1]['val_MAE_logM']:.4f}")
    print("\n>>> per-severity eval for nonlinear-conditions MLP")
    return per_severity_eval(model)


def main() -> int:
    import argparse
    global DATASET_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", type=str, default=None,
                    help="Dataset .npz. Defaults to Phase 0; pass the Phase 1 path for the live floor.")
    ap.add_argument("--param-card", type=str, default=None,
                    help="Parameter card path (provenance only).")
    args = ap.parse_args()
    if args.data_path:
        DATASET_PATH = Path(args.data_path)
    print(f"Diagnostic dataset: {DATASET_PATH}")

    today = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"diagnostic_step4.2_{today}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # (A) Spectrum-only ST-5b.
    print("=" * 72)
    print("(A) Spectrum-only ST-5b seed=42, 25 epochs")
    print("=" * 72)
    st_per_sev, st_meta = run_spectrum_only_st_5b()

    # (B) Nonlinear-conditions MLP.
    print()
    print("=" * 72)
    print("(B) Nonlinear-conditions MLP seed=42")
    print("=" * 72)
    mlp_per_sev = run_nonlinear_conditions_mlp()

    # Report.
    print()
    print("=" * 72)
    print("DIAGNOSTIC RESULTS")
    print("=" * 72)
    print()
    print(f"{'Severity':>10}  {'ST-5b (spectrum)':>22}  {'NL-cond MLP':>14}  {'delta (ST - MLP)':>18}")
    print("-" * 70)
    for sev in SEVERITIES_TO_REPORT:
        k = f"stress_severity_{sev}"
        st_val = st_per_sev[k]
        mlp_val = mlp_per_sev[k]
        delta = st_val - mlp_val
        print(f"{sev:>10.2f}  {st_val:>22.4f}  {mlp_val:>14.4f}  {delta:>18.4f}")
    print(f"{'in_dist (sev=1 val)':>10}  "
          f"{st_per_sev['in_dist_val_severity_1']:>22.4f}  "
          f"{mlp_per_sev['in_dist_val_severity_1']:>14.4f}  "
          f"{st_per_sev['in_dist_val_severity_1'] - mlp_per_sev['in_dist_val_severity_1']:>18.4f}")

    print()
    print(f"Spectrum-only ST-5b train_val_ratio (last epoch) : {st_meta['train_val_ratio']:.4f}")
    print(f"Spectrum-only ST-5b best_val_MAE_logM            : {st_meta['best_val_mae_logM']:.4f}")
    print(f"Spectrum-only ST-5b best_epoch                   : {st_meta['best_epoch']}")
    print(f"Spectrum-only ST-5b epochs_trained               : {st_meta['epochs_trained']}")

    payload = {
        "spectrum_only_st_5b": {
            "per_severity": st_per_sev,
            "run_meta": st_meta,
        },
        "nonlinear_conditions_mlp": {
            "per_severity": mlp_per_sev,
        },
    }
    report_path = out_dir / "diagnostic_report.json"
    report_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
