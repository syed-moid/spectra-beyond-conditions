"""Reproduce the Phase 0 conditions-only floor (MAE_logM ~= 0.0048).

This is the diagnostic finding that motivated the Phase 1 reformulation: on the
*deterministic-target* Phase 0 dataset, a tiny MLP that sees ONLY the
experimental conditions (T, c, E) -- never the spectrum -- predicts the merit
function M almost perfectly (MAE_logM ~= 0.0048). M is a closed-form function
of (T, c, E) in Phase 0, so the spectrum carries no information beyond the
conditions, and any "spectral representation learning" claim on this data is
ill-posed. Phase 1 fixes this by injecting latent microphysics (xi1, xi2) so
that M is no longer deterministic in (T, c, E).

This script is self-contained: it loads the Phase 0 dataset .npz directly
(no InsSpectraDataset dependency) and trains the conditions-only MLP with the
same hyperparameters used in the original diagnostic
(scripts/diagnostic_step4_2.py): lr=1e-3, 1500 epochs, batch 256, wd=1e-4,
seed=42.

Run (from this directory):
    python reproduce_phase0_floor.py

Expected output (matches floor_reproduction/expected_phase0_floor.json):
    in_dist (val, severity=1)  MAE_logM ~= 0.0049
    stress_base                MAE_logM ~= 0.0048
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from nonlinear_conditions_mlp import (  # noqa: E402
    NonlinearConditionsMLPModel,
    normalize_conditions,
)

DEFAULT_NPZ = HERE.parents[0] / "data" / "full_dataset" / "dataset.npz"
EXPECTED_JSON = HERE / "expected_phase0_floor.json"


def _split_arrays(z, split: str) -> dict[str, np.ndarray]:
    """Pull (T, c, E, M) for one split straight out of the .npz."""
    mask = z["split"] == split
    if not mask.any():
        raise ValueError(f"split {split!r} not found; have {sorted(set(z['split'].tolist()))}")
    return {
        "T_K": z["T_K"][mask].astype(np.float32),
        "c_pct": z["c_pct"][mask].astype(np.float32),
        "E_kVcm": z["E_kVcm"][mask].astype(np.float32),
        "M": z["M"][mask].astype(np.float32),
    }


def _mae_logM(model: NonlinearConditionsMLPModel, batch: dict[str, np.ndarray]) -> float:
    pred = np.asarray(model.predict_M(batch), dtype=np.float64)
    true = np.asarray(batch["M"], dtype=np.float64)
    log_pred = np.log(np.clip(pred, 1e-9, None))
    log_true = np.log(np.clip(true, 1e-9, None))
    return float(np.mean(np.abs(log_pred - log_true)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-path", type=str, default=str(DEFAULT_NPZ),
                    help="Phase 0 deterministic-target dataset .npz.")
    ap.add_argument("--device", type=str, default=None,
                    help="torch device (cpu / mps / cuda). Default: auto.")
    args = ap.parse_args()

    npz_path = Path(args.data_path)
    if not npz_path.exists():
        raise SystemExit(f"Phase 0 dataset not found at {npz_path}")
    print(f"Phase 0 dataset: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as z:
        train = _split_arrays(z, "train")
        val = _split_arrays(z, "val")
        stress = _split_arrays(z, "stress_base")

    print(f"  train n={len(train['M'])}  val n={len(val['M'])}  stress_base n={len(stress['M'])}")

    # Fit the conditions-only MLP on (T, c, E) -> log M.
    cond_tr = normalize_conditions(train["T_K"], train["c_pct"], train["E_kVcm"])
    logM_tr = np.log(np.clip(train["M"], 1e-9, None)).astype(np.float32)
    cond_val = normalize_conditions(val["T_K"], val["c_pct"], val["E_kVcm"])
    logM_val = np.log(np.clip(val["M"], 1e-9, None)).astype(np.float32)

    print(">>> training conditions-only MLP (3 -> 64 -> 64 -> 1), 1500 epochs ...")
    model = NonlinearConditionsMLPModel(lr=1e-3, n_epochs=1500, batch_size=256,
                                        weight_decay=1e-4, device=args.device, seed=42)
    model.fit_from_arrays(cond_tr, logM_tr, cond_val, logM_val)

    in_dist = _mae_logM(model, val)
    stress_mae = _mae_logM(model, stress)

    print()
    print("=" * 56)
    print("PHASE 0 CONDITIONS-ONLY FLOOR")
    print("=" * 56)
    print(f"  in_dist (val, severity=1)  MAE_logM = {in_dist:.4f}")
    print(f"  stress_base                MAE_logM = {stress_mae:.4f}")
    print("=" * 56)

    if EXPECTED_JSON.exists():
        exp = json.loads(EXPECTED_JSON.read_text())
        exp_floor = exp["nonlinear_conditions_mlp"]["per_severity"]
        print("Reference (expected_phase0_floor.json):")
        print(f"  in_dist        : {exp_floor['in_dist_val_severity_1']:.4f}")
        print(f"  stress (sev 1) : {exp_floor['stress_severity_1.0']:.4f}")
        print("\nThe headline Phase 0 floor reported in the manuscript is 0.0048.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
