"""Evaluation harness shared across all five Paper 1 models.

The harness instantiates `InsSpectraDataset` per regime/severity, runs
the model, and computes the metric set. Per Design.md, the held-out
test sets are NEVER used for hyperparameter selection — only reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..data.dataset import InsSpectraDataset
from ..models.base import BaseModel
from . import metrics


def collate_dataset(ds) -> dict[str, np.ndarray]:
    """Iterate an InsSpectraDataset (as_torch=False) into stacked numpy arrays.

    Inlined into the harness for the reproducibility package so the evaluation
    path no longer depends on the (excluded) DHO-ridge baseline module; the
    original lived in ``src/models/dho_ridge.py``.
    """
    spectra: list[np.ndarray] = []
    T_K: list[float] = []
    c_pct: list[float] = []
    E_kVcm: list[float] = []
    M: list[float] = []
    omega_Q: list[float] = []
    Gamma_Q: list[float] = []
    for i in range(len(ds)):
        it = ds[i]
        spec = it["spectrum"]
        if hasattr(spec, "numpy"):
            spec = spec.numpy()
        spectra.append(spec)
        T_K.append(it["conditions"]["T_K"])
        c_pct.append(it["conditions"]["c_pct"])
        E_kVcm.append(it["conditions"]["E_kVcm"])
        targets = it["targets"]
        M.append(float(targets["M"]))
        omega_Q.append(float(targets["omega_Q"]))
        Gamma_Q.append(float(targets["Gamma_Q"]))
    return {
        "spectrum": np.stack(spectra).astype(np.float32),
        "T_K": np.array(T_K, dtype=np.float32),
        "c_pct": np.array(c_pct, dtype=np.float32),
        "E_kVcm": np.array(E_kVcm, dtype=np.float32),
        "M": np.array(M, dtype=np.float32),
        "omega_Q": np.array(omega_Q, dtype=np.float32),
        "Gamma_Q": np.array(Gamma_Q, dtype=np.float32),
    }


@dataclass
class EvalResult:
    model_name: str
    regime: str
    severity: float
    N: int
    MAE_M: float
    MAE_logM: float
    pearson_r: float
    # Session-6-follow-up addition: report target variance alongside r so the
    # reader can distinguish "high r on a narrow-variance regime" from "high
    # r on a broad-variance regime". holdout_T (r=0.97, std(M)=0.09) vs
    # in_dist (r=0.76, std(M)=0.75) was the motivating case.
    std_M: float
    std_logM: float
    frac_relerr_gt_50pct: float
    MAE_omega_Q: float | None
    MAE_Gamma_Q: float | None
    fit_success_rate: float | None


def _is_overdamped(item_targets: dict) -> bool:
    """Filter predicate: Gamma_Q > omega_Q / 2 in the dataset's targets."""
    return float(item_targets["Gamma_Q"]) > 0.5 * float(item_targets["omega_Q"])


def evaluate(
    model: BaseModel,
    dataset: InsSpectraDataset,
    regime: str,
    severity: float,
    filter_fn: Callable[[dict], bool] | None = None,
) -> EvalResult:
    """Run `model` on `dataset` and report metrics.

    `filter_fn` is applied per-item against the dataset's targets dict;
    used to carve out the 'overdamped' sub-regime out of in-distribution
    val. Pass None for unfiltered eval.
    """
    batch = collate_dataset(dataset)
    if filter_fn is not None:
        keep = np.array([
            filter_fn({"omega_Q": batch["omega_Q"][i], "Gamma_Q": batch["Gamma_Q"][i]})
            for i in range(len(batch["M"]))
        ], dtype=bool)
        if not keep.any():
            return EvalResult(model.name, regime, severity, 0,
                              float("nan"), float("nan"), float("nan"),
                              float("nan"), float("nan"), float("nan"),
                              None, None, None)
        for k in ("spectrum", "T_K", "c_pct", "E_kVcm", "M", "omega_Q", "Gamma_Q"):
            batch[k] = batch[k][keep]

    true_M = np.asarray(batch["M"], dtype=np.float64)
    pred_M = np.asarray(model.predict_M(batch), dtype=np.float64)

    om_pred = model.predict_omega_Q(batch)
    g_pred = model.predict_Gamma_Q(batch)
    diag = model.diagnostics(batch)

    mae_om = float(np.mean(np.abs(np.asarray(om_pred) - np.asarray(batch["omega_Q"])))) if om_pred is not None else None
    mae_g = float(np.mean(np.abs(np.asarray(g_pred) - np.asarray(batch["Gamma_Q"])))) if g_pred is not None else None
    success_rate = float(np.mean(diag["fit_succeeded"])) if "fit_succeeded" in diag else None

    # Target-variance columns. log() guarded with clip(eps) to match the
    # metrics module's convention. std(M) and std(log M) are reported on
    # the *true* M only (these are properties of the regime, not the model).
    true_M_safe = np.clip(true_M, 1e-9, None)
    std_M = float(np.std(true_M))
    std_logM = float(np.std(np.log(true_M_safe)))

    return EvalResult(
        model_name=model.name,
        regime=regime,
        severity=severity,
        N=int(len(true_M)),
        MAE_M=metrics.mae_M(true_M, pred_M),
        MAE_logM=metrics.mae_log_M(true_M, pred_M),
        pearson_r=metrics.pearson_r(true_M, pred_M),
        std_M=std_M,
        std_logM=std_logM,
        frac_relerr_gt_50pct=metrics.frac_relerr_gt_50pct(true_M, pred_M),
        MAE_omega_Q=mae_om,
        MAE_Gamma_Q=mae_g,
        fit_success_rate=success_rate,
    )


STRESS_SEVERITIES: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)


def run_full_evaluation_suite(model: BaseModel, npz_path: Path) -> pd.DataFrame:
    """Run model across all regimes + the stress severity sweep."""
    rows: list[EvalResult] = []

    val = InsSpectraDataset(npz_path, "val", severity=1.0, as_torch=False)
    rows.append(evaluate(model, val, "in_dist", 1.0))
    rows.append(evaluate(model, val, "overdamped", 1.0, filter_fn=_is_overdamped))

    for h in ("holdout_T", "holdout_c", "holdout_E"):
        ds = InsSpectraDataset(npz_path, h, severity=1.0, as_torch=False)
        rows.append(evaluate(model, ds, h, 1.0))

    for sev in STRESS_SEVERITIES:
        ds = InsSpectraDataset(npz_path, "stress_base", severity=sev, as_torch=False)
        rows.append(evaluate(model, ds, "stress", sev))

    return pd.DataFrame([asdict(r) for r in rows])
