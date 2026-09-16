"""Metrics for Paper 1 evaluation."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


def mae_M(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(true - pred)))


def mae_log_M(true: np.ndarray, pred: np.ndarray, eps: float = 1e-9) -> float:
    return float(np.mean(np.abs(np.log(np.clip(true, eps, None)) - np.log(np.clip(pred, eps, None)))))


def pearson_r(true: np.ndarray, pred: np.ndarray) -> float:
    if len(true) < 2:
        return float("nan")
    finite = np.isfinite(true) & np.isfinite(pred)
    if int(finite.sum()) < 2:
        return float("nan")
    r, _ = pearsonr(true[finite], pred[finite])
    return float(r)


def frac_relerr_gt_50pct(true: np.ndarray, pred: np.ndarray, eps: float = 1e-9) -> float:
    denom = np.clip(np.abs(true), eps, None)
    relerr = np.abs(true - pred) / denom
    return float(np.mean(relerr > 0.5))
