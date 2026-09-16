"""Common interface for all Paper 1 models.

All five models (DHO+ridge, MLP, 1D CNN, wavelet, Spectral Transformer
5a/5b) subclass `BaseModel` so the evaluation harness can treat them
uniformly. Models that don't predict the auxiliary heads return None
from `predict_omega_Q` / `predict_Gamma_Q` and the harness skips those
metrics.

Multi-task loss design (sketch — implemented per-model from Step 4.2).
    Primary head:   M, regressed in log space.
        L_M = MSE(log(M_pred), log(M_true))
    Auxiliary heads (interpretability, not headline metrics per Design.md):
        omega_Q : MSE in linear space (ω_Q range is ~3-11 meV, fairly uniform).
            L_omega = MSE(omega_Q_pred, omega_Q_true)
        Gamma_Q : MSE in log space (Γ_Q range is ~0.5-15 meV, ~1.5 orders).
            L_Gamma = MSE(log(Gamma_Q_pred), log(Gamma_Q_true))
    Total loss with default weights:
        L = 1.0 * L_M + 0.1 * L_omega + 0.1 * L_Gamma

DHO+ridge (Step 4.1) does not need a multi-task loss — the DHO fit
already produces ω_Q and Γ_Q estimates "for free". Those estimates are
exposed via `predict_omega_Q` / `predict_Gamma_Q` so the cross-model
evaluation table is consistent from Step 4.1 onward.
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np


class BaseModel(abc.ABC):
    name: str  # subclasses must set; used in eval-table rows and W&B run tags

    @abc.abstractmethod
    def fit(self, train_ds, val_ds) -> None:
        """Fit the model. `train_ds` and `val_ds` are InsSpectraDataset
        instances. Models tune hyperparameters on val_ds; the held-out
        test sets are off-limits per TASKS.md."""

    @abc.abstractmethod
    def predict_M(self, batch: dict[str, Any]) -> np.ndarray:
        """Per-sample M predictions; shape (B,). `batch` is a dict with
        'spectrum' (B, N_omega), and condition keys 'T_K', 'c_pct',
        'E_kVcm' as needed."""

    def predict_omega_Q(self, batch: dict[str, Any]) -> np.ndarray | None:
        """Per-sample omega_Q predictions, or None if the model does not
        support this auxiliary head."""
        return None

    def predict_Gamma_Q(self, batch: dict[str, Any]) -> np.ndarray | None:
        """Per-sample Gamma_Q predictions, or None if the model does not
        support this auxiliary head."""
        return None

    def diagnostics(self, batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """Model-specific per-sample diagnostics (e.g.
        {'fit_succeeded': bool array} for DHO+ridge). Returns an empty
        dict for models with no diagnostics to report."""
        return {}
