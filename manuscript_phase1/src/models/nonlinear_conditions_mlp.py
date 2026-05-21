"""Nonlinear-conditions MLP baseline (Session 7).

Establishes the "nonlinear-conditions floor" that any spectral-side claim
must exceed. Sees ONLY (T_n, c_n, E_n) -- no spectrum. Predicts log_M.

dho_ridge with (T_n, c_n, E_n) + DHO features gives the LINEAR floor
(val_MAE_logM = 0.286).
This MLP gives the NONLINEAR floor by fitting the closed-form physics
function M(T, c, E) directly with 3 -> 64 -> 64 -> 1.

Per Session 7 reframe: the central scientific question becomes "how much
M-relevant information survives in degraded spectra beyond what experimental
conditions alone determine?". Spectral-side models claiming robustness must
demonstrably exceed this floor.

Sanity check expected from this model:
  Predictions are independent of severity (model doesn't see the spectrum).
  val_MAE_logM at any severity == val_MAE_logM at any other severity, modulo
  which sample indices are in the eval set (which is fixed across severities
  for stress_base). If severity-dependent MAE shows up, something is wrong
  with how the eval-time M target is computed for augmented spectra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import torch
from torch import nn

from .base import BaseModel

# Physics-derived condition normalization. Centered scaling with FIXED
# constants (not train statistics) so eval-time OOD values (T=600, c=1, E=4)
# map deterministically. Defined locally in this module; not imported from
# spectral_transformer because the spectral models do not consume conditions
# under the spectrum-only headline design.
COND_T_MEAN = 300.0   # midpoint of train [100, 500]
COND_T_SCALE = 300.0  # T=600 -> 1.0 (just past train edge)
COND_C_MEAN = 1.0     # midpoint of train {0, 2}
COND_C_SCALE = 2.0    # c=1 -> 0.0 (holdout)
COND_E_MEAN = 1.0     # midpoint of train {0, 2}
COND_E_SCALE = 2.0    # E=4 -> 1.5 (holdout)


def normalize_conditions(T_K: np.ndarray, c_pct: np.ndarray, E_kVcm: np.ndarray) -> np.ndarray:
    """Physics-derived centered scaling of (T, c, E) into ~unit-scale 3-vectors.

    Returns shape (N, 3), float32. Caller is responsible for batching.
    """
    T_n = (T_K - COND_T_MEAN) / COND_T_SCALE
    c_n = (c_pct - COND_C_MEAN) / COND_C_SCALE
    E_n = (E_kVcm - COND_E_MEAN) / COND_E_SCALE
    return np.stack([T_n, c_n, E_n], axis=-1).astype(np.float32)


class NonlinearConditionsMLP(nn.Module):
    """3 -> 64 -> 64 -> 1 with GELU, log_M output. ~4.7k params."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        # conditions: (B, 3), already normalized.
        return self.net(conditions).squeeze(-1)


class NonlinearConditionsMLPModel(BaseModel):
    """BaseModel wrapper for the nonlinear-conditions MLP.

    Training-time use: instantiate, call fit_from_arrays(...) -- the eval
    harness needs the BaseModel.fit(train_ds, val_ds) interface but iterating
    InsSpectraDataset just to grab (T, c, E) and M is silly for this baseline.
    A convenience .fit(train_ds, val_ds) is provided for interface conformance.

    Returns:
      predict_M(batch) -> array of M predictions
      predict_omega_Q  -> None (no aux head)
      predict_Gamma_Q  -> None (no aux head)
    """

    name = "nonlinear_conditions_mlp"

    def __init__(self, lr: float = 1e-3, n_epochs: int = 2000, batch_size: int = 256,
                 weight_decay: float = 1e-4, device: str | None = None, seed: int = 42):
        self.lr = lr
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.net: NonlinearConditionsMLP | None = None
        self.history: list[dict[str, float]] = []

    @staticmethod
    def _conditions_from_ds(ds) -> tuple[np.ndarray, np.ndarray]:
        """Extract normalized conditions and log_M target arrays from
        an InsSpectraDataset (uses internal arrays directly to skip
        per-sample augmentation replay)."""
        T = np.asarray(ds._T_K, dtype=np.float32)
        c = np.asarray(ds._c_pct, dtype=np.float32)
        E = np.asarray(ds._E_kVcm, dtype=np.float32)
        M = np.asarray(ds._M, dtype=np.float32)
        cond = normalize_conditions(T, c, E)
        logM = np.log(np.clip(M, 1e-9, None)).astype(np.float32)
        return cond, logM

    @staticmethod
    def _conditions_from_batch(batch: dict[str, Any]) -> np.ndarray:
        T = np.asarray(batch["T_K"], dtype=np.float32)
        c = np.asarray(batch["c_pct"], dtype=np.float32)
        E = np.asarray(batch["E_kVcm"], dtype=np.float32)
        return normalize_conditions(T, c, E)

    def fit(self, train_ds, val_ds) -> None:
        cond_tr, logM_tr = self._conditions_from_ds(train_ds)
        cond_val, logM_val = self._conditions_from_ds(val_ds)
        self.fit_from_arrays(cond_tr, logM_tr, cond_val, logM_val)

    def fit_from_arrays(self, cond_tr: np.ndarray, logM_tr: np.ndarray,
                        cond_val: np.ndarray, logM_val: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(self.seed)
        net = NonlinearConditionsMLP().to(self.device)
        X_tr = torch.from_numpy(cond_tr).to(self.device)
        y_tr = torch.from_numpy(logM_tr).to(self.device)
        X_val = torch.from_numpy(cond_val).to(self.device)
        y_val = torch.from_numpy(logM_val).to(self.device)

        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        n = X_tr.shape[0]
        rng = np.random.default_rng(self.seed)
        best_val = float("inf")
        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

        for ep in range(1, self.n_epochs + 1):
            net.train()
            perm = rng.permutation(n)
            total = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb = X_tr[idx]
                yb = y_tr[idx]
                pred = net(xb)
                loss = nn.functional.mse_loss(pred, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.detach().cpu()) * len(idx)
            train_loss = total / n
            net.eval()
            with torch.no_grad():
                val_pred = net(X_val)
                val_loss = float(nn.functional.mse_loss(val_pred, y_val).cpu())
                val_mae = float((val_pred - y_val).abs().mean().cpu())
            self.history.append({"epoch": ep, "train_loss": train_loss,
                                 "val_loss": val_loss, "val_MAE_logM": val_mae})
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

        net.load_state_dict(best_state)
        self.net = net

    @torch.no_grad()
    def predict_M(self, batch: dict[str, Any]) -> np.ndarray:
        assert self.net is not None, "Call fit() first."
        cond = self._conditions_from_batch(batch)
        cond_t = torch.from_numpy(cond).to(self.device)
        self.net.eval()
        log_M = self.net(cond_t).cpu().numpy()
        return np.exp(log_M).astype(np.float64)

    def save(self, path: Path) -> None:
        assert self.net is not None
        torch.save({
            "state_dict": self.net.state_dict(),
            "config": {"lr": self.lr, "n_epochs": self.n_epochs,
                       "batch_size": self.batch_size,
                       "weight_decay": self.weight_decay, "seed": self.seed},
            "history": self.history,
        }, path)
