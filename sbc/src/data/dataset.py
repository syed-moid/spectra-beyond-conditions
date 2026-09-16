"""PyTorch Dataset for the INS Paper 1 full dataset.

Severity is a runtime/constructor parameter, NOT baked into the stored
data. The .npz persists clean spectra + per-spectrum augmentation
parameters drawn at severity=1. At __getitem__ time, those parameters
are scaled per the Session 5 rules and applied via `apply_aug_params`.

Stress-test evaluation is simply this Dataset instantiated at multiple
severities over the `stress_base` split, sharing the same 500 base
spectra.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    _TORCH_OK = True
except ImportError:  # numpy-only fallback so the dataset module is testable without torch
    _TorchDataset = object
    _TORCH_OK = False

from .augmentations import apply_aug_params, state_from_clean
from .latent_perturbations import LatentDraw

SplitName = Literal["train", "val", "holdout_T", "holdout_c", "holdout_E", "stress_base"]


class InsSpectraDataset(_TorchDataset):
    """Dataset of INS soft-mode spectra with on-the-fly augmentation replay.

    Parameters
    ----------
    npz_path : Path
        Path to the .npz produced by `scripts/generate_full_dataset.py`.
    split : str
        One of 'train', 'val', 'holdout_T', 'holdout_c', 'holdout_E',
        'stress_base'.
    severity : float, default 1.0
        Runtime severity dial. severity=0 returns clean; severity=1
        reproduces the canonical stored augmentation; severity>1
        extrapolates per the Session 5 scaling rules.
    return_clean : bool, default False
        If True, skip augmentation replay and return the clean spectrum.
    target_keys : tuple of str, default ('M', 'omega_Q', 'Gamma_Q')
        Which targets to include in the returned dict.
    as_torch : bool, default True
        If True (and torch is available), convert outputs to
        torch.Tensor; else return numpy.
    """

    def __init__(
        self,
        npz_path: Path,
        split: SplitName,
        severity: float = 1.0,
        return_clean: bool = False,
        target_keys: tuple[str, ...] = ("M", "omega_Q", "Gamma_Q"),
        as_torch: bool = True,
    ):
        self.npz_path = Path(npz_path)
        self.split = split
        self.severity = float(severity)
        self.return_clean = bool(return_clean)
        self.target_keys = tuple(target_keys)
        self.as_torch = bool(as_torch) and _TORCH_OK

        with np.load(self.npz_path, allow_pickle=False) as z:
            split_arr = z["split"]
            mask = split_arr == split
            if not mask.any():
                raise ValueError(f"No samples in split {split!r}; available: {sorted(set(split_arr.tolist()))}")
            self._idx = np.nonzero(mask)[0]

            # Schema invariant (Session 5): coordinate arrays on the
            # augmentation replay path are stored at float64. Asserting
            # at load time catches the precision-regression class up
            # front rather than after 40 silently-wrong augmentations.
            if z["omega_grid"].dtype != np.float64:
                raise TypeError(
                    f"Schema violation: omega_grid must be float64 to preserve "
                    f"bit-exact augmentation replay (got {z['omega_grid'].dtype}). "
                    f"See SESSION_LOG.md Session 5 schema invariant."
                )
            self._omega_grid = z["omega_grid"]
            self._spectra_clean = z["spectra_clean"][self._idx].astype(np.float32)
            self._T_K = z["T_K"][self._idx].astype(np.float32)
            self._c_pct = z["c_pct"][self._idx].astype(np.float32)
            self._E_kVcm = z["E_kVcm"][self._idx].astype(np.float32)
            self._omega_Q = z["omega_Q"][self._idx].astype(np.float32)
            self._Gamma_Q = z["Gamma_Q"][self._idx].astype(np.float32)
            self._M = z["M"][self._idx].astype(np.float32)
            self._stratum = z["stratum"][self._idx]

            aug_json_arr = z["aug_params_json"][self._idx]
            self._aug_params = [json.loads(str(s)) for s in aug_json_arr]

            # Phase 1 (Session 8): per-spectrum latent draw. Absent in Phase-0
            # datasets -> all None (clean baseline reconstruction).
            if "latent_json" in z.files:
                latent_json_arr = z["latent_json"][self._idx]
                self._latents: list[LatentDraw | None] = [
                    LatentDraw.from_dict(json.loads(str(s))) if str(s) else None
                    for s in latent_json_arr
                ]
            else:
                self._latents = [None] * len(self._idx)

    def __len__(self) -> int:
        return len(self._idx)

    def get_clean(self, idx: int) -> np.ndarray:
        return self._spectra_clean[idx].copy()

    def get_augmented(self, idx: int, severity: float | None = None) -> np.ndarray:
        """Replay augmentation for sample `idx` at given (or default) severity."""
        sev = self.severity if severity is None else float(severity)
        if sev <= 0:
            return self._spectra_clean[idx].astype(np.float64)

        # Reconstruct a state on the *original* omega_grid using the stored
        # clean spectrum + per-mode regeneration (anharmonic_skew and
        # central_peak_width_perturbation need to know the modes).
        state = state_from_clean(
            float(self._T_K[idx]),
            float(self._c_pct[idx]),
            float(self._E_kVcm[idx]),
            omega_grid=self._omega_grid,  # already float64
            latent=self._latents[idx],    # None for Phase-0 data
        )
        # Sanity: clean spectrum from generator should match stored. Float32
        # storage means we accept the regenerated double-precision version
        # as the truth source; the stored float32 is a tagged-along artefact.
        state = apply_aug_params(state, self._aug_params[idx], sev, noise_rng=None)
        return state.spectrum

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.return_clean:
            spec = self._spectra_clean[idx].astype(np.float32)
        else:
            spec = self.get_augmented(idx).astype(np.float32)

        targets_map = {
            "M": self._M[idx],
            "omega_Q": self._omega_Q[idx],
            "Gamma_Q": self._Gamma_Q[idx],
        }
        out: dict[str, Any] = {
            "spectrum": spec,
            "targets": {k: targets_map[k] for k in self.target_keys},
            "conditions": {
                "T_K": float(self._T_K[idx]),
                "c_pct": float(self._c_pct[idx]),
                "E_kVcm": float(self._E_kVcm[idx]),
            },
            "meta": {
                "stratum": str(self._stratum[idx]),
                "split": self.split,
                "dataset_idx": int(idx),
                "applied_severity": self.severity,
            },
        }
        if self.as_torch:
            out["spectrum"] = torch.from_numpy(out["spectrum"])
            out["targets"] = {k: torch.tensor(v, dtype=torch.float32) for k, v in out["targets"].items()}
        return out
