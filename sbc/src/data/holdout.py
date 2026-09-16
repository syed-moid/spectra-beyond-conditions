"""Held-out extrapolation grid for Paper 1.

These constants define the variables that are reserved for the
extrapolation-beyond-calibration evaluation regime (regime 4 in
`Design.md`). Training and in-distribution validation must exclude
these points; the held-out tests are NEVER used for hyperparameter
search.
"""

from __future__ import annotations

# Training ranges (continuous).
T_TRAIN_RANGE_K: tuple[float, float] = (100.0, 500.0)
C_TRAIN_RANGE_PCT: tuple[float, float] = (0.0, 2.0)
E_TRAIN_RANGE_KVCM: tuple[float, float] = (0.0, 2.0)

# Held-out test values.
T_HOLDOUT_K: float = 600.0
C_HOLDOUT_PCT: float = 1.0
E_HOLDOUT_KVCM: float = 4.0

# Exclusion bands around held-out values inside the training range.
# T and E holdouts are already outside the training range, so the bands
# matter only for c.
C_EXCLUSION_HALFWIDTH_PCT: float = 0.15


def is_in_training_region(T_K: float, c_pct: float, E_kVcm: float) -> bool:
    """Return True iff (T, c, E) lies in the training-permissible region.

    Excludes the c=1% exclusion band and stays inside the [min, max]
    boxes for T, c, and E. The exact endpoints of the boxes are
    inclusive on both sides.
    """
    if not (T_TRAIN_RANGE_K[0] <= T_K <= T_TRAIN_RANGE_K[1]):
        return False
    if not (C_TRAIN_RANGE_PCT[0] <= c_pct <= C_TRAIN_RANGE_PCT[1]):
        return False
    if not (E_TRAIN_RANGE_KVCM[0] <= E_kVcm <= E_TRAIN_RANGE_KVCM[1]):
        return False
    if abs(c_pct - C_HOLDOUT_PCT) < C_EXCLUSION_HALFWIDTH_PCT:
        return False
    return True
