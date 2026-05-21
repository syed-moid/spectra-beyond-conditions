"""Physics-aware (T, E, c) sampler for the Paper 1 training set.

Strategy
--------
Half of the samples come from a uniform background over the training
region (so the model still sees easy underdamped cases). The other half
come from a difficult-regime mixture, split evenly across three
sub-distributions chosen because each is where the DHO-fit + ridge
baseline tends to fail:

    1. **Overdamped near T_C** — T Gaussian-clustered at 395 K with high
       c. This is where Gamma >= omega/2 and DHO fitting becomes
       unstable. It is the regime Paper 1 is fundamentally about.
    2. **Central-peak onset** — T uniform across the |T - T_C| < 50 K
       band where the central-peak term turns on. The spectrum shape
       changes qualitatively across this boundary.
    3. **Field-driven crossover** — high E combined with mid-to-high T,
       where the E^2 damping term pushes the linewidth across the
       overdamping threshold.

Each sample is tagged with its `stratum` so the coverage plot in the
inspection notebook can show the mixture's effect on density.

Held-out points (c = 1% +/- 0.15%) are excluded by rejection: if a
candidate falls in the exclusion band it is redrawn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .holdout import (
    C_EXCLUSION_HALFWIDTH_PCT,
    C_HOLDOUT_PCT,
    C_TRAIN_RANGE_PCT,
    E_TRAIN_RANGE_KVCM,
    T_TRAIN_RANGE_K,
)

T_C_K: float = 395.0

STRATA = ("uniform", "overdamped_near_Tc", "central_peak_onset", "field_crossover")

# Fraction of samples taken from the difficult mixture (vs uniform).
DIFFICULT_FRACTION_DEFAULT: float = 0.5


@dataclass
class Sample:
    T_K: float
    c_pct: float
    E_kVcm: float
    stratum: str


def _in_c_exclusion_band(c_pct: float) -> bool:
    return abs(c_pct - C_HOLDOUT_PCT) < C_EXCLUSION_HALFWIDTH_PCT


def _draw_uniform(rng: np.random.Generator) -> tuple[float, float, float]:
    T = rng.uniform(*T_TRAIN_RANGE_K)
    c = rng.uniform(*C_TRAIN_RANGE_PCT)
    E = rng.uniform(*E_TRAIN_RANGE_KVCM)
    return T, c, E


def _draw_overdamped_near_Tc(rng: np.random.Generator) -> tuple[float, float, float]:
    # Gaussian around Tc, clipped to training range.
    T = float(np.clip(rng.normal(T_C_K, 25.0), *T_TRAIN_RANGE_K))
    # Beta(2,1) biases toward upper c. Then map to [1.2%, 2.0%].
    c = 1.2 + 0.8 * rng.beta(2.0, 1.0)
    E = rng.uniform(*E_TRAIN_RANGE_KVCM)
    return T, c, E


def _draw_central_peak_onset(rng: np.random.Generator) -> tuple[float, float, float]:
    lo = max(T_TRAIN_RANGE_K[0], T_C_K - 50.0)
    hi = min(T_TRAIN_RANGE_K[1], T_C_K + 50.0)
    T = rng.uniform(lo, hi)
    c = rng.uniform(*C_TRAIN_RANGE_PCT)
    E = rng.uniform(*E_TRAIN_RANGE_KVCM)
    return T, c, E


def _draw_field_crossover(rng: np.random.Generator) -> tuple[float, float, float]:
    T = rng.uniform(300.0, T_TRAIN_RANGE_K[1])
    c = rng.uniform(*C_TRAIN_RANGE_PCT)
    # Bias E toward upper end via Beta(2,1) on [1.5, 2.0].
    E = 1.5 + 0.5 * rng.beta(2.0, 1.0)
    return T, c, E


_DIFFICULT_DRAWERS = {
    "overdamped_near_Tc": _draw_overdamped_near_Tc,
    "central_peak_onset": _draw_central_peak_onset,
    "field_crossover": _draw_field_crossover,
}


def _draw_one(rng: np.random.Generator, difficult_fraction: float) -> Sample:
    if rng.random() < difficult_fraction:
        stratum = rng.choice(list(_DIFFICULT_DRAWERS.keys()))
        T, c, E = _DIFFICULT_DRAWERS[stratum](rng)
    else:
        stratum = "uniform"
        T, c, E = _draw_uniform(rng)
    return Sample(T_K=T, c_pct=c, E_kVcm=E, stratum=stratum)


def draw_samples(
    n: int,
    rng: np.random.Generator,
    difficult_fraction: float = DIFFICULT_FRACTION_DEFAULT,
    pin: dict[str, float] | None = None,
) -> List[Sample]:
    """Draw `n` (T, c, E) samples with stratum labels.

    Parameters
    ----------
    pin : optional mapping of axis name to value
        Keys may be 'T_K', 'c_pct', or 'E_kVcm'. If provided, the
        stratum drawer's value for that axis is overridden after
        sampling, preserving any biased c-or-E sub-distribution on the
        non-pinned axes (see Session 5 stratum-pin behavior).

    The c-holdout exclusion band is applied only when c is NOT pinned —
    holdout test sets pin one axis and explicitly cover the otherwise-
    excluded c values (this is the point of the holdout).
    """
    pin = pin or {}
    out: List[Sample] = []
    while len(out) < n:
        s = _draw_one(rng, difficult_fraction)
        if "T_K" in pin:
            s.T_K = float(pin["T_K"])
        if "c_pct" in pin:
            s.c_pct = float(pin["c_pct"])
        if "E_kVcm" in pin:
            s.E_kVcm = float(pin["E_kVcm"])
        if "c_pct" not in pin and _in_c_exclusion_band(s.c_pct):
            continue
        out.append(s)
    return out
