"""Phase 1 calibration check (Session 8, Step B).

Runs BEFORE full dataset generation. Draws N physics-aware (T, c, E) samples,
draws one LatentDraw per sample, and computes the realized soft-mode
|delta-Gamma / Gamma_0| = |gamma_multiplier - 1| on the central-regime mask
(soft mode, baseline underdamped: gamma0/omega0 < 1.0). Reports
median / p90 / p95 and the floor-hit rate against the parameter-card
acceptance targets.

No model training, no full dataset generation. If acceptance fails, the
caller STOPS and reports (auto-retuning is OFF per Session 8).

Central-regime mapping (Session 8). The Phase-0 card mask was
`branch: TO, q_range [0.20, 0.40]`; the 1D Gamma-point generator has no
q-axis or branch index, so the central regime is the soft (TO) mode
restricted to baseline-underdamped spectra (gamma0/omega0 < 1.0). The
acoustic/optical modes (TA / TO2) are excluded from the central diagnostic.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .latent_perturbations import GAMMA_FLOOR, sample_latents
from .sampling import draw_samples
from .spectrum_generator import Gamma_Q, omega_Q

# Parameter-card acceptance targets (configs/phase1_parameter_card.yaml).
MEDIAN_BAND = (0.10, 0.40)
P90_MIN = 0.40
FLOOR_HIT_RATE_MAX = 0.02

DEFAULT_SEED = 20260520


@dataclass
class CalibrationResult:
    n: int
    seed: int
    n_central: int
    median_abs_delta: float
    p90_abs_delta: float
    p95_abs_delta: float
    floor_hit_rate: float
    passed: bool
    failures: list[str]


def run_calibration(n: int = 5000, seed: int = DEFAULT_SEED) -> tuple[CalibrationResult, np.ndarray]:
    """Run the calibration histogram. Returns (result, abs_delta_array)."""
    rng = np.random.default_rng(seed)
    samples = draw_samples(n, rng)

    abs_deltas: list[float] = []
    floor_hits = 0
    total = 0
    for s in samples:
        c_frac = s.c_pct / 100.0
        om0 = float(omega_Q(s.T_K, c_frac, s.E_kVcm))
        gm0 = float(Gamma_Q(s.T_K, c_frac, s.E_kVcm))

        latent = sample_latents(rng)
        raw_mult = 1.0 + latent.alpha * latent.xi1["soft"]   # pre-floor (soft mode)
        total += 1
        if raw_mult < GAMMA_FLOOR:
            floor_hits += 1
        realized_mult = max(GAMMA_FLOOR, raw_mult)
        delta = realized_mult - 1.0                          # (Gamma_realized - Gamma_0)/Gamma_0

        # Central-regime mask: soft mode, baseline underdamped.
        if gm0 / om0 < 1.0:
            abs_deltas.append(abs(delta))

    arr = np.asarray(abs_deltas, dtype=np.float64)
    median = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    p95 = float(np.percentile(arr, 95))
    floor_rate = floor_hits / max(1, total)

    failures: list[str] = []
    if not (MEDIAN_BAND[0] <= median <= MEDIAN_BAND[1]):
        failures.append(f"median_abs_delta {median:.4f} outside {MEDIAN_BAND}")
    if p90 < P90_MIN:
        failures.append(f"p90_abs_delta {p90:.4f} < {P90_MIN}")
    if floor_rate > FLOOR_HIT_RATE_MAX:
        failures.append(f"floor_hit_rate {floor_rate:.4f} > {FLOOR_HIT_RATE_MAX}")

    result = CalibrationResult(
        n=n, seed=seed, n_central=len(arr),
        median_abs_delta=median, p90_abs_delta=p90, p95_abs_delta=p95,
        floor_hit_rate=floor_rate, passed=(len(failures) == 0), failures=failures,
    )
    return result, arr


def _print_report(result: CalibrationResult) -> None:
    print("=" * 64)
    print("Phase 1 calibration check (Step B)")
    print("=" * 64)
    print(f"  N samples drawn         : {result.n}")
    print(f"  seed                    : {result.seed}")
    print(f"  N in central regime     : {result.n_central}  (soft mode, gamma0/omega0 < 1)")
    print(f"  median |dGamma/Gamma0|  : {result.median_abs_delta:.4f}   target {MEDIAN_BAND}")
    print(f"  p90 |dGamma/Gamma0|     : {result.p90_abs_delta:.4f}   target >= {P90_MIN}")
    print(f"  p95 |dGamma/Gamma0|     : {result.p95_abs_delta:.4f}")
    print(f"  floor hit rate          : {result.floor_hit_rate:.4f}   target <= {FLOOR_HIT_RATE_MAX}")
    print("-" * 64)
    if result.passed:
        print("  RESULT: PASS — calibration within acceptance band.")
    else:
        print("  RESULT: FAIL — STOP (auto-retuning is OFF). Failures:")
        for f in result.failures:
            print(f"    - {f}")
    print("=" * 64)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out", type=str, default="results/phase1_calibration.npz")
    args = p.parse_args()

    result, arr = run_calibration(n=args.n, seed=args.seed)
    _print_report(result)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        abs_delta=arr,
        result_json=json.dumps(asdict(result)),
        seed=np.array(args.seed, dtype=np.int64),
        n=np.array(args.n, dtype=np.int64),
    )
    print(f"Saved calibration array -> {out_path}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
