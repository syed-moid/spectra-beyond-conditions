"""Step D.5 — Phase 1 dataset sanity check (Session 8).

Three gates, run before training:

  (a) Latent propagation visible: spectra at matched (T, c, E) but different
      latent seeds must look meaningfully different (peak position / width),
      not noise on the same signal. Saves a comparison plot.
  (b) Target spread widened: sigma(M | T, c, E) / <M> on Phase 1 must exceed
      0.05 (Phase 0 is deterministic, == 0). Below 0.05 => latent not biting
      => parameter-card issue => STOP.
  (c) latent_json round-trip: recompute realized (M, omega_Q, Gamma_Q) from the
      stored latent + (T, c, E); must match stored targets => else path bug => STOP.

Exit code 0 if all pass; nonzero if any gate fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from src.data.merit import merit  # noqa: E402
from src.data.latent_perturbations import LatentDraw, sample_latents  # noqa: E402
from src.data.sampling import draw_samples  # noqa: E402
from src.data.spectrum_generator import (  # noqa: E402
    DEFAULT_OMEGA_GRID, Gamma_Q, generate_spectrum, omega_Q,
)

PHASE1_NPZ = ROOT / "data" / "full_dataset_phase1" / "dataset.npz"
ART = ROOT / "results"
SPREAD_MIN = 0.05


def check_a_latent_propagation(n_overlay: int = 12, seed: int = 4242) -> dict:
    """Matched (T, c, E), different latent draws -> overlay plot + spread stats."""
    T, c, E = 250.0, 0.5, 0.0   # underdamped, sharp soft mode -> shifts visible
    grid = DEFAULT_OMEGA_GRID
    rng = np.random.default_rng(seed)
    peaks, fwhms = [], []
    ART.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in range(n_overlay):
        latent = sample_latents(rng)
        out = generate_spectrum(T, c, E, grid, latent=latent)
        spec = out["spectrum"]
        ax.plot(grid, spec, lw=1.0, alpha=0.7)
        # Stokes-side peak position + crude FWHM.
        stk = grid > 0
        g, sp = grid[stk], spec[stk]
        ipk = int(np.argmax(sp))
        peaks.append(float(g[ipk]))
        half = 0.5 * sp[ipk]
        above = g[sp > half]
        fwhms.append(float(above.max() - above.min()) if above.size >= 2 else np.nan)
    ax.set_xlabel("omega (meV)")
    ax.set_ylabel("I(omega)")
    ax.set_title(f"(a) Latent propagation: {n_overlay} draws at matched (T={T}, c={c}, E={E})")
    ax.set_xlim(-15, 15)
    ax.grid(alpha=0.3)
    out_png = ART / "phase1_latent_propagation_check.png"
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

    peaks = np.array(peaks); fwhms = np.array(fwhms)
    peak_spread = float(np.nanstd(peaks))
    fwhm_spread = float(np.nanstd(fwhms))
    # "Meaningfully different": peak positions OR widths visibly spread.
    passed = (peak_spread > 0.05) or (fwhm_spread > 0.10)
    return {
        "gate": "a_latent_propagation",
        "peak_pos_std_meV": peak_spread,
        "fwhm_std_meV": fwhm_spread,
        "peak_range_meV": [float(peaks.min()), float(peaks.max())],
        "plot": str(out_png),
        "passed": bool(passed),
    }


def check_b_target_spread(n_conditions: int = 60, n_latents: int = 200, seed: int = 99) -> dict:
    """sigma(M|T,c,E)/<M> across conditions on Phase 1 (Phase 0 == 0)."""
    rng = np.random.default_rng(seed)
    samples = draw_samples(n_conditions, rng)
    grid = DEFAULT_OMEGA_GRID
    cvs = []
    for s in samples:
        Ms = np.empty(n_latents)
        for j in range(n_latents):
            latent = sample_latents(rng)
            Ms[j] = generate_spectrum(s.T_K, s.c_pct, s.E_kVcm, grid, latent=latent)["M"]
        mean = float(np.mean(Ms))
        cvs.append(float(np.std(Ms) / mean) if mean > 0 else 0.0)
    cvs = np.array(cvs)
    median_cv = float(np.median(cvs))
    return {
        "gate": "b_target_spread",
        "phase0_cv": 0.0,
        "phase1_median_cv": median_cv,
        "phase1_cv_range": [float(cvs.min()), float(cvs.max())],
        "frac_conditions_below_0.05": float(np.mean(cvs < SPREAD_MIN)),
        "threshold": SPREAD_MIN,
        "passed": bool(median_cv >= SPREAD_MIN),
    }


def check_c_latent_roundtrip(n_check: int = 20, seed: int = 7) -> dict:
    """Recompute realized (M, omega_Q, Gamma_Q) from stored latent + (T,c,E)."""
    with np.load(PHASE1_NPZ, allow_pickle=False) as z:
        if "latent_json" not in z.files:
            return {"gate": "c_latent_roundtrip", "passed": False,
                    "error": "latent_json missing from Phase 1 npz"}
        N = z["M"].shape[0]
        rng = np.random.default_rng(seed)
        idxs = rng.choice(N, size=n_check, replace=False)
        max_relerr_M = 0.0
        max_relerr_om = 0.0
        max_relerr_gm = 0.0
        for i in idxs:
            i = int(i)
            T = float(z["T_K"][i]); c = float(z["c_pct"][i]); E = float(z["E_kVcm"][i])
            latent = LatentDraw.from_dict(json.loads(str(z["latent_json"][i])))
            c_frac = c / 100.0
            om_r = omega_Q(T, c_frac, E) * latent.omega_multiplier_soft()
            gm_r = Gamma_Q(T, c_frac, E) * latent.gamma_multiplier("soft")
            M_r = merit(om_r, gm_r, T, E)
            max_relerr_M = max(max_relerr_M, abs(M_r - float(z["M"][i])) / max(1e-9, abs(float(z["M"][i]))))
            max_relerr_om = max(max_relerr_om, abs(om_r - float(z["omega_Q"][i])) / max(1e-9, abs(float(z["omega_Q"][i]))))
            max_relerr_gm = max(max_relerr_gm, abs(gm_r - float(z["Gamma_Q"][i])) / max(1e-9, abs(float(z["Gamma_Q"][i]))))
    tol = 1e-3  # float32 storage tolerance
    passed = max(max_relerr_M, max_relerr_om, max_relerr_gm) < tol
    return {
        "gate": "c_latent_roundtrip",
        "max_relerr_M": max_relerr_M,
        "max_relerr_omega_Q": max_relerr_om,
        "max_relerr_Gamma_Q": max_relerr_gm,
        "tol": tol,
        "passed": bool(passed),
    }


def main() -> int:
    print("=" * 64)
    print("Step D.5 — Phase 1 dataset sanity check")
    print("=" * 64)
    ra = check_a_latent_propagation()
    rb = check_b_target_spread()
    rc = check_c_latent_roundtrip()

    for r in (ra, rb, rc):
        print(json.dumps(r, indent=2))
        print("-" * 64)

    failures = []
    if not ra["passed"]:
        failures.append("(a) latent propagation not visible -> PATH/PHYSICS bug")
    if not rc["passed"]:
        failures.append("(c) latent_json round-trip mismatch -> PATH bug")
    if not rb["passed"]:
        failures.append("(b) target spread < 0.05 -> PARAMETER-CARD calibration issue")

    if failures:
        print("RESULT: FAIL — STOP. Issues:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("RESULT: PASS — all three gates clear. Proceed to Steps E/F.")
    (ART / "phase1_sanity_check.json").write_text(json.dumps({"a": ra, "b": rb, "c": rc}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
