"""Step A unit tests for the Phase 1 generator changes (Session 8).

Covers:
  * Delta-omega consolidation: E=0 soft-mode spectrum byte-identical to the
    old-convention (delta=Delta_omega(0)=0); E!=0 differs.
  * Latent perturbation correctness: determinism, Gamma floor, beta band,
    realized (omega, Gamma) in the output, M recomputed from realized values,
    and the M-independence of acoustic/optical perturbations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# The src/ package lives under manuscript_phase1/ in this reproducibility layout.
ROOT = Path(__file__).resolve().parents[1] / "manuscript_phase1"
sys.path.insert(0, str(ROOT))

from src.data.merit import merit  # noqa: E402
from src.data.spectrum_generator import (  # noqa: E402
    Delta_omega,
    Gamma_Q,
    dho,
    generate_spectrum,
    omega_Q,
    soft_mode_F2,
)
from src.data.latent_perturbations import (  # noqa: E402
    GAMMA_FLOOR,
    MODES,
    LatentDraw,
    sample_beta_xi2,
    sample_latents,
)


# --------------------------------------------------------------------- #
# Delta-omega consolidation                                             #
# --------------------------------------------------------------------- #

def _old_convention_soft(T, c, E, grid):
    """Reconstruct the soft mode under the OLD double-counted convention:
    delta = Delta_omega(E) passed into the DHO denominator while omega_Q
    already carries the +0.01*E^2 shift."""
    c_frac = c / 100.0
    om = omega_Q(T, c_frac, E)
    gm = Gamma_Q(T, c_frac, E)
    return dho(grid, om, gm, Delta_omega(E), T, soft_mode_F2())


def test_delta_omega_consolidation_E0_byte_identical():
    T, c, E = 250.0, 1.0, 0.0
    out = generate_spectrum(T, c, E)
    grid = out["omega_grid"]
    old_soft = _old_convention_soft(T, c, E, grid)
    # At E=0, Delta_omega(0)=0, so the new (delta=0) and old (delta=0) soft modes
    # are byte-identical.
    assert np.array_equal(out["modes"]["soft"], old_soft)


def test_delta_omega_consolidation_Enonzero_differs():
    T, c, E = 250.0, 1.0, 2.0
    out = generate_spectrum(T, c, E)
    grid = out["omega_grid"]
    old_soft = _old_convention_soft(T, c, E, grid)
    # At E!=0 the new soft mode (delta=0) must differ from the old double-counted
    # convention (delta=Delta_omega(E) != 0).
    assert not np.array_equal(out["modes"]["soft"], old_soft)


def test_M_unchanged_by_delta_fix():
    # M depends on (omega_Q, Gamma_Q, T, E), never on the lineshape delta, so the
    # consolidation must not change M at any E.
    for E in (0.0, 2.0, 4.0):
        out = generate_spectrum(300.0, 0.0, E)
        expected = merit(omega_Q(300.0, 0.0, E), Gamma_Q(300.0, 0.0, E), 300.0, E)
        assert abs(out["M"] - expected) < 1e-12


# --------------------------------------------------------------------- #
# Latent perturbations                                                  #
# --------------------------------------------------------------------- #

def test_latent_determinism():
    a = sample_latents(np.random.default_rng(7))
    b = sample_latents(np.random.default_rng(7))
    assert a.alpha == b.alpha
    assert a.xi1 == b.xi1
    assert a.beta_xi2 == b.beta_xi2


def test_latent_distinct_across_seeds():
    a = sample_latents(np.random.default_rng(7))
    b = sample_latents(np.random.default_rng(8))
    assert a.alpha != b.alpha or a.xi1 != b.xi1 or a.beta_xi2 != b.beta_xi2


def test_gamma_multiplier_floor():
    # alpha=10, xi1_soft=-1 -> 1 + 10*(-1) = -9 -> floored to 0.05.
    latent = LatentDraw(alpha=10.0, xi1={m: -1.0 for m in MODES}, beta_xi2=0.0)
    assert latent.gamma_multiplier("soft") == GAMMA_FLOOR


def test_gamma_multiplier_unfloored():
    latent = LatentDraw(alpha=0.3, xi1={m: 0.5 for m in MODES}, beta_xi2=0.0)
    assert abs(latent.gamma_multiplier("soft") - (1.0 + 0.3 * 0.5)) < 1e-12


def test_beta_xi2_within_band():
    rng = np.random.default_rng(123)
    vals = np.array([sample_beta_xi2(rng) for _ in range(2000)])
    assert vals.min() >= -0.05 - 1e-9
    assert vals.max() <= 0.07 + 1e-9
    # Positive-hardening skew: mean should be > 0.
    assert vals.mean() > 0.0


def test_latent_changes_spectrum_and_M():
    T, c, E = 380.0, 1.0, 0.0
    clean = generate_spectrum(T, c, E)
    latent = sample_latents(np.random.default_rng(3))
    pert = generate_spectrum(T, c, E, latent=latent)
    assert not np.array_equal(clean["spectrum"], pert["spectrum"])
    # M must move (latent perturbs soft omega and Gamma, both of which enter M).
    assert clean["M"] != pert["M"]
    # Output omega_Q / Gamma_Q are the REALIZED values.
    assert pert["omega_Q"] != clean["omega_Q"] or pert["Gamma_Q"] != clean["Gamma_Q"]


def test_M_recomputed_from_realized():
    T, c, E = 350.0, 0.0, 2.0
    latent = sample_latents(np.random.default_rng(11))
    out = generate_spectrum(T, c, E, latent=latent)
    # M in the output must equal merit() of the REALIZED soft (omega, Gamma).
    expected = merit(out["omega_Q"], out["Gamma_Q"], T, E)
    assert abs(out["M"] - expected) < 1e-12


def test_acoustic_optical_perturbation_does_not_change_M():
    # Two latents identical on the soft mode and beta_xi2, differing only in the
    # acoustic/optical xi1. M depends on the soft mode only -> identical M, but
    # the spectra must differ (acoustic/optical Gamma changes).
    T, c, E = 300.0, 0.0, 0.0
    base_soft_xi1 = 0.4
    latent_a = LatentDraw(alpha=0.3,
                          xi1={"soft": base_soft_xi1, "acoustic": 0.5, "optical": -0.5},
                          beta_xi2=0.02)
    latent_b = LatentDraw(alpha=0.3,
                          xi1={"soft": base_soft_xi1, "acoustic": -0.7, "optical": 0.9},
                          beta_xi2=0.02)
    out_a = generate_spectrum(T, c, E, latent=latent_a)
    out_b = generate_spectrum(T, c, E, latent=latent_b)
    assert abs(out_a["M"] - out_b["M"]) < 1e-12          # M unchanged
    assert not np.array_equal(out_a["spectrum"], out_b["spectrum"])  # spectra differ


def test_clean_path_unchanged_by_latent_none():
    # latent=None must reproduce the standard clean spectrum (regression guard).
    out1 = generate_spectrum(420.0, 2.0, 0.0)
    out2 = generate_spectrum(420.0, 2.0, 0.0, latent=None)
    assert np.array_equal(out1["spectrum"], out2["spectrum"])
    assert out1["latent"] is None


if __name__ == "__main__":  # pragma: no cover
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("All generator tests pass.")
