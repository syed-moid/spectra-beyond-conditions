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

# sbc/ is the importable package at the repository root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sbc.data.merit import merit  # noqa: E402
from sbc.data.spectrum_generator import (  # noqa: E402
    K_B,
    Delta_omega,
    Gamma_Q,
    bose,
    dho,
    generate_spectrum,
    omega_Q,
    soft_mode_F2,
)
from sbc.data.latent_perturbations import (  # noqa: E402
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


# --------------------------------------------------------------------- #
# Detailed balance (Bose-factor unit fix)                               #
# --------------------------------------------------------------------- #

def test_detailed_balance_ratio_depends_on_energy_transfer():
    """S(+w)/S(-w) = exp(w / k_B T) — a function of the ENERGY TRANSFER.

    The test this replaces asserted the ratio equalled exp(omega0 / k_B T), the
    value at the MODE frequency. That is not detailed balance, and because the
    v7/v8 lineshape was even in w the wrong invariant held at every w, so the
    test passed on defective code.
    """
    for T in (150.0, 400.0):
        for omega0 in (2.0, 5.0, 11.0):
            for w in (1.0, 2.5, 4.0, 7.0, 9.0):
                grid = np.array([-w, w])
                S = dho(grid, omega0, 0.9, 0.0, T, 1.0)
                assert np.isclose(S[1] / S[0], np.exp(w / (K_B * T)), rtol=1e-10), (
                    T, omega0, w, S[1] / S[0], np.exp(w / (K_B * T)))


def test_continuity_at_elastic_line():
    """No jump at w = 0: the two-sided difference vanishes LINEARLY in d.

    Note on the threshold. A fixed bound such as |S(+d) - S(-d)| / S(d) < 1e-6
    for d = 1e-3 meV is not satisfiable by any spectrum obeying detailed
    balance: S(+d)/S(-d) = exp(d / k_B T) exactly, so the relative two-sided
    difference is exp(d/k_B T) - 1 ~ d / k_B T, which is 7.7e-5 at d = 1e-3 meV
    and T = 150 K. A nonzero slope at the elastic line is required physics, not
    a defect. Continuity is therefore asserted as (a) the difference matching
    the detailed-balance value and (b) vanishing linearly as d -> 0.
    """
    for T in (150.0, 400.0):
        for omega0 in (2.0, 5.0, 11.0):
            prev = None
            for d in (1e-3, 1e-4, 1e-5):
                S = dho(np.array([-d, d]), omega0, 0.9, 0.0, T, 1.0)
                rel = abs(S[1] - S[0]) / S[1]
                assert np.isclose(S[1] / S[0], np.exp(d / (K_B * T)), rtol=1e-10)
                if prev is not None:
                    assert 8.0 < prev / rel < 12.0, (T, omega0, d, prev / rel)
                prev = rel
            assert rel < 1e-6, (T, omega0, rel)


def test_elastic_line_limit():
    """S(w -> 0) = k_B T * 4 * Gamma * omega0 * F2 / (pi * omega0^4)."""
    for T in (150.0, 400.0):
        for omega0, Gamma in ((2.0, 0.5), (5.0, 0.9), (11.0, 1.5)):
            expect = K_B * T * 4.0 * Gamma * omega0 / (np.pi * omega0 ** 4)
            got = float(dho(np.array([1e-9]), omega0, Gamma, 0.0, T, 1.0)[0])
            assert np.isclose(got, expect, rtol=1e-6), (T, omega0, Gamma, got, expect)


def test_overdamped_limit_is_lorentzian():
    """For Gamma >> omega0 the response is quasi-elastic with half-width
    kappa = omega0^2 / (2 Gamma) (Gamma is the HWHM convention)."""
    T, omega0 = 300.0, 5.0
    Gamma = 20.0 * omega0
    kappa = omega0 ** 2 / (2.0 * Gamma)
    w = np.linspace(-3.0 * kappa, 3.0 * kappa, 2001)
    S = dho(w, omega0, Gamma, 0.0, T, 1.0)
    lor = 1.0 / (1.0 + (w / kappa) ** 2)
    ratio = (S / S.max()) / lor
    assert np.allclose(ratio, ratio[len(ratio) // 2], rtol=1e-2), (
        float(ratio.min()), float(ratio.max()))


def test_bose_has_no_spurious_hbar_factor():
    """bose(w, T) inverts to exactly w / (k_B T), with no unit conversion."""
    for T in (150.0, 400.0):
        for omega in (2.0, 5.0, 10.0):
            n = float(bose(np.asarray(omega, dtype=float), T))
            assert np.isclose(np.log(1.0 + 1.0 / n), omega / (K_B * T), rtol=1e-10)


# --------------------------------------------------------------------------- #
# v10: detailed balance for the ASSEMBLED intrinsic spectrum (DHOs + central
# peak), before resolution and background. v7-v9 added an even Lorentzian with
# no thermal factor, which breaks the relation wherever the central peak is on.
# --------------------------------------------------------------------------- #

def _assembled(omega, T, om0=5.0, gm=1.5):
    """All three DHOs plus the central peak, as generate_spectrum assembles them."""
    from sbc.data.spectrum_generator import dho, central_peak, _MODES
    S = dho(omega, om0, gm, 0.0, T, _MODES["soft"]["F2"])
    S = S + dho(omega, _MODES["acoustic"]["omega"], _MODES["acoustic"]["Gamma"],
                0.0, T, _MODES["acoustic"]["F2"])
    S = S + dho(omega, _MODES["optical"]["omega"], _MODES["optical"]["Gamma"],
                0.0, T, _MODES["optical"]["F2"])
    return S + central_peak(omega, T)


def test_assembled_detailed_balance_with_central_peak():
    """S(+w)/S(-w) = exp(w/k_BT) for the assembled spectrum, central peak on and off.

    The gate is |T - T_C| < 50 K with T_C = 395, so the central peak is ACTIVE at
    395 and 420 K and INACTIVE at 150 and 450 K.
    """
    import numpy as np
    from sbc.data.spectrum_generator import K_B, T_C, central_peak
    ws = np.array([0.25, 0.5, 1.0, 2.5, 4.0, 7.0, 9.0])
    for T in (150.0, 395.0, 420.0, 450.0):
        on = abs(T - T_C) < 50.0
        assert (central_peak(ws, T).max() > 0) == on, f"gate wrong at T={T}"
        r = _assembled(ws, T) / _assembled(-ws, T)
        np.testing.assert_allclose(r, np.exp(ws / (K_B * T)), rtol=1e-10,
                                   err_msg=f"detailed balance fails at T={T}, cp_on={on}")


def test_assembled_continuity_at_zero():
    """No STEP at the elastic line: S(+d) - S(-d) -> 0 linearly in d.

    Detailed balance means S is not even, so the correct target is not zero
    difference but the difference detailed balance demands, exp(d/k_BT) - 1,
    vanishing linearly as d -> 0. The v7-v9 even-Lorentzian central peak left a
    finite step instead, which no choice of d removes.
    """
    import numpy as np
    from sbc.data.spectrum_generator import K_B
    for T in (150.0, 395.0, 420.0):
        prev = None
        for d in (1e-3, 1e-4, 1e-5):
            a = _assembled(np.array([d]), T)[0]
            b = _assembled(np.array([-d]), T)[0]
            rel = abs(a - b) / abs(a)
            # dividing by S(+d): 1 - S(-d)/S(+d) = 1 - exp(-d/k_BT)
            np.testing.assert_allclose(rel, -np.expm1(-d / (K_B * T)), rtol=1e-9,
                                       err_msg=f"not the detailed-balance step at T={T}")
            if prev is not None:
                assert rel < prev / 5.0, f"difference not vanishing linearly at T={T}"
            prev = rel


def test_central_peak_zero_limit_and_width():
    """S_cp(0) is finite and matches the previous even-Lorentzian peak value."""
    import numpy as np
    from sbc.data.spectrum_generator import central_peak, T_C, PI
    for T in (380.0, 395.0, 420.0):
        I_c = 600.0 * np.exp(-abs(T - T_C) / 30.0)
        legacy_peak = I_c * 1.5 / (PI * 1.5 ** 2)
        got = float(central_peak(np.array([0.0]), T)[0])
        np.testing.assert_allclose(got, legacy_peak, rtol=1e-12)


def test_central_peak_is_odd_over_bose():
    """chi'' implied by the central peak is odd: S(w)/[n(w)+1] must flip sign."""
    import numpy as np
    from sbc.data.spectrum_generator import central_peak, K_B
    T = 395.0
    w = np.array([0.3, 1.0, 3.0])
    nb1 = lambda x: 1.0 / (-np.expm1(-x / (K_B * T)))
    chi_p = central_peak(w, T) / nb1(w)
    chi_m = central_peak(-w, T) / nb1(-w)
    np.testing.assert_allclose(chi_p, -chi_m, rtol=1e-10)
