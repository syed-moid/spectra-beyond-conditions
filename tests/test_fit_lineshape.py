"""The fitting models must obey the same detailed balance as the generator.

Until round 49 the single-mode fitting baselines used the v7-era lineshape the
generator itself abandoned at v9: the occupation factor evaluated at omega0 and
applied as a step at w = 0, and a denominator using 4*omega0^2*gamma^2 instead
of 4*gamma^2*w^2. Both of these fail the tests below, which is the point of
having them: the fits were being scored against spectra they could not represent.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "manuscript_phase1"))
sys.path.insert(0, str(ROOT / "review_and_modify" / "scripts"))

import a3_fitting_baselines as A3           # noqa: E402
from src.data.merit import K_B_meV_per_K    # noqa: E402

GRID = np.linspace(-15.0, 15.0, 600)


@pytest.mark.parametrize("om0", [2.0, 5.0, 11.0])
@pytest.mark.parametrize("T", [150.0, 400.0])
@pytest.mark.parametrize("w", [1.0, 2.5, 4.0, 7.0, 9.0])
def test_detailed_balance_off_resonance(om0, T, w):
    """S(+w)/S(-w) = exp(w / kB T), at energies away from the mode."""
    if abs(w - om0) < 1e-9:
        pytest.skip("evaluated at the mode frequency")
    g = np.array([-w, w])
    S = A3._dho_lineshape(g, 1.0, om0, 0.8, T)
    assert S[0] > 0 and S[1] > 0
    assert np.isclose(S[1] / S[0], np.exp(w / (K_B_meV_per_K * T)), rtol=1e-10)


@pytest.mark.parametrize("T", [150.0, 400.0])
def test_continuous_at_zero(T):
    """No step at the elastic line: S(+d)/S(-d) -> 1 as d -> 0."""
    for d in (1e-3, 1e-4, 1e-5):
        S = A3._dho_lineshape(np.array([-d, d]), 1.0, 5.0, 0.8, T)
        assert np.isclose(S[1] / S[0], np.exp(d / (K_B_meV_per_K * T)), rtol=1e-9)


def test_zero_limit_is_finite_and_positive():
    """occ(0) = kB T, so S(0) is finite and strictly positive."""
    S = A3._dho_lineshape(np.array([0.0]), 1.0, 5.0, 0.8, 300.0)
    assert np.isfinite(S[0]) and S[0] > 0


def test_denominator_is_the_full_form():
    """4*gamma^2*w^2, not the near-resonance 4*omega0^2*gamma^2.

    The two agree at w = omega0 and diverge away from it; this pins the form at
    a point where they differ by a large factor.
    """
    om0, gm, T, w = 5.0, 0.8, 300.0, 12.0
    got = A3._dho_lineshape(np.array([w]), 1.0, om0, gm, T)[0]
    x = w / (K_B_meV_per_K * T)
    occ = w / -np.expm1(-x)
    want = occ * gm / ((w ** 2 - om0 ** 2) ** 2 + 4.0 * gm ** 2 * w ** 2)
    assert np.isclose(got, want, rtol=1e-12)
    near_resonance = occ * gm / ((w ** 2 - om0 ** 2) ** 2 + 4.0 * om0 ** 2 * gm ** 2)
    assert not np.isclose(got, near_resonance, rtol=1e-3)


def test_matched_model_parameter_indices():
    """Guards the index table in model_matched's docstring (round 49 item 1)."""
    import inspect
    src = inspect.getsource(A3.model_matched)
    assert "om0, g_soft, g_ac, g_op, cp_w, dz, sig, a0, a1, scale = p" in src
    # index 5 shifts the energy axis; index 8 multiplies the grid in the background
    assert "g = grid - dz" in src
    assert "pk * (a0 + a1 * grid)" in src
