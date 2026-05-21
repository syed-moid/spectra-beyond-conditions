"""Hybrid INS spectrum generator for BaTiO3 soft-mode region.

Direct port of `data/raw/scripts_and_data/BaTiO₃HybridINSPlots.py`.
The hybrid model combines:

  * a DFT-fitted soft-mode dispersion omega_Q(T) interpolated cubically
    from a 6-point database, with c- and E-dependent corrections;
  * a soft-mode linewidth Gamma_Q(T, c, E) whose T-dependence is
    cubically interpolated from the 6-point _GAMMA_DB table (matching
    the values used by the published Figure 5 / Tables 4, S2, S3),
    with c- and E-dependent corrections from Eq. 21 applied on top;
  * fixed acoustic-TA (9.2 meV) and optical-TO2 (22.0 meV) modes;
  * a near-Tc central-peak term active only when |T - T_C| < 50 K.

Notes on Gamma_Q (scope-affecting findings).
    Session 2 (T-dependence). BaTiO3HybridINSPlots.py defines two
    non-equivalent representations of the soft-mode linewidth:
    (a) the tabulated _GAMMA_DB (defined but never invoked in the
    source, used implicitly by the merit-plot code in merit.py and
    matching the paper's quantitative tables), and (b) a parametric
    C1*T + C2*T**2 + ... formula that the source actually feeds into
    the spectrum plotter, which over-estimates the tabulated values
    by 1.9-2.3x for T >= 200 K. We adopt the tabulated form (option
    `_GAMMA_DB`) so that our (spectrum, M) pairs are mutually
    consistent and reproduce the paper's tables.

    Session 4 (E-dependence). The published E-correction
    (C3 + C4*T)*E**2 is incompatible with the gamma_Hybrid table,
    which shows Delta-Gamma vs E is T-independent at ~0.08 meV across
    all (T, c). We refit to `C3 * E**2` with C3 = 0.020625
    (T-independent, matching the data). The original parametric
    Gamma form (including the original C3, C4) is preserved as
    `_gamma_parametric_deprecated` for traceability.

The soft mode F^2 is calibrated once at module import so that the
total spectrum at (T=300, c=0, E=0) peaks at ~698,000 (the published
intensity scale). The same scale factor is applied to the acoustic
and optical mode F^2 values.

Units.
    T         : K
    c         : percent (0..2 in training) at the public API boundary;
                converted internally to a fraction before entering the
                damping / frequency formulas, which were DFT-fit with
                c as a fraction (e.g., 2% -> 0.02).
    E         : kV/cm
    omega     : meV

TODO(physics): the central-peak intensity uses a hard cutoff at
|T - T_C| < 50 K, matching the published implementation. A smoothly
decaying envelope (e.g., Gaussian centered at T_C) would be more
physical and would remove the step discontinuity in M(T) at the
cutoff boundary. Revisit when validating Paper 1 figures.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.interpolate import interp1d

from .merit import merit
from .latent_perturbations import LatentDraw

# --------------------------------------------------------------------- #
# Constants                                                             #
# --------------------------------------------------------------------- #

K_B: float = 8.617333262145e-2   # meV/K
HBAR: float = 0.6582119569       # eV*fs (used as in the published code)
PI: float = np.pi
T_C: float = 395.0               # BaTiO3 Curie temperature, K

# DFT database used by the hybrid model.
_T_DB = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
_OMEGA_DB = np.array([11.2, 9.8, 8.2, 3.5, 5.8, 6.5])
_GAMMA_DB = np.array([0.60, 0.75, 1.05, 1.80, 2.50, 3.00])

_omega_interp = interp1d(_T_DB, _OMEGA_DB, kind="cubic", fill_value="extrapolate")
_gamma_interp = interp1d(_T_DB, _GAMMA_DB, kind="cubic", fill_value="extrapolate")

# Damping coefficients (Eq. 21 corrections layered on _gamma_interp).
#
# History:
#   _C1, _C2 — deprecated in Session 2; the parametric T-form was
#     retired in favor of _gamma_interp. Kept inside
#     `_gamma_parametric_deprecated` for traceability.
#   _C3      — refit in Session 4 (this file's git history) against the
#     12-point Delta-Gamma vs E table from merit.py. Old value 1.0e-3
#     gave Delta-Gamma at E=2 about 10-20x smaller than the published
#     table; new value 0.020625 matches the table to within rounding
#     noise (max residual 0.0175 meV at T=100, c=0 outlier; median
#     0.0025 meV). See SESSION_LOG.md Session 4.
#   _C4      — REMOVED in Session 4. The published form was
#     (C3 + C4*T)*E^2 but the calibration table shows Delta-Gamma vs E
#     is T-independent at ~0.08 meV across all (T, c). The data does
#     not support a C4*T term; including it would add a fitted constant
#     that the data drives to ~0. See SESSION_LOG.md Session 4.
_C1, _C2 = 4.8e-3, 1.1e-5
_C3 = 0.020625
_C4_DEPRECATED_REMOVED_SESSION_4 = None  # was 2.0e-6; see history above

# Default omega grid for the soft-mode region (meV).
DEFAULT_OMEGA_GRID = np.linspace(-15.0, 15.0, 600)


# --------------------------------------------------------------------- #
# Physics                                                               #
# --------------------------------------------------------------------- #

def Gamma_Q(T: float, c_frac: float, E: float) -> float:
    """Soft-mode linewidth (meV).

    Production form. T-dependence is cubic interpolation of _GAMMA_DB
    (matching the published Tables 4 / S2 / S3 and Figure 5). The
    defect term `0.6 * c * T` is from Eq. 21 and reproduces the table
    exactly. The E-correction is the refit form from Session 4:

        Gamma_Q(T, c, E) = gamma_interp(T) + 0.6 * c * T + C3 * E**2

    `c_frac` is a fraction (2% -> 0.02).

    E-correction history (resolved in Session 4):
        The published form was `(C3 + C4*T)*E**2`. Inspecting the
        12-point Delta-Gamma vs E table from merit.py showed the
        E-correction is T-independent at ~0.08 meV across T=100-600 K
        and both c values. We refit to `C3 * E**2` (dropping the C4*T
        term the data does not support) with C3 = 0.020625, giving
        Delta-Gamma agreement to median 0.0025 meV across the 12-point
        table. Calibration-grid M residuals at E=2 dropped from 3-13%
        (pre-refit) to <1% median post-refit. See SESSION_LOG.md
        Session 4 for the full chain.
    """
    return float(_gamma_interp(T)) + 0.6 * c_frac * T + _C3 * E ** 2


def _gamma_parametric_deprecated(T: float, c_frac: float, E: float) -> float:
    """Original parametric Gamma_Q from BaTiO3HybridINSPlots.py — NOT USED.

    Retained for traceability. This is the form the published source
    script actually fed into its spectrum plotter, but it disagrees
    with the tabulated _GAMMA_DB / gamma_Hybrid table that drives
    Figure 5 (overestimates by ~1.9-2.3x for T >= 200 K). We adopt
    the tabulated form in production; see module docstring.

    The E-term uses the *original* published C3 = 1.0e-3 and
    C4 = 2.0e-6 so this function reproduces the published parametric
    output exactly; the production `Gamma_Q` uses the Session-4 refit
    (C3 = 0.020625, no C4 term).
    """
    _C3_published = 1.0e-3
    _C4_published = 2.0e-6
    return (
        _C1 * T + _C2 * T ** 2
        + 0.6 * c_frac * T
        + (_C3_published + _C4_published * T) * E ** 2
    )


def omega_Q(T: float, c_frac: float, E: float) -> float:
    """Soft-mode frequency (meV), floored at 0.5. `c_frac` is a fraction."""
    base = float(_omega_interp(T))
    return max(0.5, base * (1.0 - 0.08 * c_frac) + 0.01 * E ** 2)


def Delta_omega(E: float) -> float:
    """E-field-induced frequency shift (meV). DEPRECATED (Session 8).

    The E^2 field shift is now carried solely by omega_Q() (which already
    includes +0.01*E^2). generate_spectrum() no longer passes a separate
    Delta into the DHO denominator — that double-counted the shift (it
    appeared once in omega_Q via omega0^2 and again via the 2*omega0*Delta
    term, partly cancelling the peak shift while distorting the lineshape).
    Retained only so the historical second call site (anharmonic_skew in
    augmentations.py) still imports cleanly; see SESSION_LOG Session 8.
    New code should NOT reintroduce a separate Delta channel.
    """
    return 0.01 * E ** 2


def bose(omega: np.ndarray, T: float) -> np.ndarray:
    """Bose-Einstein occupation factor (public; used by augmentations)."""
    return 1.0 / (np.exp(HBAR * np.abs(omega) / (K_B * T)) - 1.0 + 1e-12)


def dho(
    omega: np.ndarray,
    omega0: float,
    Gamma: float | np.ndarray,
    Delta: float,
    T: float,
    F2: float,
) -> np.ndarray:
    """Damped-harmonic-oscillator lineshape with detailed-balance weighting.

    One-phonon dynamic structure factor. The Bose-Einstein occupation
    factor is evaluated at the phonon frequency `omega0` (not at the
    spectral variable `omega`); the lineshape itself is a pure Lorentzian
    in `omega`. This is the correct form for a one-phonon S(Q, omega):
    Stokes side (omega > 0, energy loss) carries weight [n(omega0) + 1],
    anti-Stokes side (omega < 0, energy gain) carries n(omega0).

    Source-code finding (Session 3). The published BaTiO3HybridINSPlots.py
    applied `bose(omega, T)` as an envelope on the Lorentzian — i.e.,
    evaluated the Bose factor at the spectral variable. That introduces
    a spurious 1/omega divergence at the elastic line that does not
    exist in real INS data. We correct this here. The published scalar
    outputs (omega_Q, Gamma_Q, M) do not surface the bug because they
    don't depend on the lineshape envelope. See SESSION_LOG.

    Gamma may be a scalar (standard symmetric DHO) or an array with the
    same shape as `omega` (asymmetric / energy-dependent damping; used
    by the anharmonic_skew augmentation).
    """
    n_omega0 = bose(np.asarray(omega0, dtype=float), T)  # scalar at phonon freq
    weight = np.where(np.asarray(omega) >= 0.0, n_omega0 + 1.0, n_omega0)
    denom = (omega ** 2 - omega0 ** 2 + 2.0 * omega0 * Delta) ** 2 + 4.0 * omega0 ** 2 * Gamma ** 2
    return (1.0 / HBAR) * weight * (1.0 / PI) * F2 * (Gamma / denom)


def central_peak(omega: np.ndarray, T: float, width_meV: float = 1.5) -> np.ndarray:
    """Lorentzian central-peak term, active only for |T - T_C| < 50 K.

    `width_meV` is the HWHM of the central-peak Lorentzian (default 1.5,
    matching the published implementation). Exposed so augmentations can
    perturb it.
    """
    if abs(T - T_C) >= 50.0:
        return np.zeros_like(omega)
    I_c = 600.0 * np.exp(-abs(T - T_C) / 30.0)
    return I_c * width_meV / (PI * (omega ** 2 + width_meV ** 2))


# Backward-compatible private aliases retained internally.
_bose = bose
_dho = dho
_central_peak = central_peak


def soft_mode_F2() -> float:
    """Calibrated soft-mode F^2 (read-only; used by augmentations)."""
    return _MODES["soft"]["F2"]


# --------------------------------------------------------------------- #
# Mode table + one-time F^2 calibration                                 #
# --------------------------------------------------------------------- #

_MODES: dict[str, dict] = {
    "soft":     {"F2": 1.0},
    "acoustic": {"F2": 0.8, "omega": 9.2,  "Gamma": 0.6},
    "optical":  {"F2": 0.3, "omega": 22.0, "Gamma": 1.8},
}


def _calibrate_F2() -> None:
    """Scale all mode F^2 so the total at (T=300, c=0, E=0) peaks ~6.98e5.

    Mirrors the published calibrate() routine. Mutates the module-level
    _MODES dict exactly once, at import time.
    """
    T, c, E = 300.0, 0.0, 0.0
    grid = np.linspace(-15.0, 15.0, 800)
    S = np.zeros_like(grid)
    # Calibration uses c as a fraction (matches published calibrate()).
    # Session 8: delta=0.0 (consolidated). At E=0 (calibration condition)
    # Delta_omega(0)=0 anyway, so the calibrated F2 scale is unchanged.
    S += _dho(grid, omega_Q(T, c, E), Gamma_Q(T, c, E), 0.0, T, _MODES["soft"]["F2"])  # c=0 here, no unit ambiguity
    S += _dho(grid, _MODES["acoustic"]["omega"], _MODES["acoustic"]["Gamma"], 0.0, T, _MODES["acoustic"]["F2"])
    S += _dho(grid, _MODES["optical"]["omega"], _MODES["optical"]["Gamma"], 0.0, T, _MODES["optical"]["F2"])
    S += _central_peak(grid, T)
    scale = 698000.0 / float(np.max(S))
    for m in _MODES.values():
        m["F2"] *= scale


_calibrate_F2()


# --------------------------------------------------------------------- #
# Public API                                                            #
# --------------------------------------------------------------------- #

def generate_spectrum(
    T: float,
    c: float,
    E: float,
    omega_grid: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,  # noqa: ARG001 (clean spectra are deterministic)
    latent: Optional[LatentDraw] = None,
) -> dict:
    """Generate a clean hybrid INS spectrum for one (T, c, E) condition.

    Parameters
    ----------
    T : float
        Temperature in K.
    c : float
        Defect concentration in **percent** (0..2 in training).
        Converted to a fraction internally before entering Gamma_Q /
        omega_Q (which were fit with c as a fraction in the source).
    E : float
        Electric field in kV/cm.
    omega_grid : np.ndarray, optional
        Energy-transfer grid in meV. Defaults to np.linspace(-15, 15, 600),
        matching the published scripts.
    rng : np.random.Generator, optional
        Unused for clean generation; reserved so the augmentation
        pipeline (Task 2) can share the same call signature.

    Returns
    -------
    dict with keys:
        spectrum : (N,) ndarray   - total intensity on omega_grid
        modes    : dict[str, ndarray] - per-mode contributions
        omega_Q  : float           - soft-mode frequency (meV)
        Gamma_Q  : float           - soft-mode linewidth (meV)
        M        : float           - coherence merit (composite formula)
        omega_grid : ndarray       - the grid actually used
    """
    if omega_grid is None:
        omega_grid = DEFAULT_OMEGA_GRID

    c_frac = c / 100.0  # public-API percent -> internal fraction
    om = float(omega_Q(T, c_frac, E))
    gm = float(Gamma_Q(T, c_frac, E))
    # Delta-omega consolidation (Session 8): the E^2 field shift lives ONLY in
    # omega_Q now; the separate Delta channel in the DHO denominator is removed
    # (it double-counted the shift). delta=0 here. At E=0 this is byte-identical
    # to the pre-fix code (Delta_omega(0)=0); at E!=0 the soft-mode lineshape
    # changes. M is unaffected (it never depended on the lineshape Delta).
    delta = 0.0

    # Phase 1 latent perturbations (Session 8). latent=None -> clean baseline.
    # When provided: each mode's Gamma is scaled by a per-mode realized factor,
    # and the soft mode's omega is shifted by xi2. acoustic/optical Gamma
    # perturbations add realistic nuisance variability that does NOT enter M
    # (M depends on the soft mode only).
    if latent is not None:
        om_soft = om * latent.omega_multiplier_soft()
        gm_soft = gm * latent.gamma_multiplier("soft")
        gm_ac = _MODES["acoustic"]["Gamma"] * latent.gamma_multiplier("acoustic")
        gm_op = _MODES["optical"]["Gamma"] * latent.gamma_multiplier("optical")
    else:
        om_soft, gm_soft = om, gm
        gm_ac = _MODES["acoustic"]["Gamma"]
        gm_op = _MODES["optical"]["Gamma"]

    S_soft = _dho(omega_grid, om_soft, gm_soft, delta, T, _MODES["soft"]["F2"])
    S_ac = _dho(
        omega_grid, _MODES["acoustic"]["omega"], gm_ac, 0.0, T,
        _MODES["acoustic"]["F2"],
    )
    S_op = _dho(
        omega_grid, _MODES["optical"]["omega"], gm_op, 0.0, T,
        _MODES["optical"]["F2"],
    )
    S_cp = _central_peak(omega_grid, T)
    S_total = S_soft + S_ac + S_op + S_cp

    # M is recomputed from the REALIZED soft-mode (omega, Gamma) via the
    # validated composite merit(). Latents reach M only through these realized
    # parameters, never as a direct term.
    M = merit(om_soft, gm_soft, T, E)

    return {
        "spectrum": S_total,
        "modes": {"soft": S_soft, "acoustic": S_ac, "optical": S_op, "central": S_cp},
        "omega_Q": om_soft,
        "Gamma_Q": gm_soft,
        "M": M,
        "omega_grid": omega_grid,
        "latent": latent.to_dict() if latent is not None else None,
    }
