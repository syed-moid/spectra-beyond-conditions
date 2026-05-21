"""Augmentation pipeline for INS spectra.

Two-stage design (since Session 5).
    1. `sample_aug_params(state, config, rng) -> dict` draws each
       augmentation's parameters at *severity = 1* (the realistic
       distribution from the YAML). The returned dict is JSON-
       serializable and is what we persist in the .npz dataset.
    2. `apply_aug_params(state, params, severity, noise_rng) -> state`
       applies the augmentation pipeline at any severity, scaling the
       stored s=1 parameters per the Session 5 scaling rules. Severity
       is therefore a runtime/Dataset-config knob, not baked-in data.

Severity scaling rules (Session 5; see SESSION_LOG for physics).
    - Resolution sigma: sigma(s) = sigma(1) * sqrt(s)
        Physics: observed FWHM^2 ~ intrinsic^2 + (k*sigma)^2; growing
        the resolution contribution to observed broadening linearly
        with severity means sigma^2 grows linearly.
    - Poisson N_counts: N(s) = N(1) / s, equivalent to
        noise_amplitude(s) = noise_amplitude(1) * sqrt(s)
        Physics: noise std per bin ~ 1/sqrt(N), so SNR drops as sqrt(s).
    - Background magnitude: linear, bg(s) = s * bg(s=1)
    - Anharmonic skew: linear, skew(s) = s * skew(s=1), clamped +-0.95
        (math-defensive: at |skew| -> 1 the asymmetric Gamma_eff goes
        to 0 inside the omega_grid, dividing-by-zero in dho).
    - Central-peak width multiplier: 1 + s * (mult(1) - 1), floor 0.05
        (identity is mult=1, not mult=0; scale the deviation from
        identity linearly).
    - Zero offset: linear, delta(s) = delta(1) * s
    - Shape parameters (eta_lorentz, bg_type, bg-shape decay/slope
        sub-params) are NOT severity-scaled.

Pipeline order (matches physical data flow, unchanged from Task 2).
    Cat II (intrinsic signal off-manifold):
        1. anharmonic_skew
        2. central_peak_width_perturbation
    Cat I (measurement chain):
        3. instrument_zero_offset
        4. instrument_resolution
        5. poisson_noise
        6. sloping_background

Phase 2 stubs (raise NotImplementedError): peak_overlap, missing_bins,
multi_mode_interference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .spectrum_generator import (
    DEFAULT_OMEGA_GRID,
    bose,
    central_peak,
    dho,
    omega_Q,
    Gamma_Q,
    soft_mode_F2,
    generate_spectrum,
)

SKEW_CLAMP: float = 0.95
CP_WIDTH_MULT_FLOOR: float = 0.05


# --------------------------------------------------------------------- #
# State                                                                 #
# --------------------------------------------------------------------- #

@dataclass
class SpectrumState:
    omega_grid: np.ndarray
    spectrum: np.ndarray
    modes: dict[str, np.ndarray]
    omega_Q: float
    Gamma_Q: float
    T_K: float
    c_pct: float
    E_kVcm: float
    aug_log: list[dict] = field(default_factory=list)


def state_from_clean(
    T_K: float, c_pct: float, E_kVcm: float,
    omega_grid: np.ndarray | None = None,
    latent=None,
) -> SpectrumState:
    """Reconstruct the clean SpectrumState (with per-mode decomposition) that
    augmentations operate on.

    Session 8: `latent` (a LatentDraw or None) is threaded into
    generate_spectrum so the REALIZED clean spectrum/modes are reconstructed
    for Phase 1 data. None reproduces the Phase-0 baseline (back-compat).
    """
    clean = generate_spectrum(T_K, c_pct, E_kVcm, omega_grid, latent=latent)
    return SpectrumState(
        omega_grid=clean["omega_grid"],
        spectrum=clean["spectrum"].copy(),
        modes={k: v.copy() for k, v in clean["modes"].items()},
        omega_Q=clean["omega_Q"],
        Gamma_Q=clean["Gamma_Q"],
        T_K=T_K, c_pct=c_pct, E_kVcm=E_kVcm,
    )


def state_from_clean_array(
    clean_spectrum: np.ndarray,
    T_K: float, c_pct: float, E_kVcm: float,
    omega_grid: np.ndarray | None = None,
) -> SpectrumState:
    """Fast path used by the Dataset: re-derive per-mode breakdowns
    from (T, c, E) so anharmonic_skew and central_peak_width_perturbation
    can regenerate their target modes.
    """
    return state_from_clean(T_K, c_pct, E_kVcm, omega_grid)


# --------------------------------------------------------------------- #
# Pseudo-Voigt kernel (shared helper)                                   #
# --------------------------------------------------------------------- #

def _pseudo_voigt_kernel(grid_step: float, sigma: float, eta: float, n_points: int = 401) -> np.ndarray:
    half = (n_points // 2) * grid_step
    x = np.linspace(-half, half, n_points)
    gauss = np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    gamma_L = sigma * np.sqrt(2.0 * np.log(2.0))
    lorz = (gamma_L / np.pi) / (x ** 2 + gamma_L ** 2)
    kernel = (1.0 - eta) * gauss + eta * lorz
    kernel /= kernel.sum() * grid_step
    return kernel


# --------------------------------------------------------------------- #
# Stage 1: sample severity=1 parameters                                 #
# --------------------------------------------------------------------- #

def sample_aug_params(
    state: SpectrumState,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Sample one realization of all augmentation params at severity=1.

    The output is JSON-serializable. `noise_seed` is included so that
    Poisson noise replay does not consume the caller's RNG.
    """
    params: dict[str, Any] = {}

    # --- Cat II ---
    skew_cfg = config.get("anharmonic_skew", {})
    skew_bound = float(skew_cfg.get("realistic_skew_bound", 0.3))
    params["skew"] = float(rng.uniform(-skew_bound, +skew_bound))

    cp_cfg = config.get("central_peak_width_perturbation", {})
    if abs(state.T_K - 395.0) < 50.0:
        m_lo = float(cp_cfg.get("realistic_multiplier_min", 0.5))
        m_hi = float(cp_cfg.get("realistic_multiplier_max", 2.5))
        params["cp_width_mult"] = float(rng.uniform(m_lo, m_hi))
    else:
        params["cp_width_mult"] = None  # central peak inactive at this T

    # --- Cat I ---
    zo_cfg = config.get("instrument_zero_offset", {})
    d_bound = float(zo_cfg.get("realistic_delta_meV", 0.5))
    params["zero_offset_meV"] = float(rng.uniform(-d_bound, +d_bound))

    res_cfg = config.get("instrument_resolution", {})
    a_max = float(res_cfg.get("realistic_sigma_const_max", 0.8))
    b_max = float(res_cfg.get("realistic_sigma_lin_max", 0.06))
    eta_lo, eta_hi = res_cfg.get("eta_range", [0.1, 0.5])
    params["res_sigma_const_meV"] = float(rng.uniform(0.0, a_max))
    params["res_sigma_lin"] = float(rng.uniform(0.0, b_max))
    params["res_eta"] = float(rng.uniform(eta_lo, eta_hi))

    n_cfg = config.get("poisson_noise", {})
    amp_max = float(n_cfg.get("realistic_noise_amplitude_max", 0.02))
    params["noise_amplitude"] = float(rng.uniform(0.0, amp_max))
    params["noise_seed"] = int(rng.integers(0, 2 ** 63 - 1))

    bg_cfg = config.get("sloping_background", {})
    mag_max = float(bg_cfg.get("realistic_magnitude_max", 0.30))
    mag_s1 = float(rng.uniform(0.0, mag_max))
    bg_types = bg_cfg.get("bg_types", ["linear", "poly2", "exp"])
    bg_type = str(rng.choice(bg_types))
    if bg_type == "linear":
        slope_max = float(bg_cfg.get("linear_slope_max_per_meV", 0.05))
        bg_shape = {
            "slope_per_meV": float(rng.uniform(-slope_max, +slope_max)),
            "intercept_frac": float(rng.uniform(0.0, mag_s1)),
        }
    elif bg_type == "poly2":
        bg_shape = {
            "a0": float(rng.uniform(0.0, mag_s1)),
            "a1": float(rng.uniform(-mag_s1, +mag_s1)) / 15.0,
            "a2": float(rng.uniform(-mag_s1, +mag_s1)) / 225.0,
        }
    else:  # exp
        tau_lo, tau_hi = bg_cfg.get("exp_tau_range_meV", [3.0, 15.0])
        bg_shape = {
            "A_frac": float(rng.uniform(0.0, mag_s1)),
            "tau_meV": float(rng.uniform(tau_lo, tau_hi)),
        }
    params["bg_type"] = bg_type
    params["bg_shape"] = bg_shape
    params["bg_magnitude_s1"] = mag_s1  # for inspection; not used at apply time

    return params


# --------------------------------------------------------------------- #
# Stage 2: apply at any severity                                        #
# --------------------------------------------------------------------- #

def _apply_anharmonic_skew(state: SpectrumState, skew_s1: float, severity: float) -> None:
    s_skew = max(-SKEW_CLAMP, min(SKEW_CLAMP, skew_s1 * severity))
    if s_skew == 0.0:
        state.aug_log.append({"name": "anharmonic_skew", "skew": 0.0, "applied": False})
        return
    omega = state.omega_grid
    Gamma_eff = state.Gamma_Q * (1.0 + s_skew * np.tanh((omega - state.omega_Q) / state.Gamma_Q))
    # Session 8 Delta-omega consolidation: pass delta=0.0 (the E^2 shift lives
    # only in state.omega_Q). Previously this passed Delta_omega(E), which made
    # soft_new use a shifted denominator while the subtracted baseline
    # state.modes["soft"] (clean path, now delta=0) did not — contaminating the
    # skew delta at E!=0. At E=0 (Delta_omega(0)=0) this is byte-identical.
    soft_new = dho(omega, state.omega_Q, Gamma_eff, 0.0, state.T_K, soft_mode_F2())
    delta = soft_new - state.modes["soft"]
    state.modes["soft"] = soft_new
    state.spectrum = state.spectrum + delta
    state.aug_log.append({"name": "anharmonic_skew", "skew": s_skew, "applied": True})


def _apply_central_peak_width(state: SpectrumState, mult_s1: float | None, severity: float) -> None:
    if mult_s1 is None:
        state.aug_log.append({"name": "central_peak_width_perturbation", "applied": False})
        return
    mult_s = max(CP_WIDTH_MULT_FLOOR, 1.0 + severity * (mult_s1 - 1.0))
    new_width = 1.5 * mult_s
    cp_new = central_peak(state.omega_grid, state.T_K, width_meV=new_width)
    delta = cp_new - state.modes["central"]
    state.modes["central"] = cp_new
    state.spectrum = state.spectrum + delta
    state.aug_log.append({
        "name": "central_peak_width_perturbation",
        "width_multiplier": mult_s,
        "new_width_meV": new_width,
        "applied": True,
    })


def _apply_zero_offset(state: SpectrumState, delta_s1: float, severity: float) -> None:
    delta_s = delta_s1 * severity
    if delta_s == 0.0:
        state.aug_log.append({"name": "instrument_zero_offset", "applied": False})
        return
    shifted = np.interp(state.omega_grid - delta_s, state.omega_grid, state.spectrum, left=0.0, right=0.0)
    state.spectrum = shifted
    state.aug_log.append({"name": "instrument_zero_offset", "delta_meV": delta_s, "applied": True})


def _apply_resolution(
    state: SpectrumState,
    sigma_const_s1: float,
    sigma_lin_s1: float,
    eta: float,
    severity: float,
) -> None:
    sqrt_s = float(np.sqrt(severity))
    sigma_const_s = sigma_const_s1 * sqrt_s
    sigma_lin_s = sigma_lin_s1 * sqrt_s
    sigma_eff = sigma_const_s + sigma_lin_s * float(np.abs(state.omega_grid).mean())
    if sigma_eff <= 1e-3:
        state.aug_log.append({"name": "instrument_resolution", "applied": False})
        return
    grid_step = float(state.omega_grid[1] - state.omega_grid[0])
    kernel = _pseudo_voigt_kernel(grid_step, sigma_eff, eta)
    state.spectrum = np.convolve(state.spectrum, kernel, mode="same") * grid_step
    state.aug_log.append({
        "name": "instrument_resolution",
        "sigma_const_meV": sigma_const_s,
        "sigma_lin": sigma_lin_s,
        "sigma_eff_meV": sigma_eff,
        "eta_lorentz": eta,
        "applied": True,
    })


def _apply_poisson_noise(
    state: SpectrumState,
    noise_amp_s1: float,
    noise_seed: int,
    severity: float,
    noise_rng: np.random.Generator | None,
) -> None:
    noise_amp_s = noise_amp_s1 * float(np.sqrt(severity))
    if noise_amp_s <= 0.0:
        state.aug_log.append({"name": "poisson_noise", "applied": False})
        return
    peak = float(np.max(state.spectrum))
    if peak <= 0.0:
        state.aug_log.append({"name": "poisson_noise", "applied": False, "reason": "non-positive peak"})
        return
    N_peak_s = int(np.clip(1.0 / noise_amp_s ** 2, 10, 1e9))
    scale = N_peak_s / peak
    if noise_rng is None:
        noise_rng = np.random.default_rng(noise_seed)
    counts = noise_rng.poisson(np.clip(state.spectrum * scale, 0, None))
    state.spectrum = counts.astype(np.float64) / scale
    state.aug_log.append({
        "name": "poisson_noise",
        "noise_amplitude": noise_amp_s,
        "N_peak": N_peak_s,
        "noise_seed": noise_seed,
        "applied": True,
    })


def _apply_sloping_background(
    state: SpectrumState,
    bg_type: str,
    bg_shape: dict[str, float],
    severity: float,
) -> None:
    if severity <= 0.0:
        state.aug_log.append({"name": "sloping_background", "applied": False})
        return
    peak = float(np.max(state.spectrum)) or 1.0
    omega = state.omega_grid
    if bg_type == "linear":
        bg = peak * (bg_shape["intercept_frac"] + bg_shape["slope_per_meV"] * omega)
    elif bg_type == "poly2":
        bg = peak * (bg_shape["a0"] + bg_shape["a1"] * omega + bg_shape["a2"] * omega ** 2)
    else:  # exp
        bg = peak * bg_shape["A_frac"] * np.exp(-np.abs(omega) / bg_shape["tau_meV"])
    bg = np.clip(bg * severity, 0.0, None)
    if not np.any(bg > 0.0):
        state.aug_log.append({"name": "sloping_background", "applied": False})
        return
    state.spectrum = state.spectrum + bg
    state.aug_log.append({
        "name": "sloping_background",
        "bg_type": bg_type,
        "bg_shape": bg_shape,
        "severity": severity,
        "applied": True,
    })


def apply_aug_params(
    state: SpectrumState,
    params: dict[str, Any],
    severity: float,
    noise_rng: np.random.Generator | None = None,
) -> SpectrumState:
    """Apply the full augmentation pipeline at `severity`.

    `params` is the dict produced by `sample_aug_params` (severity=1
    canonical realization). `noise_rng` overrides the seeded RNG used
    for Poisson noise; pass None to use a fresh RNG seeded from
    `params["noise_seed"]` (the standard replay path).
    """
    if severity <= 0:
        return state  # full identity

    _apply_anharmonic_skew(state, params["skew"], severity)
    _apply_central_peak_width(state, params.get("cp_width_mult"), severity)
    _apply_zero_offset(state, params["zero_offset_meV"], severity)
    _apply_resolution(
        state,
        params["res_sigma_const_meV"], params["res_sigma_lin"], params["res_eta"],
        severity,
    )
    _apply_poisson_noise(
        state,
        params["noise_amplitude"], int(params["noise_seed"]), severity, noise_rng,
    )
    _apply_sloping_background(state, params["bg_type"], params["bg_shape"], severity)
    return state


# --------------------------------------------------------------------- #
# Backwards-compatibility shim (used by sanity_check_shirane.py)        #
# --------------------------------------------------------------------- #

def apply_pipeline(
    state: SpectrumState,
    config: dict[str, Any],
    rng: np.random.Generator,
    global_severity: float = 1.0,
) -> SpectrumState:
    """Sample s=1 params from `config` and apply at `global_severity`.

    Retained so the existing `scripts/sanity_check_shirane.py` keeps
    working unchanged. New code should call `sample_aug_params` and
    `apply_aug_params` directly.
    """
    params = sample_aug_params(state, config, rng)
    return apply_aug_params(state, params, global_severity, noise_rng=rng)


# --------------------------------------------------------------------- #
# Phase 2 stubs                                                         #
# --------------------------------------------------------------------- #

def peak_overlap(*args, **kwargs):
    raise NotImplementedError("peak_overlap deferred to Phase 2")


def missing_bins(*args, **kwargs):
    raise NotImplementedError("missing_bins deferred to Phase 2")


def multi_mode_interference(*args, **kwargs):
    raise NotImplementedError("multi_mode_interference deferred to Phase 2")
