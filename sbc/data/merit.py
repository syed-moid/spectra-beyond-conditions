"""Coherence merit function M for INS-derived soft modes in BaTiO3.

This module ports the composite merit formula from
`data/raw/scripts_and_data/merit.py`, which is the implementation that
produced the published figures (e.g., Figure 5 / Table 4) in the
Moid (Discover Applied Sciences, 2025) paper.

Discrepancy with the published Eq. 22.
    The paper text states M = [omega / (Gamma + Gamma_min)]^2, a simple
    quality factor. The published *figures*, however, were produced with
    the elaborate composite below:

        Q             = omega / Gamma
        F_proximity   = exp(-alpha * |T - T_C| / T_C)
        F_thermal     = exp(-omega / (2 * k_B * T))
        F_coherence   = 1 / (1 + (Gamma / (omega / 2))^2)
        F_field       = 1 + 0.05 * E / (1 + E)
        M             = Q * F_proximity * F_thermal * F_coherence * F_field

    We adopt the composite as the source of truth for Paper 1 because
    (a) it is what produced the published figures, (b) it captures
    physical thermal-occupation and Curie-proximity effects that the
    bare quality factor does not, and (c) it gives us a free fidelity
    check: regenerating M at the 36 calibration points should reproduce
    Figure 5 within rounding.

Units.
    omega_Q, Gamma_Q : meV
    T                : K
    E                : kV/cm
"""

from __future__ import annotations

import numpy as np

K_B_meV_per_K: float = 0.08617  # Boltzmann constant in meV/K
T_C_K: float = 395.0            # BaTiO3 Curie temperature
ALPHA_DEFAULT: float = 2.0      # Curie-proximity decay constant


def merit(
    omega_Q_meV: float,
    Gamma_Q_meV: float,
    T_K: float,
    E_kVcm: float = 0.0,
    Tc_K: float = T_C_K,
    alpha: float = ALPHA_DEFAULT,
) -> float:
    """Composite coherence merit M for a single soft-mode condition.

    Returns 0 when Gamma_Q_meV <= 0 (degenerate / unphysical input).
    """
    if Gamma_Q_meV <= 0.0:
        return 0.0

    Q = omega_Q_meV / Gamma_Q_meV
    tau = (T_K - Tc_K) / Tc_K
    F_proximity = np.exp(-alpha * abs(tau))
    F_thermal = np.exp(-omega_Q_meV / (2.0 * K_B_meV_per_K * T_K))
    F_coherence = 1.0 / (1.0 + (Gamma_Q_meV / (omega_Q_meV / 2.0)) ** 2)
    F_field = 1.0 + 0.05 * E_kVcm / (1.0 + E_kVcm)

    return float(Q * F_proximity * F_thermal * F_coherence * F_field)
