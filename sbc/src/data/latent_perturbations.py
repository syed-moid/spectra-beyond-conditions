"""Phase 1 latent-microphysics perturbations (Session 8).

Per-mode scalar Gamma perturbation (xi1) + soft-mode scalar omega-shift (xi2).

Design (Session 8 decision; supersedes the Phase-0 q-field design).
    The production generator is 1D I(omega) at the Gamma point (Design.md
    frozen scope: NOT 2D S(Q, omega)); it has no q-axis. The Phase-0
    q-correlated Gamma field therefore reduces honestly to a PER-MODE SCALAR
    perturbation. Per the parameter card formula:

        Gamma_realized[mode] = Gamma_0[mode] * floor(0.05, 1 + alpha * xi1[mode])
        omega0_realized_soft = omega0_soft * (1 + beta * xi2)

    where (per configs/phase1_parameter_card.yaml):
        alpha     ~ lognormal(mu_log=-1.204, sigma_log=0.40)   ONE draw per spectrum
                    (shared amplitude; note `alpha` has no [mode] index in the
                    card formula while xi1[mode] does)
        xi1[mode] ~ N(0, 1)                                     THREE draws per
                    spectrum, one per mode (soft, acoustic, optical), independent
        beta*xi2  ~ skew-normal mapped to [-0.05, +0.07]        ONE draw per spectrum,
                    applied to the soft (TO) mode frequency only

M is NOT perturbed here. It is recomputed downstream from the realized
soft-mode (omega0_realized, gamma_realized) via the validated composite
merit() (src/data/merit.py). The latents reach M only through these realized
parameters — never as a direct additive term. This is the property that makes
the task an inverse problem the spectrum can solve while a conditions-only
model cannot.

RNG: repo convention is np.random.default_rng(seed) (no get_rng() helper
exists). Callers pass a seeded Generator; draw order within sample_latents is
fixed (alpha, xi1[soft], xi1[acoustic], xi1[optical], beta*xi2) for
reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import skewnorm

# ---- Parameter-card constants (configs/phase1_parameter_card.yaml) ----
GAMMA_FLOOR: float = 0.05            # hard floor on the realized Gamma multiplier
ALPHA_MU_LOG: float = -1.204         # ln(0.30)
ALPHA_SIGMA_LOG: float = 0.40
BETA_XI2_LOW: float = -0.05
BETA_XI2_HIGH: float = 0.07
BETA_XI2_SKEW: float = 4.0           # skew-normal shape (positive = hardening)
# Skew-normal -> fractional-shift mapping (brief-specified, approximate band).
_BETA_XI2_SCALE: float = 0.025
_BETA_XI2_LOC: float = 0.01

MODES: tuple[str, ...] = ("soft", "acoustic", "optical")


# --------------------------------------------------------------------- #
# Latent draw container                                                  #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class LatentDraw:
    """One per-spectrum realization of the latent variables.

    alpha     : shared amplitude (one per spectrum)
    xi1       : {mode: N(0,1)} per-mode linewidth latent
    beta_xi2  : realized fractional omega-shift for the soft mode
    """
    alpha: float
    xi1: dict[str, float]
    beta_xi2: float

    def gamma_multiplier(self, mode: str, floor: float = GAMMA_FLOOR) -> float:
        """Realized Gamma multiplier for `mode`: floor(0.05, 1 + alpha * xi1[mode])."""
        return float(max(floor, 1.0 + self.alpha * self.xi1[mode]))

    def omega_multiplier_soft(self) -> float:
        """Realized soft-mode omega multiplier: (1 + beta*xi2)."""
        return 1.0 + self.beta_xi2

    def to_dict(self) -> dict:
        """JSON-serializable record for dataset audit trails."""
        return {
            "alpha": self.alpha,
            "xi1": dict(self.xi1),
            "beta_xi2": self.beta_xi2,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LatentDraw":
        """Reconstruct a LatentDraw from its to_dict() record (replay path)."""
        return cls(
            alpha=float(d["alpha"]),
            xi1={str(k): float(v) for k, v in d["xi1"].items()},
            beta_xi2=float(d["beta_xi2"]),
        )


# --------------------------------------------------------------------- #
# Samplers                                                               #
# --------------------------------------------------------------------- #

def sample_alpha(rng: np.random.Generator,
                 mu_log: float = ALPHA_MU_LOG,
                 sigma_log: float = ALPHA_SIGMA_LOG) -> float:
    """Per-spectrum amplitude alpha ~ lognormal(mu_log, sigma_log). Median e^mu_log."""
    return float(rng.lognormal(mean=mu_log, sigma=sigma_log))


def sample_beta_xi2(rng: np.random.Generator,
                    low: float = BETA_XI2_LOW,
                    high: float = BETA_XI2_HIGH,
                    skew: float = BETA_XI2_SKEW) -> float:
    """Realized soft-mode fractional omega-shift, skew-normal mapped to [low, high].

    Mapping (brief-specified, approximate): raw ~ skewnorm(a=skew); shifted and
    scaled by (_BETA_XI2_LOC, _BETA_XI2_SCALE) then hard-clipped to [low, high].
    Positive skew encodes the physical hardening bias (oxygen-vacancy / eight-site
    disorder harden the TO mode more often than they soften it).
    """
    raw = float(skewnorm.rvs(a=skew, random_state=rng))
    return float(np.clip(raw * _BETA_XI2_SCALE + _BETA_XI2_LOC, low, high))


def sample_latents(rng: np.random.Generator) -> LatentDraw:
    """Draw one per-spectrum LatentDraw.

    Fixed draw order for reproducibility: alpha, then xi1 for each mode in
    MODES order, then beta*xi2.
    """
    alpha = sample_alpha(rng)
    xi1 = {mode: float(rng.standard_normal()) for mode in MODES}
    beta_xi2 = sample_beta_xi2(rng)
    return LatentDraw(alpha=alpha, xi1=xi1, beta_xi2=beta_xi2)
