"""A3 - conventional DHO-fitting baselines and a simple-feature regression.

Three baselines, all evaluated on the 500-spectrum stress base across the
severity sweep s in {0.25, 0.5, 1, 2, 4}:

  DHO-matched   the generator's own forward model as the fit function (all
                three modes + central peak + zero offset + pseudo-Voigt
                resolution + linear background), initialised from the baseline
                parameters at the nominal (T, c, E). Best case for fitting.

  DHO-single    one DHO + linear background + a central Lorentzian, with the
                detailed-balance weighting at the known T. No knowledge of the
                acoustic/optical modes, the central-peak model, or the
                resolution function. Realistic case.

  feat_gbm      peak position / FWHM / height / integrated intensity in the
                soft-mode window / first and second spectral moments /
                central-to-peak ratio, fed to gradient boosting, 5 seeds,
                trained on the train split at the same log-uniform severity
                distribution the spectral transformers saw.

From the fitted (omega0, Gamma) the merit is recomputed with the composite
merit(); errors are reported on omega0, Gamma, Gamma/omega0 and log M, with
fit failures both excluded and replaced by the metadata-only prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sbc.data.augmentations import _pseudo_voigt_kernel  # noqa: E402
from sbc.data.dataset import InsSpectraDataset  # noqa: E402
from sbc.data.merit import merit, K_B_meV_per_K  # noqa: E402
from sbc.data.spectrum_generator import (  # noqa: E402
    Gamma_Q, bose, central_peak, dho, omega_Q, soft_mode_F2, _MODES,
)
import dataset_paths as DS  # noqa: E402

NPZ = DS.FULL
A2_PRED = ROOT / "results" / "revision" / "A2" / "predictions_seedavg.csv"
OUTDIR = DS.rev("A3")
SEVERITIES = (0.25, 0.5, 1.0, 2.0, 4.0)
SEEDS = (42, 43, 44, 45, 46)
TRAIN_SEVERITY_RANGE = (0.25, 4.0)   # matches train_spectral_transformer.py


# --------------------------------------------------------------------------- #
# forward models used as fit functions                                        #
# --------------------------------------------------------------------------- #

def _convolve(y, grid, sigma, eta):
    step = float(grid[1] - grid[0])
    if sigma <= 1e-3:
        return y
    k = _pseudo_voigt_kernel(step, sigma, eta)
    return np.convolve(y, k, mode="same") * step


def model_matched(p, grid, T, E):
    """Generator forward model.

    Parameter vector, by index -- the authoritative mapping:

        0 om0    soft-mode frequency (meV)
        1 g_soft soft-mode damping parameter (meV)
        2 g_ac   acoustic damping parameter (meV)
        3 g_op   optical damping parameter (meV)
        4 cp_w   central-peak WIDTH (meV). Its amplitude is supplied by the
                 generator's envelope inside central_peak() and is NOT fitted.
        5 dz     energy zero-offset (meV)
        6 sig    resolution width (meV). The pseudo-Voigt mixing parameter is
                 supplied at 0.3 and is not fitted.
        7 a0     background intercept, as a fraction of the peak
        8 a1     background slope, as a fraction of the peak per meV
        9 scale  overall scale

    Rounds 46-48 mis-stated indices 4-8 and applied an "offset bound" to index 8,
    which is the background slope. The model itself was never wrong; the
    description of it was.
    """
    om0, g_soft, g_ac, g_op, cp_w, dz, sig, a0, a1, scale = p
    g = grid - dz
    S = dho(g, om0, g_soft, 0.0, T, _MODES["soft"]["F2"])
    S = S + dho(g, _MODES["acoustic"]["omega"], g_ac, 0.0, T, _MODES["acoustic"]["F2"])
    S = S + dho(g, _MODES["optical"]["omega"], g_op, 0.0, T, _MODES["optical"]["F2"])
    S = S + central_peak(g, T, width_meV=cp_w)
    S = scale * _convolve(S, grid, sig, 0.3)
    pk = float(np.max(S)) if np.max(S) > 0 else 1.0
    return S + pk * (a0 + a1 * grid)



def _dho_lineshape(grid, A, om0, gamma, T):
    """Single damped-harmonic-oscillator term, in the generator's convention.

    S(w) = A * occ(w) * gamma / ((w^2 - w0^2)^2 + 4 gamma^2 w^2),
    occ(w) = w / (1 - exp(-w / kB T)),  occ(0) = kB T.

    Two things here were wrong in the single-mode fitting baselines until round
    49, and both are the defects the GENERATOR was corrected for at v9:

      * the occupation factor was evaluated at omega0 and applied as a step,
        `where(w >= 0, n0 + 1, n0)`, which is constant per spectrum and violates
        detailed balance as a function of energy transfer;
      * the denominator used 4 * omega0^2 * gamma^2, the near-resonance
        approximation, instead of 4 * gamma^2 * w^2.

    So the fitting baselines were being scored against spectra generated with a
    lineshape they could not represent. This is the corrected form; the unit test
    in tests/test_fit_lineshape.py fails on the old one.
    """
    w = np.asarray(grid, dtype=float)
    x = w / (K_B_meV_per_K * T)
    near = np.abs(x) < 1e-8
    occ = np.where(near, K_B_meV_per_K * T, w / np.where(near, 1.0, -np.expm1(-x)))
    return A * occ * gamma / ((w ** 2 - om0 ** 2) ** 2 + 4.0 * gamma ** 2 * w ** 2)



def _amp_for_peak(ymax, om0, gamma, T):
    """Amplitude A putting the trial DHO's peak at `ymax`, in the corrected form.

    At w = om0 the corrected lineshape is A * occ(om0) / (4 * gamma * om0^2), so
    A = ymax * 4 * gamma * om0^2 / occ(om0).

    The initialiser used until round 49 inverted the OLD lineshape instead, giving
    A0 = ymax * 4 * om0^2 * gamma / (n0 + 1), which is a factor of om0 (3-12 here)
    too large. Correcting the lineshape without correcting this would have started
    every single-mode fit an order of magnitude off its own peak.
    """
    om0 = float(om0)
    occ = om0 / -np.expm1(-om0 / (K_B_meV_per_K * T))
    return max(ymax * 4.0 * gamma * om0 ** 2 / max(occ, 1e-12), 1e-9)


def model_single(p, grid, T):
    """Single DHO + central Lorentzian + linear background.
    p = [A, om0, gamma, A_cp, w_cp, b0, b1]."""
    A, om0, gamma, A_cp, w_cp, b0, b1 = p
    S = _dho_lineshape(grid, A, om0, gamma, T)
    S = S + A_cp * w_cp / (np.pi * (grid ** 2 + w_cp ** 2))
    return S + b0 + b1 * grid


# --------------------------------------------------------------------------- #
# fitting                                                                     #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Generator latent prior, tabulated once, for the MAP fitting baseline.
# --------------------------------------------------------------------------- #
from scipy.stats import norm as _norm, skewnorm as _skewnorm  # noqa: E402
from sbc.data.latent_perturbations import (  # noqa: E402
    ALPHA_MU_LOG, ALPHA_SIGMA_LOG, BETA_XI2_HIGH, BETA_XI2_LOW, BETA_XI2_SKEW,
    GAMMA_FLOOR, _BETA_XI2_LOC, _BETA_XI2_SCALE)

_PRIOR_WEIGHT = 1.0           # explicit prior-vs-data weight; see fit_matched_prior
_PRIOR_FLOOR = 1e-12          # density floor; -log = 27.6, finite but strongly disfavoured


def _log_prior_multiplier(m, n_quad=2001):
    """log p of the realized linewidth multiplier max(GAMMA_FLOOR, 1 + alpha*xi1).

    alpha ~ LogNormal, xi1 ~ N(0,1), so u = alpha*xi1 is a scale mixture of
    normals with no closed form. Its density is obtained by quadrature over
    alpha, p(u) = int p_a(a) (1/a) phi(u/a) da, which stays accurate in the
    tails where a histogram estimate floors out. The floor at GAMMA_FLOOR is a
    point mass (0.9% of draws); values at or below it are given the mass rather
    than a density, so the fit is not pushed onto the floor by a histogram spike.
    """
    m = float(m)
    a = np.exp(ALPHA_MU_LOG + ALPHA_SIGMA_LOG * np.linspace(-6.0, 6.0, n_quad))
    w = _norm.pdf((np.log(a) - ALPHA_MU_LOG) / ALPHA_SIGMA_LOG) / (a * ALPHA_SIGMA_LOG)
    if m <= GAMMA_FLOOR + 1e-12:
        # P(1 + alpha*xi1 <= GAMMA_FLOOR) = E_a[ Phi((GAMMA_FLOOR-1)/a) ]
        mass = np.trapezoid(w * _norm.cdf((GAMMA_FLOOR - 1.0) / a), a)
        return float(np.log(max(mass, _PRIOR_FLOOR)))
    u = m - 1.0
    dens = np.trapezoid(w * _norm.pdf(u / a) / a, a)
    return float(np.log(max(dens, _PRIOR_FLOOR)))


def _log_prior_shift(r):
    """log p of beta*xi2 = clip(skewnorm*scale + loc, LOW, HIGH), in closed form.

    The clips carry point masses (1.6% at the upper clip), handled explicitly;
    the interior is the transformed skew-normal density.
    """
    r = float(r)
    lo, hi = BETA_XI2_LOW, BETA_XI2_HIGH
    z = lambda v: (v - _BETA_XI2_LOC) / _BETA_XI2_SCALE
    if r < lo - 1e-12 or r > hi + 1e-12:
        return float(np.log(_PRIOR_FLOOR))
    if abs(r - lo) <= 1e-12:
        return float(np.log(max(_skewnorm.cdf(z(lo), BETA_XI2_SKEW), _PRIOR_FLOOR)))
    if abs(r - hi) <= 1e-12:
        return float(np.log(max(_skewnorm.sf(z(hi), BETA_XI2_SKEW), _PRIOR_FLOOR)))
    d = _skewnorm.pdf(z(r), BETA_XI2_SKEW) / _BETA_XI2_SCALE
    return float(np.log(max(d, _PRIOR_FLOOR)))


_LOG_PRIOR_MULT, _LOG_PRIOR_SHIFT = _log_prior_multiplier, _log_prior_shift


# ---------------------------------------------------------------------------
# Round 46: the prior residual, corrected.
#
# The previous form was  r_prior = sqrt(2 * max(nlp, 0)) * W.  The max() was a
# literal clip, present only because sqrt() of a negative number is not real --
# and nlp IS negative over much of the support, because both factors are
# continuous densities that exceed 1 near their modes (max log p(m) = +0.365,
# max log p(r) = +3.327, so nlp reaches -3.69). Wherever the clip bound, the
# penalty and its gradient were exactly zero, so the prior exerted no force at
# all -- over a region covering nearly the whole support of the frequency shift.
#
# The fix is to subtract the global minimum instead of clipping:
#
#     r_prior = sqrt(2 * W * (nlp - nlp_min)),   nlp_min = min over the support
#
# nlp_min is a constant, so the objective changes by a constant and the MAP is
# unchanged -- but the residual is now real everywhere, smooth, and carries the
# correct gradient across the whole support.
#
# The boundary atoms are handled by restricting the prior to the continuous
# interior (round 46 option ii): mixing a probability mass with a density is
# dimensionally inconsistent, and the masses are small. The fraction of realized
# parameters sitting on an atom is reported in A23/prior_clip_diagnostic.json.
# ---------------------------------------------------------------------------
_QUAD_CACHE = {}


def _quad_grid(n_quad=2001):
    """alpha grid and weights for the multiplier quadrature; built once."""
    if n_quad not in _QUAD_CACHE:
        a = np.exp(ALPHA_MU_LOG + ALPHA_SIGMA_LOG * np.linspace(-6.0, 6.0, n_quad))
        w = _norm.pdf((np.log(a) - ALPHA_MU_LOG) / ALPHA_SIGMA_LOG) / (a * ALPHA_SIGMA_LOG)
        _QUAD_CACHE[n_quad] = (a, w, w / a)
    return _QUAD_CACHE[n_quad]


def _log_prior_multiplier_interior(m, n_quad=2001):
    """Continuous part only of log p(multiplier); no point mass at the floor."""
    m = float(m)
    if m <= GAMMA_FLOOR + 1e-12:
        return float(np.log(_PRIOR_FLOOR))
    a, _, w_over_a = _quad_grid(n_quad)
    dens = np.trapezoid(w_over_a * _norm.pdf((m - 1.0) / a), a)
    return float(np.log(max(dens, _PRIOR_FLOOR)))


def _log_prior_shift_interior(r):
    """Continuous part only of log p(beta*xi2); no masses at the two clips."""
    r = float(r)
    lo, hi = BETA_XI2_LOW, BETA_XI2_HIGH
    if r <= lo + 1e-12 or r >= hi - 1e-12:
        return float(np.log(_PRIOR_FLOOR))
    z = (r - _BETA_XI2_LOC) / _BETA_XI2_SCALE
    d = _skewnorm.pdf(z, BETA_XI2_SKEW) / _BETA_XI2_SCALE
    return float(np.log(max(d, _PRIOR_FLOOR)))


_NLP_MIN_CACHE = {}


def _nlp_min_interior(n=4001):
    """Global minimum of the interior nlp over the support. Parameter-independent."""
    if "v" not in _NLP_MIN_CACHE:
        ms = np.linspace(GAMMA_FLOOR, 6.0, n)
        rs = np.linspace(BETA_XI2_LOW, BETA_XI2_HIGH, n)
        lpm = max(_log_prior_multiplier_interior(float(x)) for x in ms)
        lpr = max(_log_prior_shift_interior(float(x)) for x in rs)
        _NLP_MIN_CACHE["v"] = -(lpm + lpr)
    return _NLP_MIN_CACHE["v"]


def nlp_interior(m, fr):
    """Negative log prior on the continuous interior, shifted to be >= 0."""
    raw = -_log_prior_multiplier_interior(m) - _log_prior_shift_interior(fr)
    return max(raw - _nlp_min_interior(), 0.0)   # >=0 by construction; guards float noise

BOUND_TOL = 1e-3   # relative distance to a bound that counts as "at bound"


def _at_bound(x, lo, hi):
    span = max(hi - lo, 1e-12)
    return (x - lo) / span < BOUND_TOL or (hi - x) / span < BOUND_TOL


def fit_matched(y, grid, T, c, E):
    om_b = omega_Q(T, c / 100.0, E)
    gm_b = Gamma_Q(T, c / 100.0, E)
    p0 = np.array([om_b, gm_b, _MODES["acoustic"]["Gamma"], _MODES["optical"]["Gamma"],
                   1.5, 0.0, 0.4, 0.02, 0.0, 1.0])
    # Bounds, by the index table in model_matched(). All FIXED: none varies with
    # severity.
    #   5 energy offset  +-3.0 meV. The generator draws it uniformly on
    #     +-0.5*severity, so this covers 100% of draws at every severity tested
    #     (max |offset| 2.0 meV at s = 4). Rounds 46-48 believed this bound sat at
    #     index 8 and "widened" it; index 8 is the background slope, and the
    #     energy offset was never restricted.
    #   2 acoustic Gamma lower bound 0.02 meV against a realized minimum of
    #     0.030 meV (0.6 * GAMMA_FLOOR). Widened in round 46, and correct.
    #   8 background slope +-0.6 per meV as a fraction of the peak. Loose by
    #     design: the generator's linear background reaches ~0.3 of the peak
    #     across the window, implying |a1| ~ 0.02. It is a nuisance parameter.
    lo = np.array([0.5, 0.02, 0.02, 0.05, 0.05, -3.0, 1e-4, 0.0, -0.6, 0.2])
    hi = np.array([15.0, 40.0, 10.0, 20.0, 12.0, 3.0, 4.0, 3.0, 0.6, 5.0])
    p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)
    scale = max(float(np.max(y)), 1e-12)
    try:
        r = least_squares(lambda p: (model_matched(p, grid, T, E) - y) / scale,
                          p0, bounds=(lo, hi), max_nfev=4000, method="trf")
    except Exception:
        return dict(ok=False, reason="exception", omega0=np.nan, Gamma=np.nan, status=-99)
    p = r.x
    ok = r.status > 0
    reason = "" if ok else "no_convergence"
    if _at_bound(p[0], lo[0], hi[0]):
        ok, reason = False, "omega0_at_bound"
    elif _at_bound(p[1], lo[1], hi[1]):
        ok, reason = False, "Gamma_at_bound"
    return dict(ok=ok, reason=reason, omega0=float(p[0]), Gamma=float(p[1]),
                status=int(r.status), cost=float(r.cost),
                params=[float(v) for v in p])


def fit_matched_prior(y, grid, T, c, E, weight=None):
    """Matched forward model plus a penalty built from the soft-mode marginals.

    `weight` overrides _PRIOR_WEIGHT. Setting it to 0.0 gives the **W = 0
    control**: the identical objective, optimizer, initialisation, bounds and
    tolerances with the penalty switched off, so that the difference from the
    regularized fit isolates the penalty rather than confounding it with the
    change of solver that separates this from `fit_matched` (round 48 item 4).

    The prior is the generator's, not an approximation of it:

        Gamma = Gamma_0(T,c,E) * max(GAMMA_FLOOR, 1 + alpha*xi1),
            alpha ~ LogNormal(ALPHA_MU_LOG, ALPHA_SIGMA_LOG), xi1 ~ N(0,1)
        omega0 = omega_0(T,c,E) * (1 + beta*xi2),
            beta*xi2 ~ clipped skew-normal, matched here by its moments

    Caveat on weighting, stated because it bounds what this baseline shows. The
    data residuals are scaled by max(y) rather than by a per-bin noise sigma, so
    the chi-squared term is not in calibrated units and the prior's weight
    relative to the data is not uniquely determined. `_PRIOR_WEIGHT` makes that
    choice explicit and is set to 1. This baseline therefore answers "does a
    generator-matched prior close part of the gap", not "what is the exact
    posterior". A fully calibrated comparison needs the per-bin counting
    variance, which the stored spectra do not carry.
    """
    om_b = omega_Q(T, c / 100.0, E)
    gm_b = Gamma_Q(T, c / 100.0, E)
    p0 = np.array([om_b, gm_b, _MODES["acoustic"]["Gamma"], _MODES["optical"]["Gamma"],
                   1.5, 0.0, 0.4, 0.02, 0.0, 1.0])
    # Identical fixed bounds to fit_matched(); see the index table in
    # model_matched(). No bound varies with severity.
    lo = np.array([0.5, 0.02, 0.02, 0.05, 0.05, -3.0, 1e-4, 0.0, -0.6, 0.2])
    hi = np.array([15.0, 40.0, 10.0, 20.0, 12.0, 3.0, 4.0, 3.0, 0.6, 5.0])
    p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)
    scale = max(float(np.max(y)), 1e-12)
    n = len(grid)

    # Round 46 option (a): the direct scalar objective
    #
    #     J(p) = 0.5 * sum(((f(p) - y)/max y)^2) + W * nlp(p)
    #
    # minimised with a bounded scalar optimiser. Option (b) -- appending
    # sqrt(2W(nlp - nlp_min)) as a least-squares residual row -- was implemented
    # first and rejected on evidence: nlp attains its minimum in the interior, so
    # that residual behaves like |p - p_mode| near the prior mode. The kink is
    # invisible to a Gauss-Newton convergence test, and TRF failed to declare
    # convergence on 78.5% of spectra (471/600 `no_convergence`, none at a bound,
    # all with sensible fitted parameters). The scalar objective is smooth there.
    def objective(p):
        m = p[1] / max(gm_b, 1e-12)                       # linewidth multiplier
        fr = p[0] / max(om_b, 1e-12) - 1.0                # fractional frequency shift
        r = (model_matched(p, grid, T, E) - y) / scale
        w = _PRIOR_WEIGHT if weight is None else weight
        if w == 0.0:
            return 0.5 * float(np.dot(r, r))          # W = 0 control: no penalty term
        return 0.5 * float(np.dot(r, r)) + w * nlp_interior(m, fr)

    try:
        res = minimize(objective, p0, method="L-BFGS-B",
                       bounds=list(zip(lo, hi)),
                       options={"maxfun": 20000, "maxiter": 4000})
    except Exception:
        return dict(ok=False, reason="exception", omega0=np.nan, Gamma=np.nan, status=-99)
    p = res.x
    ok = bool(res.success)
    reason = "" if ok else "no_convergence"
    r = type("R", (), {"status": 1 if ok else 0, "cost": float(res.fun)})()
    if _at_bound(p[0], lo[0], hi[0]):
        ok, reason = False, "omega0_at_bound"
    elif _at_bound(p[1], lo[1], hi[1]):
        ok, reason = False, "Gamma_at_bound"
    return dict(ok=ok, reason=reason, omega0=float(p[0]), Gamma=float(p[1]),
                status=int(r.status), cost=float(r.cost),
                params=[float(v) for v in p])


def fit_single(y, grid, T, c, E):
    om_b = omega_Q(T, c / 100.0, E)
    gm_b = Gamma_Q(T, c / 100.0, E)
    ymax = max(float(np.max(y)), 1e-12)
    ymin = float(np.min(y))
    # amplitude initialisation so the peak of the trial DHO matches the data peak
    A0 = _amp_for_peak(ymax, om_b, gm_b, T)
    p0 = np.array([A0, om_b, gm_b, 0.05 * ymax, 1.5, max(ymin, 0.0), 0.0])
    lo = np.array([1e-12, 0.5, 0.02, 0.0, 0.05, -0.5 * ymax, -0.5 * ymax / 15.0])
    hi = np.array([1e6 * A0, 15.0, 40.0, 50.0 * ymax, 12.0, 1.5 * ymax, 0.5 * ymax / 15.0])
    p0 = np.clip(p0, lo + 1e-12, hi - 1e-12)
    scale = ymax
    try:
        r = least_squares(lambda p: (model_single(p, grid, T) - y) / scale,
                          p0, bounds=(lo, hi), max_nfev=4000, method="trf")
    except Exception:
        return dict(ok=False, reason="exception", omega0=np.nan, Gamma=np.nan, status=-99)
    p = r.x
    ok = r.status > 0
    reason = "" if ok else "no_convergence"
    if _at_bound(p[1], lo[1], hi[1]):
        ok, reason = False, "omega0_at_bound"
    elif _at_bound(p[2], lo[2], hi[2]):
        ok, reason = False, "Gamma_at_bound"
    return dict(ok=ok, reason=reason, omega0=float(p[1]), Gamma=float(p[2]),
                status=int(r.status), cost=float(r.cost),
                params=[float(v) for v in p])


# --------------------------------------------------------------------------- #
# simple spectral features                                                    #
# --------------------------------------------------------------------------- #

SOFT_WINDOW = (0.5, 15.0)     # meV, Stokes side where the soft mode lives


def features(y, grid):
    yb = y - np.median(np.concatenate([y[:30], y[-30:]]))   # crude background removal
    w = (grid >= SOFT_WINDOW[0]) & (grid <= SOFT_WINDOW[1])
    yw, gw = yb[w], grid[w]
    i = int(np.argmax(yw))
    pk_pos, pk_h = float(gw[i]), float(yw[i])
    half = 0.5 * pk_h
    above = yw >= half
    fwhm = float(gw[above].max() - gw[above].min()) if above.any() else 0.0
    step = float(grid[1] - grid[0])
    integ = float(np.sum(np.clip(yw, 0, None)) * step)
    wt = np.clip(yw, 0, None)
    tot = max(float(wt.sum()), 1e-12)
    m1 = float(np.sum(gw * wt) / tot)
    m2 = float(np.sum((gw - m1) ** 2 * wt) / tot)
    cen = (np.abs(grid) <= 1.0)
    central = float(np.max(yb[cen]))
    ratio = central / pk_h if pk_h > 0 else 0.0
    ptot = max(float(np.sum(np.clip(yb, 0, None)) * step), 1e-12)
    return [pk_pos, fwhm, pk_h, integ, m1, m2, ratio, central,
            float(np.max(y)), float(np.min(y)), integ / ptot]


FEATURE_NAMES = ["peak_pos_meV", "fwhm_meV", "peak_height", "integ_soft_window",
                 "moment1_meV", "moment2_meV2", "central_to_peak_ratio", "central_height",
                 "raw_max", "raw_min", "soft_window_intensity_fraction"]


# --------------------------------------------------------------------------- #

def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def config_hash():
    h = hashlib.sha256()
    for p in (ROOT / "src" / "data" / "spectrum_generator.py",
              ROOT / "src" / "data" / "augmentations.py",
              Path(__file__)):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--severities", type=float, nargs="+", default=list(SEVERITIES))
    ap.add_argument("--limit", type=int, default=None, help="fit only the first N stress spectra")
    ap.add_argument("--skip-fits", action="store_true")
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    ap.add_argument("--npz", type=Path, default=NPZ)
    ap.add_argument("--dataset-version", type=str, default=DS.VERSION)
    ap.add_argument("--a2-pred", type=Path, default=A2_PRED)
    ap.add_argument("--fallback-model", type=str, default="mlp_mse")
    ap.add_argument("--feature-ablation-severity", type=float, default=1.0)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    stress = InsSpectraDataset(args.npz, "stress_base", severity=1.0, as_torch=False)
    grid = stress._omega_grid
    n = len(stress) if args.limit is None else min(args.limit, len(stress))
    T = stress._T_K[:n].astype(float); c = stress._c_pct[:n].astype(float)
    E = stress._E_kVcm[:n].astype(float)
    om_true = stress._omega_Q[:n].astype(float); gm_true = stress._Gamma_Q[:n].astype(float)
    logM_true = np.log(np.clip(stress._M[:n].astype(float), 1e-9, None))

    # metadata-only fallback predictions for failed fits (A2 seed-averaged MLP on stress_base)
    a2 = pd.read_csv(args.a2_pred)
    fb = a2[(a2.model == args.fallback_model)
            & (a2.split == "stress_base")]["logM_pred_seedavg"].to_numpy()[:n]

    rows = []
    if not args.skip_fits:
        for sev in args.severities:
            t0 = time.time()
            # Round 47 item 5: the prior-regularized fit joins the severity sweep,
            # so Figure 2 and the A6 completeness tables carry all three fitting
            # variants rather than only the unregularized pair.
            for method, fn in (("dho_matched", fit_matched),
                               ("dho_matched_prior", fit_matched_prior),
                               ("dho_single", fit_single)):
                for i in range(n):
                    y = stress.get_augmented(i, severity=sev)
                    r = fn(y, grid, float(T[i]), float(c[i]), float(E[i]))
                    om, gm = r["omega0"], r["Gamma"]
                    lm = (np.log(max(merit(om, gm, float(T[i]), float(E[i])), 1e-9))
                          if np.isfinite(om) and np.isfinite(gm) and gm > 0 else np.nan)
                    rows.append({"dataset_version": args.dataset_version,
                                 "method": method, "severity": sev, "idx": i,
                                 "T_K": T[i], "c_pct": c[i], "E_kVcm": E[i],
                                 "omega0_fit": om, "Gamma_fit": gm,
                                 "omega0_true": om_true[i], "Gamma_true": gm_true[i],
                                 "logM_fit": lm, "logM_true": logM_true[i],
                                 "logM_metadata_fallback": fb[i],
                                 "fit_ok": r["ok"], "fail_reason": r["reason"],
                                 "status": r["status"]})
            print(f"  severity {sev}: {3*n} fits in {time.time()-t0:.0f}s", flush=True)
        fits = pd.DataFrame(rows)
        fits.to_csv(args.outdir / "fit_results_per_spectrum.csv", index=False)
    else:
        fits = pd.read_csv(args.outdir / "fit_results_per_spectrum.csv")

    # ---- simple-feature GBM -------------------------------------------------
    train = InsSpectraDataset(args.npz, "train", severity=1.0, as_torch=False)
    rng = np.random.default_rng(args.seed)
    lo, hi = TRAIN_SEVERITY_RANGE
    tr_sev = np.exp(rng.uniform(np.log(lo), np.log(hi), size=len(train)))
    t0 = time.time()
    Xtr = np.array([features(train.get_augmented(i, severity=float(tr_sev[i])), grid)
                    for i in range(len(train))])
    ytr = np.log(np.clip(train._M.astype(float), 1e-9, None))
    print(f"  train features: {Xtr.shape} in {time.time()-t0:.0f}s", flush=True)

    feat_rows = []
    for sev in args.severities:
        Xte = np.array([features(stress.get_augmented(i, severity=sev), grid) for i in range(n)])
        for seed in args.seeds:
            g = HistGradientBoostingRegressor(loss="absolute_error", random_state=seed,
                                              early_stopping=True, validation_fraction=0.1)
            g.fit(Xtr, ytr)
            p = g.predict(Xte)
            feat_rows.append(pd.DataFrame({"dataset_version": args.dataset_version,
                                           "method": "feat_gbm", "severity": sev, "seed": seed,
                                           "idx": np.arange(n), "logM_pred": p,
                                           "logM_true": logM_true}))
    feats = pd.concat(feat_rows, ignore_index=True)
    feats.to_csv(args.outdir / "feature_gbm_predictions.csv", index=False)

    # ---- summary ------------------------------------------------------------
    summ = []
    for method in ("dho_matched", "dho_matched_prior", "dho_single"):
        for sev in args.severities:
            d = fits[(fits.method == method) & (fits.severity == sev)]
            ok = d.fit_ok.astype(bool).to_numpy()
            gt_ratio = d.Gamma_true / d.omega0_true
            ft_ratio = d.Gamma_fit / d.omega0_fit
            lm_excl = np.abs(d.logM_fit - d.logM_true)[ok & np.isfinite(d.logM_fit)]
            lm_repl = np.where(ok & np.isfinite(d.logM_fit), d.logM_fit, d.logM_metadata_fallback)
            summ.append({
                "dataset_version": args.dataset_version,
                "method": method, "severity": sev, "N": len(d),
                "fit_failure_rate": float(1.0 - ok.mean()),
                "fail_no_convergence": float(np.mean(d.fail_reason == "no_convergence")),
                "fail_Gamma_at_bound": float(np.mean(d.fail_reason == "Gamma_at_bound")),
                "fail_omega0_at_bound": float(np.mean(d.fail_reason == "omega0_at_bound")),
                "MAE_omega0_meV_ok": float(np.abs(d.omega0_fit - d.omega0_true)[ok].mean()),
                "MAE_Gamma_meV_ok": float(np.abs(d.Gamma_fit - d.Gamma_true)[ok].mean()),
                "MAE_Gamma_over_omega0_ok": float(np.abs(ft_ratio - gt_ratio)[ok].mean()),
                "MAE_logM_failures_excluded": float(lm_excl.mean()),
                "MAE_logM_failures_replaced_by_metadata": float(np.abs(lm_repl - d.logM_true).mean()),
            })
    for sev in args.severities:
        d = feats[feats.severity == sev]
        per = [np.abs(d[d.seed == s].logM_pred.to_numpy() - logM_true).mean() for s in args.seeds]
        summ.append({"dataset_version": args.dataset_version,
                     "method": "feat_gbm", "severity": sev, "N": n,
                     "fit_failure_rate": 0.0,
                     "MAE_logM_failures_excluded": float(np.mean(per)),
                     "MAE_logM_failures_replaced_by_metadata": float(np.mean(per)),
                     "MAE_logM_sd_over_seeds": float(np.std(per, ddof=1))})
    # ---- feature-set ablation (which features carry the advantage?) ---------
    GROUPS = {
        "position_width_only": ["peak_pos_meV", "fwhm_meV"],
        "plus_intensity": ["peak_pos_meV", "fwhm_meV", "peak_height", "integ_soft_window",
                           "central_height", "raw_max", "raw_min"],
        "all_11": list(FEATURE_NAMES),
    }
    sev_ab = args.feature_ablation_severity
    Xte_ab = np.array([features(stress.get_augmented(i, severity=sev_ab), grid) for i in range(n)])
    abl_rows = []
    for gname, cols in GROUPS.items():
        sel = [FEATURE_NAMES.index(c) for c in cols]
        per = []
        for seed in args.seeds:
            g = HistGradientBoostingRegressor(loss="absolute_error", random_state=seed,
                                              early_stopping=True, validation_fraction=0.1)
            g.fit(Xtr[:, sel], ytr)
            per.append(float(np.abs(g.predict(Xte_ab[:, sel]) - logM_true).mean()))
        abl_rows.append({"dataset_version": args.dataset_version, "feature_set": gname,
                         "n_features": len(cols), "severity": sev_ab, "N": n,
                         "MAE_logM_mean": float(np.mean(per)),
                         "MAE_logM_sd": float(np.std(per, ddof=1)),
                         "features": ";".join(cols)})
        print(f"  ablation {gname:20s} ({len(cols):2d} feats): "
              f"MAE_logM {np.mean(per):.4f} +/- {np.std(per, ddof=1):.4f}")
    pd.DataFrame(abl_rows).to_csv(args.outdir / "feature_set_ablation.csv", index=False)

    sdf = pd.DataFrame(summ)
    sdf.to_csv(args.outdir / "summary_by_severity.csv", index=False)
    print(sdf.to_string(index=False))

    sidecar = {"task": "A3", "script": str(Path(__file__).relative_to(ROOT)),
               "dataset_version": args.dataset_version,
               "dataset_npz": str(Path(args.npz).resolve().relative_to(ROOT)),
               "metadata_fallback_model": args.fallback_model,
               "feature_ablation_severity": args.feature_ablation_severity,
               "master_seed": args.seed, "gbm_seeds": list(args.seeds),
               "severities": list(args.severities), "n_stress_spectra": int(n),
               "train_severity_range": list(TRAIN_SEVERITY_RANGE),
               "feature_names": FEATURE_NAMES,
               "config_hash_sha256_16": config_hash(), "git_commit": git_sha(),
               "date_utc": datetime.now(timezone.utc).isoformat(),
               "numpy": np.__version__, "scipy_least_squares": "trf, max_nfev=4000",
               "pandas": pd.__version__}
    (args.outdir / "run_sidecar.json").write_text(json.dumps(sidecar, indent=2))
    print(f"\nwrote {args.outdir}")


if __name__ == "__main__":
    main()
