# Phase 1 Parameter Card — human-readable explanation

This document explains `manuscript_phase1/configs/phase1_parameter_card.yaml`,
the calibration card for the Phase 1 latent-microphysics generator. The YAML is
the machine-readable source of truth; this is the prose companion. The actual
numerical constants used at generation time live in
`manuscript_phase1/src/data/latent_perturbations.py` — the card is the
calibration record and provenance trail.

> **One-line summary.** Phase 1 makes the merit function **M** *non-deterministic*
> in the experimental conditions `(T, c, E)` by injecting two physically-motivated
> latent variables per spectrum: `ξ₁` (per-mode linewidth scatter) and `ξ₂`
> (soft-mode frequency shift). This is what makes spectral representation learning
> a well-posed problem (see `../legacy_phase0/` for why Phase 0 was not).

---

## 1. The acceptance gate

| Field | Value |
|---|---|
| Metric | `MAE_logM` (mean absolute error of `log M`) |
| Rule | spectrum-only Spectral Transformer must score **≥30 % below** the recomputed NL-cond MLP floor on Phase 1 data |
| Phase 0 floor (context only) | 0.0048 — *stale*, deterministic-target data |
| Phase 1 floor (live) | **0.6232** — recomputed on Phase 1 data; this is the real gate basis |
| Live computation required | yes — the floor must be recomputed, not assumed |

Once `ξ₁/ξ₂` make M non-deterministic in `(T, c, E)`, the conditions-only floor
rises far above 0.0048, so the gate must use the **live** recomputed floor
(0.6232), not the historical static value.

## 2. ξ₁ — per-mode linewidth perturbation (the primary latent)

The dominant source of variability. Each spectrum draws **three independent**
perturbations (one each for the soft, acoustic, and optical modes):

```
Γ_realized[mode] = Γ_0[mode] · max(0.05, 1 + α · ξ₁[mode])
```

- `ξ₁[mode] ~ N(0, 1)` (standard normal), one draw per mode per spectrum.
- `α ~ Lognormal(μ_log = −1.204, σ_log = 0.40)` → median 0.30, 68 % CI [0.20, 0.45],
  95 % CI [0.135, 0.665].
- **Hard floor at 0.05**: the DHO numerator is ∝ Γ, so a negative Γ would produce
  unphysical negative spectra. Floor-hit rate is held below 2 % (Monte-Carlo
  expected 0.9 %).

**Why per-mode scalars, not a q-correlated field?** The 1D Γ-point generator has
no q-axis, so the Phase 0 q-correlated field design collapses to three correlated
per-mode scalars. The mechanism stays honest: three linewidths varying together
per spectrum, not a fabricated spatial field.

**Calibration acceptance targets** (soft mode, all spectra):
median |Δ| ∈ [0.10, 0.40], p90 |Δ| ≥ 0.40, floor-hit ≤ 2 %.
Monte-Carlo realized: median 0.19, p90 0.56, floor-hit 0.9 % — within targets.

## 3. ξ₂ — soft-mode frequency shift (the secondary latent)

One scalar draw per spectrum:

```
ω₀_realized = ω₀_baseline(T, c, E) · (1 + β · ξ₂)
```

- `β · ξ₂ ~ SkewNormal`, range **[−0.05, 0.07]**, **positively skewed** (hardening).
- Physical rationale: oxygen-vacancy and eight-site Ti disorder modes more often
  *harden* than soften the TO frequency. The range is deliberately narrower than
  the empirical Hlinka σ ≈ 0.14, so that `ξ₁` remains the dominant variability
  driver and the two latents are not confounded.
- **Phase-1.5 widening option**: if the acceptance gate misses by 10–20 % and the
  per-severity breakdown implicates ω₀, widen to [−0.08, 0.10].

**Δω consolidation (important).** The deterministic field-shift term (`+0.01·E²`)
previously appeared in *both* `omega_Q` and `Delta_omega`, double-counting it in
the DHO denominator. Phase 1 consolidates the field shift into `ω₀_baseline` only
and removes `Delta_omega` from the generator call site. Verification invariant:
spectra at **E = 0 are byte-identical** pre/post fix; spectra at E ≠ 0 differ.
This is exercised by `tests/test_generator.py`.

## 4. M recomputation

- M is recomputed from the **realized** `(ω₀, Γ)` via the validated composite
  `merit()` in `src/data/merit.py` — M is never perturbed directly.
- The rejected simple ratio `M = ω² / (Γ² + Γ_min²)` (old Eq. 22) is **not** used.

## 5. Literature calibration anchors

| Source | Contribution |
|---|---|
| Tomeno et al., JPSJ **89**, 054601 (2020) | q-resolved Γ, cubic 453 K; finite-q TO ≈ 25–40 cm⁻¹ |
| Hlinka et al., PRL **101**, 167402 (2008) | tetragonal RT Γ ≈ 100 ± 10 cm⁻¹ on ω₀ ≈ 35 ± 5 cm⁻¹ |
| Shirane–Axe–Harada, PRB **2**, 3651 (1970) | tetragonal ω₀ cross-check (4 % agreement) |
| Harada–Axe–Shirane, PRB **4**, 155 (1971) | cubic TO/TA overlap qualitative anchor |

See `literature_extraction_notes.md` for how the numeric ranges were extracted
from these sources.

## 6. Implementation conventions

- RNG: `np.random.default_rng(seed)`; the seed is logged in every artifact.
- Diagnostic artifacts (this package): `results/phase1_calibration.npz`,
  `results/phase1_diagnostic_report.md`.
- The DHO lineshape (`src/data/spectrum_generator.py::dho()`, Bose-weighted) is
  validated and must not be rewritten.
