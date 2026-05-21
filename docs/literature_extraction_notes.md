# Literature extraction notes

> **Status: newly compiled for this reproducibility package.** This file did not
> exist as a standalone document in the source repository. It distills the
> literature-extraction work that was previously embedded in
> `phase0_literature.md` (v0.3) and in the `empirical_scatter` / `source_notes`
> blocks of `manuscript_phase1/configs/phase1_parameter_card.yaml`. Numbers here
> are reproduced from those two sources; consult them for full context.

## Purpose

The Phase 1 latent-microphysics generator injects two latents — `ξ₁` (per-mode
linewidth scatter) and `ξ₂` (soft-mode frequency shift). Their distributions are
not free parameters: they are anchored to measured BaTiO₃-family INS scatter in
the literature. This document records *what* was extracted from each source and
*how reliable* each extraction is.

## Sources and what was extracted

### Tomeno et al., JPSJ 89, 054601 (2020) — q-resolved Γ, cubic phase
- **Extracted:** q-resolved linewidths Γ(q) at the cubic phase, 453 K; finite-q
  TO frequencies ≈ 25–40 cm⁻¹.
- **Use:** primary anchor for the *magnitude and q-spread* of linewidth scatter
  that motivates `ξ₁`. The intrinsic q-variation factor of ≈ 3–5× across the
  Brillouin zone is what justifies order-unity linewidth tails.
- **Extraction quality:** FWHM values were read off dispersion-plot error bars;
  estimated ~±20 % per-point eyeball error. Open access via OSTI
  (osti.gov/servlets/purl/1684666).

### Hlinka et al., PRL 101, 167402 (2008) — tetragonal room-temperature
- **Extracted:** tetragonal RT soft-mode Γ ≈ 100 ± 10 cm⁻¹ on ω₀ ≈ 35 ± 5 cm⁻¹,
  i.e. a deeply overdamped mode (Γ/ω₀ ≈ 3).
- **Use:** single-paper relative scatter σ(Γ)/Γ ≈ 0.10 and σ(ω₀)/ω₀ ≈ 5/35 ≈ 0.14.
  The 0.14 frequency scatter is the empirical anchor for `ξ₂`; the Phase 1 range
  for `β·ξ₂` is kept **narrower** ([−0.05, 0.07]) so that `ξ₁` stays dominant.
- **Extraction quality:** not open-access; values taken from the reported
  ± uncertainties. Deferred unless the gate fails by 10–20 % with a
  miscalibrated-σ signature.

### Shirane, Axe, Harada, PRB 2, 3651 (1970) — tetragonal ω₀ cross-check
- **Extracted:** tetragonal soft-mode ω₀ extrapolation to ≈ 4.5 meV (36.3 cm⁻¹).
- **Use:** cross-paper consistency check — anchors the Hlinka ω₀ to within ≈ 4 %.
  This cross-paper agreement (0.04) is the lower bound on frequency-scatter
  realism; the single-paper Hlinka σ (0.14) is the upper bound.

### Harada, Axe, Shirane, PRB 4, 155 (1971) — cubic TO/TA overlap
- **Extracted:** qualitative cubic TO/TA mode-overlap behavior.
- **Use:** qualitative anchor only; supports the multi-mode structure (soft +
  acoustic + optical) over which `ξ₁` draws independent perturbations.

## Synthesized scatter ranges (audit trail)

From the four sources above, the parameter card adopts:

| Quantity | Range | Source basis |
|---|---|---|
| Single-paper σ(Γ)/Γ | 0.10 – 0.20 | Hlinka within-sample |
| Cross-paper σ(Γ)/Γ | 0.25 – 0.40 | single- vs poly-crystal disagreement |
| Intrinsic q-variation factor | 3× – 5× | Tomeno q-resolved Γ |
| σ(ω₀)/ω₀ single-paper (Hlinka) | 0.14 | 5/35 from PRL 2008 |
| σ(ω₀)/ω₀ cross-paper (Hlinka↔Shirane) | 0.04 | tetragonal ω₀ agreement |
| Tomeno q-resolved σ typical | 0.10 – 0.15 | dispersion-plot read |

These ranges map onto the latent distributions as documented in
`phase1_parameter_card.md` §2–§3: `ξ₁`'s lognormal `α` (median 0.30) sits inside
the single-to-cross-paper Γ-scatter band, and `ξ₂`'s skew-normal range is held
conservatively below the empirical Hlinka frequency σ.

## Caveats

- All FWHM/linewidth reads from published dispersion plots carry ~±20 % digitization
  error; the latent ranges are intentionally conservative to absorb this.
- The Shirane Fig. 4 data used elsewhere for the synthetic-realism overlay was
  measured at Q = (0.15, 0, 0), not exactly at Γ — a known caveat baked into the
  realism-check protocol (that overlay script is *not* included in this package;
  it lives in the source repository).
