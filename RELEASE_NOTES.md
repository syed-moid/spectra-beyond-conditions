# Spectra Beyond Conditions — Paper 1 Reproducibility Package (v1.0.0)

Initial public release of the reproducibility package for the manuscript:

> **Spectra Beyond Conditions: A Diagnostic Framework for Spectral Representation
> Learning from Inelastic Neutron Scattering** *(in preparation).*

This package contains the code, data, configs, trained-model metrics, and figures
needed to reproduce the four main empirical results of Paper 1, a diagnostic study
of spectral representation learning from inelastic neutron scattering (INS).

## What's in this release

- **`manuscript_phase1/`** — the paper's main claims: the latent-microphysics
  generator (latents `ξ₁` linewidth scatter, `ξ₂` frequency shift), the
  conditions-only floor and Spectral Transformer (architectures 5a/5b), the
  evaluation harness, bundled per-seed metrics and checkpoints, and the figure
  scripts.
- **`legacy_phase0/`** — a self-contained reproduction of the deterministic-target
  diagnostic (the `MAE_logM ≈ 0.0048` conditions-only floor) that motivated the
  reformulation.
- **`docs/`** — literature grounding, the human-readable parameter card, and
  calibration-extraction notes.
- **`tests/`** — unit tests for the latent-perturbation generator.

## Reproducible claims (all run from bundled artifacts, < ~2 min, CPU only)

| # | Claim | Key result |
|---|-------|------------|
| 1 | Phase 0 conditions-only floor is ill-posed | `MAE_logM ≈ 0.0048` (conditions alone determine `M`) |
| 2 | Phase 1 generator calibration within band | median \|δΓ/Γ₀\| 0.19, p90 0.58, floor-hit 1.0% — **PASS** |
| 3 | Phase 1 acceptance gate | ST beats recomputed floor 0.6232 by ≥30% (5a +52.5%, 5b +34.7%) — **GATE PASSED** |
| 4 | Severity-resolved crossover | ST-5a below floor through severity 2 (0.4258), crosses above by severity 4 (0.7788) |

See the `README.md` for exact commands and expected output for each claim.

## Requirements

- Python 3.11+ (developed on 3.14.5); dependencies in `requirements.txt`.
- CPU is sufficient for all four reproductions, the test suite, and figure
  regeneration. A GPU (CUDA or Apple MPS) is optional and only useful for
  retraining the Spectral Transformer from scratch.
- ~52 MB disk, < 2 GB RAM.

## Citation

If you use this package, please cite the repository (and the manuscript once
published). A DOI for this release is minted by Zenodo on archive — please cite
that DOI. See `README.md` for the BibTeX entry.
