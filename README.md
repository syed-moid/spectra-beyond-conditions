# Spectra Beyond Conditions — Paper 1 Reproducibility Package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20332582.svg)](https://doi.org/10.5281/zenodo.20332582)

Reproducibility package for the manuscript:

> **Spectra Beyond Conditions: A Diagnostic Framework for Spectral Representation
> Learning from Inelastic Neutron Scattering** *(in preparation).*

This repository contains the code, data, configs, trained-model metrics, and
figures needed to reproduce the four main empirical results of Paper 1. The paper
is a **diagnostic study**: it first shows that the original deterministic-target
formulation (Phase 0) is ill-posed for spectral representation learning — a
conditions-only MLP that never sees the spectrum predicts the merit function `M`
to `MAE_logM ≈ 0.0048`, because `M` is deterministic in the experimental
conditions `(T, c, E)`. It then introduces a **latent-microphysics generator**
(Phase 1) that injects two physically-motivated latents (`ξ₁` linewidth scatter,
`ξ₂` frequency shift) so that `M` is no longer determined by `(T, c, E)`. On this
well-posed data, a spectrum-only Spectral Transformer beats the recomputed
conditions-only floor (0.6232) by ≥30% in-distribution, and a severity-resolved
sweep locates the degradation level at which spectral information stops helping.

The package is split into the **manuscript Phase 1 results** and a self-contained
**legacy Phase 0 diagnostic** that reproduces the floor finding that motivated the
reformulation.

## Repository structure

```
spectra-beyond-conditions/
├── README.md                     # this file
├── requirements.txt              # Python dependencies (verified against actual imports)
├── manuscript_phase1/            # the Phase 1 results (the paper's main claims)
│   ├── configs/                  # phase1_parameter_card.yaml + augmentation_realistic.yaml
│   ├── data/full_dataset_phase1/ # Phase 1 latent-microphysics dataset (dataset.npz)
│   ├── src/
│   │   ├── data/                 # generator, latents, augmentations, dataset, calibration, merit, sampling, holdout
│   │   ├── models/               # nonlinear_conditions_mlp (floor) + spectral_transformer (5a/5b)
│   │   └── evaluation/           # evaluation harness + metrics
│   ├── scripts/                  # dataset generation, sanity check, ST training, diagnostics, acceptance report
│   ├── results/                  # diagnostic report, floor json, ST per-seed metrics + checkpoints, calibration, figures' data
│   └── figures/                  # fig1–3 (PDF+SVG) + generate_figures.py
├── legacy_phase0/                # the original deterministic task (NOT a paper result)
│   └── original_deterministic_task/
│       ├── data/full_dataset/    # Phase 0 deterministic-target dataset
│       ├── configs/              # augmentation config used at generation time
│       ├── floor_reproduction/   # self-contained reproduction of the 0.0048 floor
│       └── README.md             # explains scope and how to reproduce 0.0048
├── docs/
│   ├── phase0_literature.md      # v0.3 literature grounding (companion to Section 3)
│   ├── phase1_parameter_card.md  # human-readable explanation of the parameter card
│   └── literature_extraction_notes.md  # Tomeno/Hlinka/Shirane/Harada calibration extraction
└── tests/
    └── test_generator.py         # unit tests for the latent-perturbation generator
```

## Setup

The source environment used **Python 3.14.5**; **Python 3.11+** is recommended
for broad compatibility.

```bash
# 1. create and activate a virtual environment
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt
```

**System dependencies:** none beyond the Python packages. A CUDA GPU is optional
(see Hardware below); PyTorch is installed via pip and will use CPU, CUDA, or
Apple MPS automatically. No CUDA toolkit is required for the bundled
reproductions.

## Reproducing the four main claims

All reproductions below run from the **bundled artifacts** and complete in under
~2 minutes on a CPU — no GPU and no retraining required. Re-training the Spectral
Transformer from scratch is optional and documented under Claim 3.

### Claim 1 — Phase 0 conditions-only floor (0.0048)

```bash
cd legacy_phase0/original_deterministic_task/floor_reproduction
python reproduce_phase0_floor.py
```

Expected output (~15 s, CPU):

```
in_dist (val, severity=1)  MAE_logM = 0.0049
stress_base                MAE_logM = 0.0048
```

The ~5e-3 magnitude is the load-bearing result: the conditions alone determine
`M`, so the spectrum is uninformative under the Phase 0 formulation.

### Claim 2 — Phase 1 generator calibration

```bash
cd manuscript_phase1
python -m src.data.calibration_diagnostics --out results/phase1_calibration.npz
python scripts/phase1_sanity_check.py
```

Expected calibration output (~5 s):

```
median |dGamma/Gamma0|  : 0.1905   target (0.1, 0.4)
p90 |dGamma/Gamma0|     : 0.5800   target >= 0.4
floor hit rate          : 0.0102   target <= 0.02
RESULT: PASS — calibration within acceptance band.
```

The sanity check clears all three gates (latent propagation visibility, target
spread widening, latent round-trip) and writes
`results/phase1_latent_propagation_check.png` and `results/phase1_sanity_check.json`.

To regenerate the Phase 1 dataset itself (optional, ~5–10 min, CPU):

```bash
python scripts/generate_full_dataset.py --phase1   # -> data/full_dataset_phase1/dataset.npz
```

### Claim 3 — Phase 1 acceptance gate result

```bash
cd manuscript_phase1
python scripts/phase1_acceptance_report.py
```

This retrains the conditions-only floor on Phase 1 data (~1 min, CPU) and reads
the bundled per-seed Spectral Transformer metrics in `results/phase1_st_run/`.
Expected output:

```
Phase 1 NL-cond floor (live, new gate basis): 0.6232
| Arch | ST MAE_logM (mean ± std, 3 seeds) | gap vs floor | pass |
| 5a   | 0.2960 ± 0.0148                   | +52.5%       | PASS |
| 5b   | 0.4067 ± 0.0287                   | +34.7%       | PASS |
GATE PASSED
```

**Optional — retrain the Spectral Transformer from scratch** (GPU recommended;
~3–5 min per (arch, seed), ~15–30 min for all six runs on a single A100):

```bash
python scripts/train_spectral_transformer.py --arch 5a --seeds 42 43 44 --device cuda
python scripts/train_spectral_transformer.py --arch 5b --seeds 42 43 44 --device cuda
python scripts/phase1_acceptance_report.py \
    --st-run-dir results/training_runs/spectral_transformer_step4.2_<date>
```

(Use `--device mps` on Apple Silicon or `--device cpu` to run without a GPU.)

### Claim 4 — Severity-resolved evaluation (crossover between severity 2 and 4)

The per-severity table is emitted by the same acceptance-report command as Claim 3,
under "Per-severity breakdown". Expected:

```
| Severity | NL-cond floor | ST-5a  | ST-5b  |
| 0.25     | 0.6510        | 0.1989 | 0.2576 |
| 0.5      | 0.6510        | 0.2069 | 0.3059 |
| 1.0      | 0.6510        | 0.2676 | 0.4073 |
| 2.0      | 0.6510        | 0.4258 | 0.6023 |
| 4.0      | 0.6510        | 0.7788 | 0.9713 |
```

ST-5a stays below the 0.6510 floor through severity 2 (0.4258) and crosses above
it by severity 4 (0.7788) — the spectral-information crossover that is the headline
of Figure 3.

## Regenerating the figures

```bash
python manuscript_phase1/figures/generate_figures.py
```

Reads from `manuscript_phase1/results/` and writes `fig1_concept`,
`fig2_main_result`, `fig3_severity_boundary` as both PDF and SVG into
`manuscript_phase1/figures/` (~5 s, CPU). The numeric values are printed to stdout
for cross-checking against the tables above.

## Running the tests

```bash
python -m pytest tests/test_generator.py -q     # 12 tests: latent perturbation + Δω consolidation
```

## Hardware requirements

- **CPU is sufficient** for: the Phase 0 floor reproduction, the Phase 1 generator
  calibration and sanity check, the acceptance report (reads bundled metrics), the
  test suite, and figure regeneration. All run in under ~2 minutes.
- **GPU recommended** only for re-training the Spectral Transformer from scratch
  (CUDA or Apple MPS; CPU works but is slower). The model is small (~0.7 M params).
- **Memory:** < 2 GB RAM; GPU peak ~1.5 GB.
- **Disk:** ~52 MB for the full package (datasets ~35 MB, model checkpoints ~16 MB).

## License

The source code in this repository is released under the MIT License; see [`LICENSE`](LICENSE).

Generated synthetic data, post-processed numerical results, figure data, configuration files, literature-extraction notes, and documentation are released under the Creative Commons Attribution 4.0 International License (CC BY 4.0); see [`DATA_LICENSE.md`](DATA_LICENSE.md).

This repository does not redistribute copyrighted third-party figures or articles. Literature-derived anchoring values are provided as extracted numerical notes for reproducibility and should be cited through the original publications listed in the manuscript.

## Citation

If you use this package, please cite the archived release via its Zenodo DOI:
<https://doi.org/10.5281/zenodo.20332582>

```bibtex
@misc{moid_spectra_beyond_conditions,
  author       = {Moid, Syed A.},
  title        = {Spectra Beyond Conditions: A Diagnostic Framework for Spectral
                  Representation Learning from Inelastic Neutron Scattering},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20332582},
  url          = {https://doi.org/10.5281/zenodo.20332582},
  note         = {Manuscript in preparation}
}
```

## Contact

`syedmoid@gmail.com`
