# Legacy Phase 0 — original deterministic task

This directory preserves the **Phase 0 diagnostic** that motivated the entire
reformulation behind the manuscript. It is *not* part of the Phase 1 results;
it is here so a reviewer can reproduce the finding that triggered the pivot.

## What Phase 0 was

The original framing predicted the coherence merit function **M** from 1D INS
spectra using a *deterministic* generator: given conditions `(T, c, E)`, the
generator produced one `(omega_0, Gamma)` pair, and `M = merit(omega_0, Gamma)`
was a closed-form function of those — hence a deterministic function of
`(T, c, E)`.

## The diagnostic finding (the floor of 0.0048)

A tiny MLP that sees **only** the conditions `(T, c, E)` — never the spectrum —
predicts `log M` to **MAE_logM ≈ 0.0048** on this dataset. Because the
spectrum adds no information beyond the conditions, *any* spectral
representation-learning claim on Phase 0 data is ill-posed: a model can "win"
without using the spectrum at all. This is the conditions-only floor.

Phase 1 (see `../../manuscript_phase1/`) fixes this by injecting latent
microphysics (`xi1` linewidth, `xi2` frequency) so that M is **no longer
deterministic** in `(T, c, E)`. The recomputed conditions-only floor on Phase 1
data rises to **0.6232**, which is the live gate basis for the manuscript.

## Contents

```
original_deterministic_task/
├── data/full_dataset/dataset.npz      # Phase 0 deterministic-target dataset (7,400 spectra)
├── configs/augmentation_realistic.yaml # augmentation config used at generation time
├── floor_reproduction/
│   ├── base.py                         # BaseModel ABC (dependency of the MLP)
│   ├── nonlinear_conditions_mlp.py     # the conditions-only MLP (3→64→64→1)
│   ├── reproduce_phase0_floor.py       # self-contained reproduction script
│   └── expected_phase0_floor.json      # reference output (NL-cond ≈ 0.0048)
└── README.md                           # this file
```

`expected_phase0_floor.json` is the original `diagnostic_report.json` produced by
`scripts/diagnostic_step4_2.py`. Its `nonlinear_conditions_mlp.per_severity`
block holds the floor values (`in_dist_val_severity_1 = 0.00491`,
`stress_severity_* = 0.00477`); the `spectrum_only_st_5b` block is the
companion spectrum-only check from the same diagnostic run.

## Reproduce the floor

```bash
# from this directory, with the package venv active (see top-level README)
cd floor_reproduction
python reproduce_phase0_floor.py
```

Expected (CPU or MPS; ~10–20 s):

```
in_dist (val, severity=1)  MAE_logM = ~0.0049
stress_base                MAE_logM = ~0.0048
```

The exact digits vary by a few units in the 4th decimal across hardware/BLAS,
but the order of magnitude (~5e-3) is the load-bearing result.
