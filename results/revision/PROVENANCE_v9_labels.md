# v9 result labels — correction record (2026-09-09)

## What was wrong

`dataset_version` was written from a hardcoded `"v8"` string literal in the
training scripts (`a4_train_v8.py`, `a17_arch_sweep.py`, `a19_confirm.py`,
`a18b_energy_tilt.py`, `a5_label_permutation.py`). The v9 stages that ran on
2026-09-08 therefore recorded `dataset_version=v8` while reading v9 data.

**No number is affected.** The dataset actually consumed is recorded
independently, per run, in the `dataset` column, and it reads
`manuscript_phase1/data/full_dataset_phase1_v9/dataset.npz` on every row of
every affected file. Only the version label was wrong.

## Corrected

| file | rows relabelled |
|---|---|
| `A4_v9_extended/training_runs.csv` | 29 / 29 |
| `A4_v9_extended/training_summary.csv` | 7 / 7 |
| `A17_v9/variant_runs.csv` | 21 / 21 |
| `A19_v9/variant_runs.csv` | 18 / 18 |
| `A18b_v9_tilt/training_runs.csv` | 1 / 1 |

Sidecars `A4_v9_extended/run_sidecar_gate.json`, `A17_v9/run_sidecar.json` and
`A19_v9/run_sidecar.json` were corrected the same way. `A19_v9/run_sidecar.json`
additionally recorded `"task": "A17"`: that directory holds the A19 width
variants, which are produced by `a17_arch_sweep.py`, so the script name was
right and the task label was wrong. It now reads `"task": "A19"` with the
script named explicitly. `A1_v9` and `A3_v9` were already correct.

## Not corrected, deliberately

The 69 `.pt` checkpoints under the v9 result directories carry
`"dataset_version": "v8"` in their saved payload. Nothing reads that field —
the authoritative provenance is the run CSV — and rewriting 69 trained
checkpoints to correct an inert string is a worse risk than leaving it. It is
recorded here instead.

## Root cause, now fixed

`review_and_modify/scripts/dataset_paths.py` is a new single source of truth.
Every script resolves its dataset paths, result directories and provenance
fields through it, from one `VERSION` (default `v9`, override with
`INS_DATASET_VERSION`). No script contains a version literal any more, so the
label cannot drift from the data again.

The same change closed a second, larger hazard: the eval-block scripts
(A6, A8, A9, A10, A14, A18b, A18) still pointed at `full_dataset_phase1_v8`
with no override, and wrote to unversioned output directories. Run as they
stood, they would have evaluated v9-trained models on **v8 spectra** and
overwritten the v8 results that the digest's v8→v9 delta table is built from.
They now read v9 and write to `A6_v9`, `A8_v9`, `A9_v9`, `A10_v9`, `A14_v9`,
`A18b_v9`, `A18_v9`.

`A2` is intentionally left unversioned: the metadata family models take
(T, c, E) only and never touch a spectrum, so their results are identical
across v7/v8/v9.
