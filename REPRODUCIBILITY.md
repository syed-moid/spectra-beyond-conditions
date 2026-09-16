# Reproducibility: what determines a result here

**Results are determined by the generator source version, not by the dataset file.**

`src/data/dataset.py::InsSpectraDataset.get_augmented` (lines 121-141) reconstructs each spectrum
at run time from the stored conditions `(T, c, E)` and the stored latent draw, by calling
`src/data/spectrum_generator.py::generate_spectrum`, and then replays the stored severity-1
augmentation parameters at the requested severity. The `spectra_clean` array persisted in the
`.npz` is returned only when `severity <= 0`; no training or evaluation path reads it. The
in-code comment at `dataset.py:137-139` describes it as "a tagged-along artefact", and that is
accurate.

Two consequences that a reader reproducing this work must know:

1. **Loading a different dataset file does not change the spectra a model sees.** The Phase 1
   datasets `full_dataset_phase1/` (v7) and `full_dataset_phase1_v8/` (v8) have bitwise-identical
   conditions, latent draws, targets and augmentation parameters; they differ only in
   `spectra_clean`. Loading either one under a given generator source produces the same training
   data.
2. **To reproduce a published number, pin the generator source.**

| target | generator identity |
|---|---|
| v7 published results (Zenodo 10.5281/zenodo.20332582) | repo commit `7c44defb…` — pre-fix `bose()`, which multiplies the exponent by hbar = 0.6582 |
| v8 results | post-fix `bose()`; sha256 `9e68bfede827fe51b43b2a5e5ed6786d125c549a01b8b1c0f0b48d9be808d502` (pre-fix: `306dc76710ab7f52cfaffb21936f304353da73a56f2ff209374eaf18e2217c2c`) |
| v9 results | standard DHO lineshape, parent convention, Gamma = HWHM; sha256 `4035f7a39cc079c473d909bf15d50302164d6552ef0f0e9f59ba445176cae1bd` at generation time. A later comment-only edit that neutralises a divide-by-zero warning changed the file hash while leaving `dho()` output bitwise identical; both hashes and the verification are recorded in each v9 dataset's `PROVENANCE.md` |

## The `7c44defb` SHA and the `v7-initial` tag

The SHA stored inside the v7 dataset, `7c44defbd3c2068efe17003c6293c4858d999ccf`, **refers to an
earlier repository** and is not an object in this one — it is absent from the history, the reflog
and the dangling objects. The v7 generator source is nevertheless byte-identical across the last
six commits of this repository (sha256 `306dc767…` at `e97ec85` through `997f2a0`); the commits
after `e97ec85` touched only README, LICENSE and the DOI.

The tag **`v7-initial`** in this repository therefore marks the equivalent state: the last commit
before the corrections, whose generator source is the one the v7 results were produced with. Anyone
reproducing the v7 results should check out `v7-initial` here rather than looking for
`7c44defb`, which cannot be resolved.

`np.savez` metadata inside each dataset records `generator_git_sha` at generation time, but note
that this identifies the commit that *wrote the file*, which is not necessarily the generator a
later run imports.

## Numerical reproducibility

* Stored checkpoints evaluated on CPU reproduce their recorded validation scores to ~1e-8.
* Training on Apple MPS reproduces to ~1e-4 nats in MAE across torch versions, not bitwise. The
  v7 runs used torch 2.12.1; v8 onwards use torch 2.14.0.
* The dataset arrays `T_K`, `c_pct`, `E_kVcm` are float32 while the sampler drew float64. Any
  regeneration that wants to match the stored augmented spectra must round the conditions to
  float32 first, because the integer Poisson draw amplifies last-digit differences. The main
  dataset's `spectra_clean` was written from the float64 values and so differs slightly from what
  `get_augmented` reconstructs; since nothing reads `spectra_clean`, this affects no result, but
  the Zenodo README should state that the regenerated array is authoritative.
