# Spectra beyond conditions: benchmarking information recovery in inelastic neutron scattering

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20332582.svg)](https://doi.org/10.5281/zenodo.20332582)

Code, data, trained-model checkpoints and numerical results for:

> **Spectra beyond conditions: benchmarking information recovery in inelastic neutron scattering.**
> Syed A. Moid, Ronin Institute for Independent Scholarship.
> *Journal of Applied Crystallography*, submitted. Release tag `[[TAG]]`,
> data deposit `[[ZENODO-VERSION]]`.

---

## What this is

A machine-learning model given both a spectrum and its measurement conditions can score well while
reading only the conditions. This package makes that distinction quantitative and measures how much
a spectrum adds on top of it.

The reference is the **conditions-only Bayes risk under absolute loss** — the error of the optimal
predictor that sees the measurement conditions `(T, c, E)` and nothing else — estimated by Monte
Carlo from the generative model that produced the data. Spectral models are benchmarked against it
on simulated inelastic neutron spectra of a perovskite soft-mode system, in which two latent
channels perturb linewidth and frequency independently of the conditions. Because a conditions-only
predictor cannot order realizations that share conditions, **within-condition pairwise ranking
accuracy**, whose chance level is exactly 0.5, isolates the spectral contribution without a fitted
null. The package also compares learning against parametric fitting — a matched forward-model fit,
the same fit with a penalty built from the generator's soft-mode marginals, and a `W = 0` control
that isolates the penalty from the solver. **Everything here is simulated; performance on measured
spectra is unvalidated.**

## Layout

```
sbc/               importable package: generator, latents, augmentations, models, evaluation harness
scripts/           dataset generation and training, from the original package
scripts/revision/  the revision's analysis tasks a1-a38, the figure generator, the pipeline driver
configs/           parameter card and augmentation configuration
results/           the numerical record — one directory per analysis task, per generator version
figures/           one CSV per figure, each with a provenance header naming its population,
                   sample count, checkpoints, aggregation and failure policy
tests/             generator invariants, the fitting lineshape, the log-M failure policy
data/             the v10 dataset; the rest are on Zenodo, see "Getting the data"
docs/             pipeline documentation and literature notes
legacy/phase0/     the deterministic-target diagnostic that motivated the reformulation
```

Install the package before running anything: `pip install -e .`

`PROVENANCE.md` has one row per table and figure in the paper: the generator version, the
evaluation set, **the dataset the models were trained on**, the selection rule, the aggregation
convention and the file the numbers are read from. It is the first place to look.

## Generator versions

The generator was corrected three times during revision. **Results are determined by the generator
source, not by the dataset file**: a checkpoint trained on one version and evaluated on spectra from
another is not a valid substitute for retraining, and doing exactly that once during this revision
produced an MAE of 1.69 against a true 0.194. Every main-text number is on **v10**.

The version is identified by the **SHA-256 of `sbc/data/spectrum_generator.py`**, which is pinned
into every result table and every checkpoint's run metadata. Git commits do not separate these
versions: the corrections were made in the working tree and are committed for the first time at
`[[TAG]]`, so v8, v9 and v10 datasets all record the same repository commit. The source hash does
separate them, and so does the data — `scripts/revision/a14_data_check.py` recomputes the detailed-balance
residual for any dataset file.

| version | what changed | effect | generator source SHA-256 |
|---|---|---|---|
| v7 | the submitted version | — | `306dc767…` |
| v8 | per-spectrum thermal weight evaluated at ω₀ rather than at ω | occupation scale corrected | `9e68bfed…` |
| v9 | the lineshape was even: S(ω)/S(−ω) did not follow exp(ω/k_BT) | corrected to the standard damped-harmonic-oscillator response | `4035f7a3…` |
| v10 | the central peak was an even Lorentzian, so the *assembled* spectrum broke detailed balance even though the oscillator part did not | relaxational form; assembled-spectrum residual 2.5 × 10⁻⁵ → 9.0 × 10⁻⁸ | `8cb82ecf…` |

`tests/test_generator.py` and `tests/test_fit_lineshape.py` fail on each pre-correction form; that
is how the corrections are established rather than asserted. `CORRECTIONS.md` records what each one
changed in the results.

## Getting the data

Datasets and checkpoints are on Zenodo (`[[ZENODO-VERSION]]`), not in git: 1.0 GB of datasets and
1.46 GB of checkpoints. Unpack the deposit so that `data/` and `results/` sit at the repository root, matching the layout
above. The v10 training dataset (`data/full_dataset_phase1_v10/`) is already in the repository; the
deposit adds the evaluation and ablation sets and the checkpoints.

Most tables and figures rebuild from the committed result tables alone, with no deposit. Two need
the spectra themselves and so need `data/test_v10/` from the deposit: Figure 6 (example spectra)
and Figure 7 (fit overlays). `scripts/revision/a35_zenodo_manifest.py` reports exactly which
declared inputs are absent from your tree, so it is the quickest way to see what you still need. Every dataset can also be regenerated from source — see
"Regenerating the datasets".

## Installing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`env/freeze.txt` records the exact interpreter environment the reported results were produced in.
The results were computed on Apple silicon with the PyTorch MPS backend;
`results/revision/A12_v10/mps_reproducibility.csv` records the reproducibility check for that
backend.

## Reproducing a specific table or figure

Each main-text table and figure rebuilds with one command from the released results, with **no
training**. `INS_DATASET_VERSION=v10` selects the generator version; it is required, because the
scripts refuse to mix versions silently.

| paper item | what it is | command | reads |
|---|---|---|---|
| Table 1 (§2.5) | architecture specifications | — (specification) | `scripts/revision/a17_arch_sweep.py` |
| Table (§3.1) | parameter recovery, ω₀ and Γ | `python scripts/revision/a28_fill_a24_tables.py` | `A24_v10/parameter_recovery_v10.csv` |
| Table (§3.2) | merit prediction against the reference | `python scripts/revision/a12_digest.py --force` | `A22_v10/test_eval_runs.csv`, `A1_v10/oracle_floor.json` |
| Table (§3.2) | pairwise comparisons, Holm-corrected | `python scripts/revision/a22_test_eval.py` | `A22_v10/pairwise_holm.csv` |
| Table (§3.3) | hold-outs | `python scripts/revision/a26_holdouts_v10.py` | `A10_v10/holdout_table_v10.csv` |
| Tables (§3.4) | boundaries and ranking nulls | `python scripts/revision/a28_fill_a24_tables.py` | `A24_v10/ranking_by_severity_v10.csv` |
| Table (§3.5) | channel ablation | `python scripts/revision/a7_postprocess.py` | `A7_v10/oracle_per_setting.csv`, `rho_per_setting.csv` |
| Table (§3.6), S5.5, S5.7 | fitting versus learning | `python scripts/revision/a32_primary_comparison.py` then `a28_fill_a24_tables.py` | `A23_v10/primary_fit_summary.csv`, `primary_gap_fractions.json`, `primary_per_spectrum.csv` |
| S5.7.1 | fitting across severity | `python scripts/revision/a33_severity_map.py` | `A23_v10/severity_map_fits.csv` |
| Figures 1–8, S1–S9 | all 17 figures | `python scripts/revision/generate_figures_v10.py` | the CSVs listed in `ZENODO_MANIFEST.md` |
| any figure alone | one figure | `python scripts/revision/generate_figures_v10.py --only fig7` | as above |

`scripts/revision/a32_primary_comparison.py` re-fits 3000 spectra (about 45 minutes). If you only need the
derived log-M columns rebuilt — after changing the failure policy, say — `scripts/revision/a38_reaggregate_logM.py`
does that from the saved per-spectrum table without refitting, and is idempotent.

Two checks are worth running on any tree you have changed:

```bash
INS_DATASET_VERSION=v10 python scripts/revision/a12_main_text_provenance.py   # every main-text number is on v10
INS_DATASET_VERSION=v10 python scripts/revision/a36_number_sweep.py           # every number in the text exists in a source file
INS_DATASET_VERSION=v10 python scripts/revision/a35_zenodo_manifest.py        # no figure is older than its own input
```

## Regenerating the datasets and retraining

This is the expensive path: about 30 hours on one Apple-silicon machine for the full set.

```bash
# 1. datasets
INS_DATASET_VERSION=v10 python scripts/revision/gen_eval_sets_v8.py          # test, replicate, hold-out sets
INS_DATASET_VERSION=v10 python scripts/revision/a7_generate_ablation.py      # the 12 latent-ablation sets
INS_DATASET_VERSION=v10 python scripts/revision/regenerate_augmented_blocks.py   # the replicate severity blocks

# 2. everything else, in dependency order, resumable and deadline-aware
INS_DATASET_VERSION=v10 python scripts/revision/run_pipeline.py --status     # what is done, what is left
INS_DATASET_VERSION=v10 python scripts/revision/run_pipeline.py --until 08:00 --keep-going
```

`run_pipeline.py` keeps a per-version state file and **refuses to run against a state file written
for a different generator version** — that guard exists because a stale state file would mark every
stage done and silently skip the entire rerun. `--keep-going` continues past a failing stage and
reports the failures at the end; `--dry-run` prints the plan with time estimates; `--only <stage>`
runs one stage.

The augmented severity blocks are **regenerated, not shipped**: they are a deterministic replay of
the stored augmentation configuration, and the replay is asserted against a stored reference inside
each dataset file.

## Tests

```bash
python -m pytest tests -q
```

`tests/test_logm_fallback.py` pins the rule that a failed fit is scored at the conditional median
of log M rather than at merit() evaluated at the median parameters — two estimators that are close
on this data and are not the same thing — and asserts that the shipped per-spectrum table already
carries it. `tests/test_generator.py` asserts the generator's invariants, including detailed
balance as a law in ω rather than at a single point — the original test asserted a ratio at the mode frequency,
which is constant per spectrum, and therefore **passed on defective code**. `tests/test_fit_lineshape.py`
(35 cases) asserts the same law for the fitting baselines' lineshape at five energies, three mode
frequencies and two temperatures, plus continuity at ω = 0 and the full denominator. Both fail on
every pre-correction form; that is what they are for.

## Licence

Code: MIT (`LICENSE`). Datasets, numerical results, figure data, configuration files and
documentation: CC BY 4.0 (`DATA_LICENSE.md`).

## Citation

```bibtex
@article{moid_spectra_beyond_conditions,
  author  = {Moid, Syed A.},
  title   = {Spectra beyond conditions: benchmarking information recovery
             in inelastic neutron scattering},
  journal = {Journal of Applied Crystallography},
  year    = {2026},
  note    = {Code and data: \url{https://doi.org/10.5281/zenodo.20332582},
             version [[ZENODO-VERSION]], release tag [[TAG]]}
}
```

`[[TAG]]` and `[[ZENODO-VERSION]]` are filled in when the release is tagged and the deposit
published. They are the only placeholders in this file.
