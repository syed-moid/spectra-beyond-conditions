# Provenance ledger — DRAFT_v8_9

One row per table and figure. Columns: the generator version the numbers were produced on, the
evaluation set, **the dataset file the models were trained on**, the rule that selected the model
or checkpoint, the aggregation convention, the file the numbers are read from (relative to
`manuscript_phase1/results/revision/`), and the `run_id` family.

**On the two v10 dataset files.** `full_dataset_phase1_v10` and `ablation_v10_both_seed{0,1,2}`
are independent draws of the **same generator configuration** — identical augmentation-config
SHA-256 `3271f151…`, the unmodified `both` latent setting, 5000 training spectra over the same
condition support — differing in the master seed and in which evaluation blocks they carry (the
main file has the three hold-out blocks, each realization has a 2000-spectrum replicate block
instead). §2.3 states this. The column below says which file each row's models saw, so "the
dataset" is never ambiguous.

The ledger is built **before** the prose. Any paragraph whose numbers are not readable from the
file named in its row is a defect, not a rounding difference. Three defects in DRAFT_v8_2 were
found this way and are recorded at the foot of this file.

**Version convention.** `v10` is the current generator. `v9` rows are supplementary tables
deliberately retained; each says why. No main-text row may read anything but v10.

---

## Main text

| # | table / figure | gen | evaluation set | **trained on** | selection rule | aggregation | source file | run_id family |
|---|---|---|---|---|---|---|---|---|
| T1 | §2.4 degradation model | v10 | — (generator specification) | — (no trained model) | — | nominal values at each severity | `A0` report §6, generator source | — |
| T2 | §2.5 architectures | — | — (specification) | — (no trained model) | — | — | `a4_train_v8.py`, `a17_arch_sweep.py` | — |
| T3 | §3.1 parameter recovery (ω₀, Γ) | v10 | `test_v10`, s = 1 | `ablation_v10_both_seed{0,1,2}` | val split of each realization, frozen before the test sets were drawn | mean over 9 runs (3 realizations × 3 training seeds) | `A24_v10/parameter_recovery_v10.csv` | `a24_<model>_ds<0-2>_s<42-44>` |
| T4 | §3.2 merit prediction vs the metadata bound | v10 | `test_v10`, s = 1 | `ablation_v10_both_seed{0,1,2}` | as T3 | mean ± SD over 9 runs | `A22_v10/test_eval_runs.csv`; bound from `A1_v10/oracle_floor.json` | `a21_<model>_ds<0-2>_s<42-44>` |
| T5 | §3.2 pairwise model comparisons, Holm-corrected | v10 | `test_v10`, s = 1 | `ablation_v10_both_seed{0,1,2}` | as T3 | paired on realization means, df = 2; Holm across all 15 | `A22_v10/pairwise_holm.csv` | as T4 |
| T6 | §3.3 hold-outs | v10 | `full_dataset_phase1_v10` splits `val`, `holdout_c`, `holdout_E`, `holdout_T`; `holdout_c_extrap_v10` | `ablation_v10_both_seed{0,1,2}` | as T3 | mean over 9 runs; both metadata baselines per row | `A10_v10/holdout_table_v10.csv` | `a26_<model>_ds<0-2>_s<42-44>` |
| T7 | §3.4 severity sweep | v10 | `test_v10`, s ∈ {0.5, 1, 2, 4} | `ablation_v10_both_seed{0,1,2}` | as T3 | mean over 9 runs | `A22_v10/test_eval_runs.csv` | as T4 |
| T8 | §3.4 recoverability boundaries | v10 | `replicate_test_v10`, s ∈ {0.25 … 12} | `ablation_v10_both_seed{0,1,2}` | as T3 | mean over 9 runs; pairwise accuracy primary | `A24_v10/ranking_by_severity_v10.csv` | as T3 |
| T9 | §3.4 ranking at s = 1 with shuffle nulls | v10 | `replicate_test_v10`, s = 1 | `ablation_v10_both_seed{0,1,2}` | as T3 | 200 permutations + 200 derangements, mean and SD of each | `A24_v10/ranking_by_severity_v10.csv` | as T3 |
| T10 | §3.5 which channel carries the signal | v10 | `ablation_v10_{both, xi1_only, xi2_only, alpha_per_mode}_seed{0,1,2}` | `ablation_v10_{setting}_seed{0,1,2}` | val split of each ablation dataset | mean over the 3×3 grid | `A7_v10/oracle_per_setting.csv`, `A7_v10/rho_per_setting.csv` | `a7_<setting>_ds<0-2>` |
| T11 | §3.6 fitting versus learning | v10 | `test_v10`, s = 1, the 600-spectrum subset `linspace(0, N−1, 600)` | `ablation_v10_both_seed{0,1,2}` | as T3 for the network rows; fits have no selection | networks: mean over 9 runs; fits: single pass; conditions-only: MC conditional median | `A23_v10/network_vs_fit_test.csv` | `a25_<model>_ds<0-2>_s<42-44>`, `a23_<method>` |
| F1 | parameter recovery | v10 | `test_v10`, the 600-spectrum subset | `ablation_v10_both_seed{0,1,2}` | as T3 | **ensemble-mean predictions** (the 9 runs' predictions averaged per spectrum), which is a different statistic from the mean per-run error in T3 and F8; fits single pass, failures replaced | `A23_v10/primary_per_spectrum.csv` | `a32_primary` |
| F2 | severity map with fitting at s = 1 (log M) | v10 | `test_v10`, the 600-spectrum subset — the same cases the fits use | `ablation_v10_both_seed{0,1,2}` | as T3 for the learning rows | learning: per-run MAE, mean ± SD over 9 runs (no ensemble); fitting single pass, **failures replaced**; reference MC conditional median, 20000 draws | `A23_v10/primary_per_spectrum.csv`, `A23_v10/network_by_severity_fitting_subset.csv`, `A23_v10/conditions_only_reference.json` | `a32_primary`, `a34_network_subset` |
| F3 | recoverability boundaries (pairwise accuracy) | v10 | `replicate_test_v10` | `ablation_v10_both_seed{0,1,2}` | as T3 | mean over 9 runs | `A24_v10/ranking_by_severity_v10.csv` | as T3 |
| F4 | hold-out heat map | v10 | as T6 | `ablation_v10_both_seed{0,1,2}` | as T3 | mean over 9 runs | `A10_v10/holdout_table_v10.csv` | as T6 |
| F5 | channel ablation | v10 | as T10 | `ablation_v10_{setting}_seed{0,1,2}` | as T10 | as T10 | `A7_v10/oracle_per_setting.csv`, `A7_v10/rho_per_setting.csv` | as T10 |
| F6 | example spectra | v10 | `test_v10` | — (no trained model) | — | six drawn spectra, indices recorded in the CSV | `manuscript_phase1/data/full_dataset_phase1_v10/dataset.npz` | — |
| F7 | representative converged fits, two regimes | v10 | `test_v10`, the 600-spectrum subset | — (no trained model) | median Γ/ω₀ **converged** case per regime, selected in the plotting code | one spectrum per panel; curves rendered from the stored optimized parameter vectors | `A23_v10/primary_per_spectrum.csv`, `A23_v10/fit_parameters_dho_matched.csv`, `A23_v10/fit_parameters_dho_matched_prior.csv`, `manuscript_phase1/data/test_v10/dataset.npz` | `a32_primary` |
| F8 | model-family bars | v10 | `test_v10`, s = 1, the 600-spectrum subset — **one population for every series** | `ablation_v10_both_seed{0,1,2}` | as T3 | networks: per-run MAE, mean ± SD over 9 runs (no ensemble); fits single pass, failures replaced; reference MC conditional median | `A23_v10/network_runs_on_fitting_subset.csv`, `A23_v10/primary_per_spectrum.csv`, `A23_v10/conditions_only_reference.json` | `a34_network_subset`, `a32_primary` |

## Supplement

| # | table / figure | gen | evaluation set | **trained on** | selection rule | aggregation | source file | run_id family |
|---|---|---|---|---|---|---|---|---|
| S-T1 | interpolation tables, condition sampling | v10 | — (specification) | — (no trained model) | — | — | generator source, `A0` report §1–§3 | — |
| S-T2 | complete architecture specifications | — | — | — (no trained model) | — | — | `a4_train_v8.py`, `a17_arch_sweep.py` | — |
| S-T3 | fitting objective, priors, bounds, initialisation, failure handling | v10 | — (specification) | — (no trained model) | — | — | `a3_fitting_baselines.py`, `A23_v10/run_sidecar.json` | — |
| S-T4 | variance components | v10 | `test_v10`, s = 1 — **one common set**, not per-realization val | `ablation_v10_both_seed{0,1,2}` | as T3 | one-way random effects over the 3×3 grid | `A21_v10/variance_components_test.csv` | as T4 |
| S-T5 | protocol health; retired vs extended stopping rule | v10 | `full_dataset_phase1_v10` split `val` | `full_dataset_phase1_v10` | — | per-run stopping diagnostics; 5 seeds per arm | `A21_v10/training_runs.csv`; `A4_v10/training_runs.csv`, `A4_v10_extended/training_runs.csv` | `a4_*` |
| S-T6 | validation rows of the hold-out table | v10 | `full_dataset_phase1_v10` split `val` | `ablation_v10_both_seed{0,1,2}` | as T3 | mean over 9 runs | `A10_v10/holdout_table_v10.csv` | as T6 |
| S-T7 | feature-set ablation | v10 | `full_dataset_phase1_v10` split `stress_base`, s = 1 | — (no trained model) | — (no model selection) | mean ± SD over 5 GBM seeds | `A3_v10/feature_set_ablation.csv` | `a27_feat_gbm_s<42-46>` |
| S-T8 | occlusion region shares | v10 | `full_dataset_phase1_v10` split `stress_base` | `ablation_v10_both_seed{0,1,2}` + `full_dataset_phase1_v10` (tuned) | as T3 | mean over models, per severity | `A9_v10/region_shares.csv` | `a9_<model>_s<seed>` |
| S-T9 | architecture sweeps (patch width, kernel width) | v10 | `full_dataset_phase1_v10` split `val` | `full_dataset_phase1_v10` | per-variant val selection | seed ensemble and mean over seeds, both reported | `A17_v10/variant_runs.csv`, `variant_summary.csv`, `variant_paired_vs_base.csv`; `A19_v10/confirmation_heldout.csv` | `a17_*`, `a19_*` |
| S-T10 | local encoding scale / saturation onset | v10 | `full_dataset_phase1_v10` split `val` | `full_dataset_phase1_v10` | per-variant val selection | mean ± SD over 3 training seeds | `A17_v10/variant_summary.csv` | `a17_*` |
| S-T11 | energy-tilt sensitivity | v10 | `full_dataset_phase1_v10` split `stress_base` | `full_dataset_phase1_v10` (tilt-trained, 3 runs) | tilt-trained checkpoints | mean over 3 runs | `A18b_v10/energy_tilt_sensitivity.csv`, `A18b_v10_tilt/tilt_trained_beta_grid.csv` | `a18b_*` |
| S-T12 | occupation-factor control | **v8, both arms** | `full_dataset_phase1_v8` val split | `full_dataset_phase1_v8` (corrected arm); same split under v7 generator semantics (defective arm) | v7 training protocol in both arms | mean ± SD over 5 training seeds per arm | `A4_control_buggy_bose/control_runs.csv`, `A4/training_runs.csv` (**literal paths, not version-resolved**) | `a4_control_*`, `a4_*` |
| S-T13 | Appendix A literature context | — | published literature | — (no trained model) | — | classification per row; no numerical conversions retained | `REFERENCES_VERIFIED.md`, sources cited per row | — |
| S-T14 | prior-residual diagnostic (S8.3) | v10 | `test_v10`, the 600-spectrum subset | — (no trained model) | — | per-fit trajectory counts over 600 fits | `A23_v10/prior_clip_diagnostic.csv`, `.json` | `a29_prior_diag` |
| S-T15 | Figure 2 underlying table (S5.7) | v10 | `test_v10`, the 600-spectrum subset — **one common population for all three families** | `ablation_v10_both_seed{0,1,2}` | as T3 for the learning rows | learning: per-run mean over 9 runs, no ensemble; fitting: single pass, **failures replaced**; reference: MC conditional median, 20000 draws | `A23_v10/primary_per_spectrum.csv`, `A23_v10/network_by_severity_fitting_subset.csv`, `A23_v10/conditions_only_reference.json` | `a32_primary`, `a34_network_subset` |
| S-T16 | fitting across severity (S5.7.1) | v10 | `test_v10`, the 600-spectrum subset, s ∈ {0.25, 0.5, 2, 4} | — (no trained model) | — | single pass per severity, **failures replaced**, fixed bounds at every severity | `A23_v10/severity_map_fits.csv`, `A23_v10/severity_map_verdict.json` | `a33_severity_map` |
| S-F1 | local encoding scale | v10 | as S-T10 | `full_dataset_phase1_v10` | as S-T10 | as S-T10 | `A17_v10/variant_summary.csv` | `a17_*` |
| S-F2 | one-factor degradation sweeps, four panels | v10 | `test_v10`, each degradation channel varied alone | `ablation_v10_both_seed{0,1,2}` | as T3 | ensemble over the 9 runs; x axes are the physical level, not the combined severity s | `A8_v10/one_factor_sweeps.csv`, `A8_v10/one_factor_background_shaped.csv` | `a8_dense_sweep` |
| S-F3 | occlusion | v10 | as S-T8 | as S-T8 | as S-T8 | as S-T8 | `A9_v10/occlusion_sensitivity.csv` | as S-T8 |
| S-F4 | conditional spread of log M | v10 | `full_dataset_phase1_v10`, val + stress pooled | — (no trained model) | — | from the A1 oracle | `A1_v10/conditional_spread_vs_conditions.csv` | `a1_v10` |
| S-F5 | protocol sensitivity | v10 | as S-T5 | as S-T5 | — | 5 seeds per arm, **both arms on v10** | `A4_v10/training_runs.csv`, `A4_v10_extended/training_runs.csv` | `a4_*` |
| S-F6 | occupation-factor control | **v8, both arms** | as S-T12 | as S-T12 | as S-T12 | as S-T12 | `A4_control_buggy_bose/control_runs.csv`, `A4/training_runs.csv` (**literal paths**) | as S-T12 |
| S-F7 | feature ablation | v10 | as S-T7 | — (no trained model) | as S-T7 | as S-T7 | `A3_v10/feature_set_ablation.csv` | as S-T7 |
| S-F8 | replicate spread vs Monte Carlo | v10 | `replicate_eval_v10` | — (no trained model) | — | 250 tuples | `A1_v10/oracle_convergence.csv` | `a1b_v10` |
| S-F9 | energy-tilt sensitivity | v10 | as S-T11 | as S-T11 | as S-T11 | as S-T11 | `A18b_v10_tilt/tilt_trained_beta_grid.csv` | as S-T11 |

---

## Version coverage

**Every row above reads v10 except `S-T12` and `S-F6`**, which are v8 **by design**: they are the
record of a corrected generator defect, and reproducing them on v10 would remove the defect they
exist to document.

Earlier drafts retained several v9 rows because each needed *retraining* rather than
re-evaluation — results are determined by the generator source rather than the dataset file
(§2.10), so a v9-trained checkpoint cannot simply be scored on v10 spectra. That retraining has
since been done: the channel ablation (54 runs), the architecture sweep (21), the family-base grid
(29) and the tilt-trained models (3), followed by the evaluation-only analyses that depend on them.

The deferral argument was also tested rather than merely asserted. The v9 → v10 change alters the
stored spectra by a relative L2 of **8.0 × 10⁻⁷**, and the T = 600 K hold-out — where the central
peak is gated off — is bitwise identical between versions. The prediction was that nothing would
move. On recomputation the channel-ablation oracle reproduced to three decimals and its model
results to within training-seed noise, and the occlusion shares moved from 0.49–0.75 to 0.50–0.73
(acoustic) and 0.001–0.052 to 0.000–0.049 (central peak). No conclusion moved. What v10 buys is
not a different spectrum but an **untouched test set**.

## The §3.5 exception, and its removal

Round 45 §1 marked the §3.5 channel-ablation table and Figure 5 as "(E) recomputed on v10" while
also forbidding training. Those rows are models trained on four different ablated latent models,
so they could not be recomputed inside the round and were retained as v9 and labelled.

**They have since been retrained on v10** (54 runs: ST-5a on all four settings, plus the 1D CNN
on `both` and `alpha_per_mode`, over three dataset realizations × three training seeds). The
exception is removed: every main-text row now reads v10, and
`scripts/a12_main_text_provenance.py` carries an empty exception list.

The recomputation is also a check on the retention argument made above. The v10 oracle per
setting is identical to the v9 value at three decimals — as it must be, since the latents are
bitwise unchanged — and the model results move only by training-seed noise:

| setting | oracle v9 → v10 | model MAE v9 → v10 | ρ(log M) v9 → v10 |
|---|---|---|---|
| both | 0.605 → 0.605 | 0.196 → 0.197 | 0.900 → 0.897 |
| ξ₁ only | 0.608 → 0.608 | 0.201 → 0.201 | 0.895 → 0.895 |
| α per mode | 0.587 → 0.587 | 0.194 → 0.191 | 0.896 → 0.899 |
| ξ₂ only | 0.025 → 0.025 | 0.126 → 0.127 | 0.200 → 0.206 |

The shared-α control reproduces as well: cross-mode Pearson 0.196 → 0.005 on v10 (0.196 → 0.005
on v9), with the per-mode marginal unchanged (P50 0.1955 against 0.1960). No conclusion moved,
which is what the 8.0 × 10⁻⁷ spectral difference predicted. The retained v9 supplementary rows
rest on the same argument, now with this as direct evidence for it.

## Defects this ledger found in earlier drafts

1. **§4.1 cited `A3_v10/feature_set_ablation.csv`, which did not exist.** The numbers (0.526,
   0.398, 0.334) were read from `A3_v9`. Fixed by recomputing the ablation on v10
   (`a27_feature_ablation_v10.py`) rather than by relabelling the citation.
2. **§3.6's table cited `A23_v10/fit_summary_test.csv`, which contains only the three fitting
   methods.** The tuned-CNN and ST-5a rows in that table came from a different evaluation, on a
   different spectrum subset, and were not readable from the cited file. Fixed by evaluating the
   networks on the *identical* 600-spectrum subset A23 fits and writing one combined table
   (`A23_v10/network_vs_fit_test.csv`, `a25_network_vs_fit.py`).
3. **§3.3 cited `A10_v9/holdout_table.csv` for a main-text claim**, with one run per model and
   one metadata baseline. Fixed by recomputing the hold-outs on v10 with nine runs per model and
   both baselines in every row (`a26_holdouts_v10.py`).

None of the three was a wrong number in the sense of a miscalculation. All three were numbers a
reader could not reach from the file the manuscript pointed them to, which for a paper whose
argument is about provenance is the more serious failure.
