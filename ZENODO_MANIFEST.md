# Zenodo deposit manifest — plotting inputs

Every file below is an input that a submitted figure declares in its CSV header.
A deposit containing these files, at these hashes, reproduces the figures in the
package when passed through `review_and_modify/scripts/generate_figures_v10.py`.

Generated 2026-09-16T18:49:01+00:00.

## Figures and the inputs they declare

| figure | PDF present | inputs |
|---|---|---|
| `fig1_parameter_recovery` | yes | `manuscript_phase1/results/revision/A23_v10/primary_per_spectrum.csv` |
| `fig2_recoverability_map` | yes | `manuscript_phase1/results/revision/A23_v10/primary_per_spectrum.csv`<br>`manuscript_phase1/results/revision/A23_v10/network_by_severity_fitting_subset.csv`<br>`manuscript_phase1/results/revision/A23_v10/conditions_only_reference.json` |
| `fig3_two_boundaries` | yes | `manuscript_phase1/results/revision/A24_v10/ranking_by_severity_v10.csv` |
| `fig4_holdout_heatmap` | yes | `manuscript_phase1/results/revision/A10_v10/holdout_table_v10.csv` |
| `fig5_channel_ablation` | yes | `manuscript_phase1/results/revision/A7_v10/oracle_per_setting.csv`<br>`manuscript_phase1/results/revision/A7_v10/rho_per_setting.csv` |
| `fig6_example_spectra` | yes | `manuscript_phase1/data/full_dataset_phase1_v10/dataset.npz` |
| `fig7_fit_overlay` | yes | `manuscript_phase1/results/revision/A23_v10/primary_per_spectrum.csv`<br>`manuscript_phase1/data/test_v10/dataset.npz`<br>`manuscript_phase1/results/revision/A23_v10/fit_parameters_dho_matched.csv`<br>`manuscript_phase1/results/revision/A23_v10/fit_parameters_dho_matched_prior.csv` |
| `fig8_model_family_bars` | yes | `manuscript_phase1/results/revision/A23_v10/network_runs_on_fitting_subset.csv`<br>`manuscript_phase1/results/revision/A23_v10/primary_per_spectrum.csv`<br>`manuscript_phase1/results/revision/A23_v10/conditions_only_reference.json` |
| `figS1_receptive_field` | yes | `manuscript_phase1/results/revision/A17_v10/variant_summary.csv` |
| `figS2_one_factor` | yes | `manuscript_phase1/results/revision/A8_v10/one_factor_sweeps.csv`<br>`manuscript_phase1/results/revision/A8_v10/one_factor_background_shaped.csv` |
| `figS3_occlusion` | yes | `manuscript_phase1/results/revision/A9_v10/occlusion_sensitivity.csv` |
| `figS4_conditional_spread` | yes | `manuscript_phase1/results/revision/A1_v10/conditional_spread_vs_conditions.csv` |
| `figS5_protocol_sensitivity` | yes | `manuscript_phase1/results/revision/A4_v10/training_runs.csv`<br>`manuscript_phase1/results/revision/A4_v10_extended/training_runs.csv` |
| `figS6_bose_control` | yes | `manuscript_phase1/results/revision/A4_control_buggy_bose/control_runs.csv`<br>`manuscript_phase1/results/revision/A4/training_runs.csv` |
| `figS7_feature_ablation` | yes | `manuscript_phase1/results/revision/A3_v10/feature_set_ablation.csv` |
| `figS8_replicate_vs_mc` | yes | `manuscript_phase1/results/revision/A14_v10/replicate_vs_mc_conditional.csv` |
| `figS9_tilt_sensitivity` | yes | `manuscript_phase1/results/revision/A18b_v10/energy_tilt_sensitivity.csv`<br>`manuscript_phase1/results/revision/A18b_v10_tilt/tilt_trained_beta_grid.csv` |

## Input files and SHA-256 (25 distinct)

| file | sha256 |
|---|---|
| `manuscript_phase1/data/full_dataset_phase1_v10/dataset.npz` | `20dba4b6c152e269d174f2e1eb2955afc01790eddc52264a8fe23f9809452829` |
| `manuscript_phase1/data/test_v10/dataset.npz` | `9f6b24b1565df803260965d81b2cdbb616aa65343c504dba94c8d025286869b8` |
| `manuscript_phase1/results/revision/A10_v10/holdout_table_v10.csv` | `b5ce46a774f3967f99cb42467e17c91a29270e92d304886df5b63bba6d941b92` |
| `manuscript_phase1/results/revision/A14_v10/replicate_vs_mc_conditional.csv` | `314ae33cfb9c4b47b090654d972ed5cf8c7b9e554575bc8cfac0a5c0ddacca0b` |
| `manuscript_phase1/results/revision/A17_v10/variant_summary.csv` | `dcd0354d6e0e9ae5f91bb00ede3ed7accd21e02e4c5ac132262d8a8f80e12f75` |
| `manuscript_phase1/results/revision/A18b_v10/energy_tilt_sensitivity.csv` | `2a136d733d29a2a4d706703076adc785a17a840f7bc9df97c823cf2a83846b77` |
| `manuscript_phase1/results/revision/A18b_v10_tilt/tilt_trained_beta_grid.csv` | `0a03f8c768887f8239d41131dd35d798e968997ca56571846ee0072b348c0fbe` |
| `manuscript_phase1/results/revision/A1_v10/conditional_spread_vs_conditions.csv` | `0a9d4f1c8742e68b54d15ca5affa45febff17c8101278fe3a9cf38ba5060d0c1` |
| `manuscript_phase1/results/revision/A23_v10/conditions_only_reference.json` | `6786614c7af4855d09cf18395fd2c685d84f924f4a7e8653a81259bfb3ba0aa2` |
| `manuscript_phase1/results/revision/A23_v10/fit_parameters_dho_matched.csv` | `574580473bab32811bc7d69540cc63e8d72237036d4a4ed63b746b6d682cfcac` |
| `manuscript_phase1/results/revision/A23_v10/fit_parameters_dho_matched_prior.csv` | `987afced4c84f6e5fe8c3c60434bd46dec71e51f8b4459fb0c00508fd2a3940d` |
| `manuscript_phase1/results/revision/A23_v10/network_by_severity_fitting_subset.csv` | `15a8c116a7166c532b8563430714f2cb7ded552c4227e6baf57e131ffc67e4b3` |
| `manuscript_phase1/results/revision/A23_v10/network_runs_on_fitting_subset.csv` | `0dbb0b8b9b26e5464888a2f5ca4c3ed26fa0831a069912368e29cf77ebfce2d4` |
| `manuscript_phase1/results/revision/A23_v10/primary_per_spectrum.csv` | `36cfc4968aaaaba8ac33b2e65c0dd5043fdbe3dc09d3accb0c68351920b13ec1` |
| `manuscript_phase1/results/revision/A24_v10/ranking_by_severity_v10.csv` | `b4d60bead3d2e448c0308acae667b0c7e3a334217a39e2bd70645ab65fa2e78c` |
| `manuscript_phase1/results/revision/A3_v10/feature_set_ablation.csv` | `95e55da4ce8253b79e70361c7f4274c12925b7e9d9f3a35945d72af8fe5c48ff` |
| `manuscript_phase1/results/revision/A4/training_runs.csv` | `a9abcf0f6c97ea0721090a97b7d0bb79df1cc534039895208d9a4bece6b86a5e` |
| `manuscript_phase1/results/revision/A4_control_buggy_bose/control_runs.csv` | `8ce4c6c2acaf8cdcc2a7ec398999eb7555b71c387d4757d42ad3f501e0ff6553` |
| `manuscript_phase1/results/revision/A4_v10/training_runs.csv` | `56db4de7c65e9a4a429c524f9351ba4c8c9da8ca078b2683c1211c8c29a32f14` |
| `manuscript_phase1/results/revision/A4_v10_extended/training_runs.csv` | `ed0d2231321446df3b9623a585a995eb439b1835fa34610c6568fc38ce93fb63` |
| `manuscript_phase1/results/revision/A7_v10/oracle_per_setting.csv` | `667ec2334158314bad8aed82300cf367ab71fdbc4f3ac0cfe7a344331177c313` |
| `manuscript_phase1/results/revision/A7_v10/rho_per_setting.csv` | `b61eea0a7d125a14aa29f308e7799e62aecf79e5bcf745e50a9058ac672f9ea4` |
| `manuscript_phase1/results/revision/A8_v10/one_factor_background_shaped.csv` | `ecdd8dc3d17894fb0416bd4564cf426dd5401aadcc79a289ff1387a4fc0c7546` |
| `manuscript_phase1/results/revision/A8_v10/one_factor_sweeps.csv` | `fceb6f8c0d1898038a6557f1390d5492f6f4a9d253f6804b621c4c3c4adfaa3e` |
| `manuscript_phase1/results/revision/A9_v10/occlusion_sensitivity.csv` | `1f18ce48432cf6551d0737e5db1d7d2ceb8a8e73cfbe844eb7d8581f03f3caf8` |

## Checks

- figures with a PDF: **17 / 17**
- additional figure artefacts deposited (not figures): **1** — fig7_fit_overlay_parameters
- declared inputs missing from the tree: **0**
- figures older than one of their inputs (stale): **0**

Every figure is present and newer than every input it names.
