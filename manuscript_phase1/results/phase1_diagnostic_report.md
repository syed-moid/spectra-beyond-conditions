# Phase 1 acceptance report (Step F)

- Dataset: `/Users/syed.a.moid/Documents/WORKSPACES/PHD/INS_ML/spectra-beyond-conditions/manuscript_phase1/data/full_dataset_phase1/dataset.npz`
- ST ablation run dir: `/Users/syed.a.moid/Documents/WORKSPACES/PHD/INS_ML/spectra-beyond-conditions/manuscript_phase1/results/phase1_st_run`
- Phase 0 NL-cond floor (historical reference): **0.0048**
- Phase 1 NL-cond floor (live, new gate basis): **0.6232**
- Live acceptance threshold (0.70 x Phase 1 floor): **0.4363**
- Gate: spectrum-only ST in_dist MAE_logM <= threshold (>= 30% below floor)

## In-distribution gate (val, severity=1)

| Arch | ST MAE_logM (mean ± std, 3 seeds) | gap vs floor | pass |
|------|-----------------------------------|--------------|------|
| 5a | 0.2960 ± 0.0148 | +52.5% | PASS |
| 5b | 0.4067 ± 0.0287 | +34.7% | PASS |

## Per-severity breakdown (ST mean MAE_logM vs NL-cond floor)

| Severity | NL-cond floor | ST-5a | ST-5b |
|---|---|---|---|
| 0.25 | 0.6510 | 0.1989 | 0.2576 |
| 0.5 | 0.6510 | 0.2069 | 0.3059 |
| 1.0 | 0.6510 | 0.2676 | 0.4073 |
| 2.0 | 0.6510 | 0.4258 | 0.6023 |
| 4.0 | 0.6510 | 0.7788 | 0.9713 |

## Verdict

**GATE PASSED** — spectrum-only ST beats the Phase 1 NL-cond floor by >= 30% (best: 5a, gap +52.5%). Option 3 (latent-microphysics) is validated; Phase 2 may proceed.
