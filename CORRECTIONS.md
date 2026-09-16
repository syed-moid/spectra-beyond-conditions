# Corrections relative to the initial release (v7)

The initial release of this work (repository tag `v7-initial`, Zenodo version 1 of
10.5281/zenodo.20332582) rested on a generator with three defects and on several analyses that did
not support the claims drawn from them. This file is the short record: what was wrong, what forced
the correction, and what changed. The long record — the measurements that establish each one — is
in the supplementary material. `PROVENANCE.md` states, per table and figure, the version
and file the current numbers come from.

Nothing in the current results rests on a pre-v10 computation. That is asserted, not claimed:
`scripts/a12_main_text_provenance.py` fails if any main-text row reads from a pre-v10 file.

## Generator

| version | defect | how it was found | effect |
|---|---|---|---|
| v7 → v8 | the per-spectrum thermal weight was evaluated at ω₀ rather than at ω | the detailed-balance test, rewritten | occupation scale corrected; the elastic-line discontinuity of the v7 spectra (median 15%, up to a factor 2 below 200 K) removed |
| v8 → v9 | the lineshape was even: S(ω)/S(−ω) did not follow exp(ω/k_BT) | the same test, once it asserted the right invariant | corrected to the standard damped-harmonic-oscillator response |
| v9 → v10 | the central peak was added as an even Lorentzian, so the *assembled* spectrum broke detailed balance even though the oscillator part did not | assembled-spectrum residual check | relaxational form; residual 2.5 × 10⁻⁵ → 9.0 × 10⁻⁸ |

**The test that missed all three.** The original detailed-balance test asserted a ratio taken at the
mode frequency. That ratio is constant per spectrum, so the test **passed on defective code**. It is
now four tests — the ω-dependent ratio at five energies for three mode frequencies and two
temperatures, continuity at ω = 0, the ω → 0⁺ limit, and the overdamped limit — and all four fail on
each pre-correction generator.

**What v10 actually buys.** The v9 → v10 change is numerically tiny: the corrected term is at most
0.05% of the intensity at ω = 0 and the stored spectra differ by a relative L2 of 8.0 × 10⁻⁷. What
it buys is an **untouched test set**: `test_v10` and `replicate_test_v10` were drawn with fresh
seeds after model selection was frozen. Labels, latents, conditions and splits are bitwise unchanged
across v7–v10; only the spectra differ.

## Analysis

| what was wrong | what forced it | what changed |
|---|---|---|
| overdamped fraction quoted as 63.2% | recount | that used Γ > ω₀/2; the criterion is Γ > ω₀, giving 38.0% on the test set and 34.7% on the 600-spectrum fitting subset |
| the nested variance estimator double-counted σ²_train/n_train | E[Var(dataset means)] = σ²_dataset + σ²_train/n_train, verified as exactly MS_within/3 | replaced by a one-way random-effects ANOVA |
| training-seed variance inflated by a fixed 1.96× as a proxy for realization variance | the factor had no derivation applicable per comparison | replaced by three independent dataset realizations and a paired df = 2 test, Holm-corrected across all fifteen comparisons |
| a fitted baseline claimed to sit 0.001 nats from the Bayes reference | the evaluation-set standard error is 0.015 nats | claim withdrawn; comparisons are by paired differences on the same cases |
| "the networks are not doing implicit peak fitting" | the probes do not distinguish that from an amortized inverse | claim withdrawn |
| the patch/kernel sweep described as a receptive-field result | global self-attention does not restrict a transformer to its patch width | restated as local encoding scale |
| the central peak described as setting the difficulty and carrying spectral-weight transfer | its amplitude is ≤ 0.05% of I(0) | claim withdrawn; the benchmark does not test central-peak physics |
| the hold-out conclusion drawn against a single conditions-only baseline | the two baselines disagree by up to 0.12 nats | both reported in every row |
| the fitting comparison's network rows came from a different evaluation than the fit rows beside them | the provenance ledger | both recomputed on the identical 600 spectra |
| the prior-regularized fit clipped a negative penalty to zero | found in review of the prior residual | the clip bound for 89.8% of residual evaluations and 94.0% of fits ended inside it, so most "regularized" fits were not regularized at their own solution; replaced by a direct scalar objective |
| the fitting comparison excluded each method's own failures, scoring the methods on different subsets | found in review of the failure policy | **replacement is now primary** — every method scored on all 600 cases, a failed fit replaced by the conditions-only prediction; successful-fit-only errors retained as a secondary diagnostic |
| the penalty's benefit attributed to the penalty, while the comparison also changed the optimizer | found in review of the failure policy | a **W = 0 control** was added: the same objective, solver, initialisation, bounds and tolerances with the penalty off. The solver contributes 0.004 meV overall and 0.036 meV overdamped; the gap closure is 40.0% / 52.7% against that control |
| a three-way recoverability map claimed across severity | found in review of the failure policy | narrowed to a severity-dependent **spectral** analysis supplemented by fitting at nominal quality; the unregularized procedures fail on 24–30% of spectra at s = 4 |
| the single-mode fitting baselines carried the v7-era lineshape | found in a read of the fitting code | corrected, with a 35-case test that fails on the old form; the row labelled "windowed" was the full-range fit, and both are now reported under their own names |
| failed fits' log M scored as merit(median ω₀, median Γ) rather than the conditional median of log M | found in review of the log-M fallback | the conditions-only *reference* had been corrected to the median of log M in round 49; the failed-fit predictions had not inherited it. Corrected in the driver and reaggregated from the saved per-spectrum table without refitting. The all-case log-M MAEs move by at most 0.0002 nats; the linewidth errors, failure counts and the 40.0% / 52.7% gap closures are untouched |
| Figure S6, described as a v8 control, compared a v7-semantics defective arm against a **v10** extended-protocol arm | found in review of the log-M fallback | both arms pinned to literal v8, v7-protocol paths, so the figure isolates the occupation factor as described. On five seeds per arm the two are not distinguishable |
| an "energy-offset bound" reported as widened and then scaled with severity | found in a read of the fitting code | the bound acted on the background slope; the energy offset was never restrictive. Claim withdrawn, all bounds fixed |
| the elastic-line ratio "predicted" as 1.001558 | re-derivation | that is exp(2δ/k_BT), the law at the bin *separation*; the prediction at the bin energy is 1.000779, which is what the data give |
| Appendix A applied a ÷2 conversion uniformly to literature values | frequency rows are mode positions, not widths | applied to width rows only; the quantitative calibration claim withdrawn and the frequency range described as a phenomenological design choice |

## What did not change

The central results survived every correction. The conditions-only reference remains the thing to
beat; the spectral models still beat it by a wide margin on log merit; the frequency channel remains
dispensable and the linewidth channel indispensable; and conditions alone still locate the mode
frequency better than any network does. What changed is what can be *claimed* from them.
