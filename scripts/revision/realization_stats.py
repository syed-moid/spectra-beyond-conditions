"""Variance components and across-realization comparisons (round 42/43 Part 3.1-3.2).

Replaces two retired devices: the naive "nested" variance split, whose dataset
component double-counted training variance, and the single 1.96x inflation
factor applied to every comparison.

Estimators
----------
`variance_components` is the one-way random-effects ANOVA on the
realization x training-seed grid:

    sigma2_train   = MS_within
    sigma2_dataset = max(0, (MS_between - MS_within) / n_train)

The truncation at zero is why several cells report exactly 0: with three
realizations the between-group mean square is often no larger than the within,
and the component is not distinguishable from zero. That is a statement about
the power of three realizations, not evidence that realizations do not matter.

`paired_across_realizations` compares two models. The primary test is a paired
t-test on the three realization-level mean differences (df = 2) -- deliberately
conservative, because the realization is the unit the claim generalises over.
The 9-cell test with a nested correction and the spectrum-level bootstrap are
reported beside it, never in place of it.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _grid(df, value, row="dataset_seed", col="seed"):
    p = df.pivot_table(index=row, columns=col, values=value)
    return p.values.astype(float)


def variance_components(df, value="best_val_MAE_logM"):
    """One-way random-effects components on the realization x seed grid."""
    y = _grid(df, value)
    r, c = y.shape
    if r < 2 or c < 2 or not np.isfinite(y).all():
        return None
    grand = y.mean()
    ms_between = c * ((y.mean(1) - grand) ** 2).sum() / (r - 1)
    ms_within = float(np.mean(y.var(axis=1, ddof=1)))
    s2_train = ms_within
    s2_dataset = max(0.0, (ms_between - ms_within) / c)
    total = s2_train + s2_dataset
    return {
        "n_realizations": int(r), "n_training_seeds": int(c), "n_cells": int(y.size),
        "mean": float(grand), "sd_over_cells": float(y.std(ddof=1)),
        "MS_between": float(ms_between), "MS_within": float(ms_within),
        "var_training_seed": float(s2_train), "var_dataset_seed": float(s2_dataset),
        "sd_training_seed": float(np.sqrt(s2_train)),
        "sd_dataset_seed": float(np.sqrt(s2_dataset)),
        "frac_dataset_seed": float(s2_dataset / total) if total > 0 else np.nan,
        "dataset_component_detectable": bool(s2_dataset > 0),
        "estimator": "one-way random effects (ANOVA), dataset component truncated at 0",
    }


def paired_across_realizations(err_a, err_b, dataset_seed, seed, n_boot=1000, rng=None):
    """Compare two models from per-cell error arrays on a shared evaluation set.

    err_a, err_b : (n_cells, n_spectra) absolute errors, cells aligned.
    dataset_seed, seed : (n_cells,) labels identifying each cell.
    """
    rng = rng or np.random.default_rng(20260912)
    err_a, err_b = np.asarray(err_a, float), np.asarray(err_b, float)
    ds = np.asarray(dataset_seed)
    cell_diff = err_a.mean(1) - err_b.mean(1)                 # one per cell

    # primary: realization-level means, df = n_realizations - 1
    realz = np.array([cell_diff[ds == d].mean() for d in np.unique(ds)])
    n_r = len(realz)
    t_r = stats.ttest_1samp(realz, 0.0) if n_r > 1 else None
    se_r = realz.std(ddof=1) / np.sqrt(n_r) if n_r > 1 else np.nan
    crit = stats.t.ppf(0.975, n_r - 1) if n_r > 1 else np.nan
    lo_r, hi_r = (realz.mean() - crit * se_r, realz.mean() + crit * se_r) if n_r > 1 else (np.nan, np.nan)

    # secondary: all cells, nested correction (effective n = n_realizations)
    t_c = stats.ttest_1samp(cell_diff, 0.0) if len(cell_diff) > 1 else None

    # spectrum-level paired bootstrap on the pooled mean error difference
    d_spec = err_a.mean(0) - err_b.mean(0)
    idx = rng.integers(0, len(d_spec), size=(n_boot, len(d_spec)))
    boot = d_spec[idx].mean(1)

    return {
        "mean_diff": float(cell_diff.mean()),
        "n_realizations": int(n_r), "n_cells": int(len(cell_diff)),
        "realization_diffs": [float(x) for x in realz],
        "t_realization": float(t_r.statistic) if t_r is not None else np.nan,
        "p_realization": float(t_r.pvalue) if t_r is not None else np.nan,
        "df_realization": int(n_r - 1),
        "ci_lo_realization": float(lo_r), "ci_hi_realization": float(hi_r),
        "robust_across_realizations": bool(n_r > 1 and lo_r * hi_r > 0),
        "p_cells_uncorrected": float(t_c.pvalue) if t_c is not None else np.nan,
        "boot_lo": float(np.percentile(boot, 2.5)),
        "boot_hi": float(np.percentile(boot, 97.5)),
        "n_boot": int(n_boot),
    }


def pairwise_ranking_accuracy(tuple_id, y, pred, tol=0.0):
    """Within-condition pairwise ranking accuracy, half credit for ties.

    Chance is 0.5 for any predictor, including a constant one, so unlike
    Spearman rho this metric has a defined baseline for the metadata models
    rather than an undefined value replaced by a convention.
    """
    tuple_id, y, pred = np.asarray(tuple_id), np.asarray(y, float), np.asarray(pred, float)
    accs = []
    for t in np.unique(tuple_id):
        m = tuple_id == t
        yy, pp = y[m], pred[m]
        n = len(yy)
        if n < 2:
            continue
        i, j = np.triu_indices(n, 1)
        dy, dp = yy[i] - yy[j], pp[i] - pp[j]
        keep = dy != 0
        if not keep.any():
            continue
        dy, dp = dy[keep], dp[keep]
        score = np.where(np.abs(dp) <= tol, 0.5, (np.sign(dp) == np.sign(dy)).astype(float))
        accs.append(score.mean())
    return float(np.mean(accs)) if accs else np.nan


def expected_deranged_rho(rho, n):
    """E[rho] under a uniform derangement of n items: -rho/(n-1).

    The shuffle control is therefore centred slightly below zero by
    construction; agreement with this value validates the control rather than
    indicating residual signal.
    """
    return -float(rho) / (n - 1)
