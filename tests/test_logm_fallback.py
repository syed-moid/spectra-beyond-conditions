"""A failed fit's log M must be the conditional median, not merit() at the medians.

Round 51. The primary driver replaced a failed fit's (omega0, Gamma) with their
own conditional medians and pushed that pair through merit(). That yields
log merit(median omega0, median Gamma), which is not median(log M | T, c, E).
Round 49 established the same distinction for the conditions-only reference and
corrected it there; the failed-fit predictions did not inherit the correction.

These tests pin both halves: that the substitution happens at all, and that the
two estimators are genuinely different quantities, so the first test cannot pass
by accident on data where they coincide.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "revision"))

from a32_primary_comparison import logM_with_fallback          # noqa: E402
from sbc.data.merit import merit                               # noqa: E402


def test_failed_case_takes_the_conditional_median():
    ok = np.array([True, False, True])
    logM_fit = np.array([-1.0, -2.0, -3.0])
    logM_cond = np.array([0.10, 0.20, 0.30])
    got = logM_with_fallback(ok, logM_fit, logM_cond)
    assert got[1] == pytest.approx(0.20), "the failed case must take logM_cond"
    assert got[0] == pytest.approx(-1.0), "a successful fit must keep its own value"
    assert got[2] == pytest.approx(-3.0)


def test_failed_case_is_not_merit_at_the_median_parameters():
    """The substituted value must not be log merit(median omega0, median Gamma).

    A synthetic condition with a spread in Gamma: the median of log M over draws
    and merit() evaluated at the median parameters differ, because log merit is
    not linear in Gamma.
    """
    T, E = 300.0, 1.0
    rng = np.random.default_rng(0)
    om = np.full(20000, 5.0)
    gm = np.exp(rng.normal(np.log(2.0), 0.6, 20000))            # lognormal spread

    logM_draws = np.log(np.clip([merit(float(a), float(b), T, E)
                                 for a, b in zip(om[:4000], gm[:4000])], 1e-9, None))
    median_logM = float(np.median(logM_draws))
    at_medians = float(np.log(max(merit(float(np.median(om)), float(np.median(gm)),
                                        T, E), 1e-9)))

    assert median_logM != pytest.approx(at_medians, abs=1e-6), (
        "the two estimators coincide on this synthetic case, so the test below "
        "would not discriminate; choose a condition with more spread in Gamma")

    got = logM_with_fallback([False], [at_medians], [median_logM])
    assert got[0] == pytest.approx(median_logM)
    assert got[0] != pytest.approx(at_medians, abs=1e-6)


def test_stored_per_spectrum_table_carries_the_fallback():
    """The shipped table must already have the fallback applied.

    Guards against the released `primary_per_spectrum.csv` drifting back to the
    pre-round-51 form if the driver is ever re-run from an older checkout.
    """
    pd = pytest.importorskip("pandas")
    p = (ROOT / "results" / "revision" / "A23_v10"
         / "primary_per_spectrum.csv")
    if not p.exists():
        pytest.skip("v10 results not present in this checkout")
    d = pd.read_csv(p)
    methods = [c[len("ok_"):] for c in d.columns if c.startswith("ok_")]
    assert methods, "no fitting methods found in the per-spectrum table"
    for m in methods:
        failed = ~d[f"ok_{m}"].to_numpy().astype(bool)
        if not failed.any():
            continue
        np.testing.assert_allclose(
            d.loc[failed, f"logM_{m}"].to_numpy(),
            d.loc[failed, "logM_cond"].to_numpy(),
            rtol=0, atol=1e-12,
            err_msg=f"{m}: failed cases do not carry logM_cond")
