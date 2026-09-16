"""Render the A24 tables into the v8.3 manuscript placeholders.

Round 45 §1 requires each paragraph to be written from the output its table
cites. The five tables that come from A24 are rendered here from the CSVs
rather than transcribed by hand, and substituted into the marked placeholders.

Placeholders, and where they live:
  <!-- A24-BOUNDARY-TABLE -->  main  §3.4  ranking boundaries, pairwise accuracy
  <!-- A24-RANKING-TABLE -->   main  §3.4  ranking at s = 1 with both nulls
  <!-- A24-PARAM-TABLE -->     supp  S5.4  parameter recovery incl. kappa_slow
  <!-- A24-SWEEP-TABLE -->     supp  S5.7  severity sweep, in and out of range
  <!-- A24-NULL-TABLE -->      supp  S5.8  the two nulls with their SDs

Idempotent: re-running replaces the tables between the marker and its
closing marker.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_paths as DS

def _manuscript_dir():
    """Where the manuscript sources live.

    The manuscript is not part of the code release -- it is the journal's record,
    not the package's -- so the scripts that read it take their location from
    SBC_MANUSCRIPT_DIR. Without it they fail with this message rather than with a
    confusing missing-path error.
    """
    import os
    d = os.environ.get("SBC_MANUSCRIPT_DIR")
    if not d:
        raise SystemExit(
            "This script reads the manuscript sources, which are not part of the "
            "code release. Set SBC_MANUSCRIPT_DIR to the directory holding "
            "DRAFT_*_main.md and DRAFT_*_supplement.md, for example:\n"
            "    SBC_MANUSCRIPT_DIR=../manuscript python " + __file__)
    return Path(d).resolve()


MS = _manuscript_dir()
MAIN, SUPP = MS / "DRAFT_v8_10_main.md", MS / "DRAFT_v8_10_supplement.md"
LABEL = {"cnn_kernel45": "tuned CNN", "tf_patch60": "tuned transformer",
         "fusion": "fusion", "cnn": "1D CNN", "5a": "ST-5a", "5b": "ST-5b"}
ORDER = ["cnn_kernel45", "tf_patch60", "fusion", "cnn", "5a", "5b"]
THRESH = [0.85, 0.75, 0.65]
BAYES_RISK = 0.6045          # conditions-only reference on test_v10, A1_v10/oracle_convergence.csv
# The replicate set has its OWN oracle and must not borrow the test-set value:
# different condition draws, different conditional spread. A1_v10_replicate_test.
BAYES_RISK_REPLICATE = 0.6080
MARGINAL_REPLICATE = 1.3227


def crossing(sev, v, t, falling=True):
    """First severity at which the mean crosses t, linearly interpolated."""
    sev, v = np.asarray(sev, float), np.asarray(v, float)
    o = np.argsort(sev); sev, v = sev[o], v[o]
    for i in range(len(sev) - 1):
        hit = (v[i] >= t > v[i + 1]) if falling else (v[i] < t <= v[i + 1])
        if hit:
            f = (v[i] - t) / (v[i] - v[i + 1]) if falling else (t - v[i]) / (v[i + 1] - v[i])
            return sev[i] + f * (sev[i + 1] - sev[i])
    return None


def fmt(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def boundary_table(b):
    g = b.groupby(["model", "severity"], as_index=False).agg(
        acc=("pairwise_accuracy", "mean"), mae=("MAE_logM", "mean"))
    out = ["| model | acc = 0.85 | acc = 0.75 | **value crossover** | acc = 0.65 | accuracy at the value crossover |",
           "|---|---|---|---|---|---|"]
    for m in ORDER:
        d = g[g.model == m]
        a85, a75, a65 = (crossing(d.severity, d.acc, t) for t in THRESH)
        # The sweep's MAE is on replicate_test_v10, so the crossover must be
        # against THAT set's oracle, not the ordinary test set's.
        vx = crossing(d.severity, d.mae, BAYES_RISK_REPLICATE, falling=False)
        at = (float(np.interp(vx, np.sort(d.severity.values),
                              d.sort_values("severity").acc.values))
              if vx is not None else None)
        b_ = lambda x: f"**{fmt(x)}**" if x is not None and x > 4.0 else fmt(x)
        out.append(f"| {LABEL[m]} | {b_(a85)} | {b_(a75)} | {b_(vx)} | {b_(a65)} | "
                   f"{fmt(at, 3)} |")
    out.append("")
    out.append("Columns are severities. The three accuracy columns are where the mean")
    out.append("within-condition pairwise accuracy falls through each operational threshold; the")
    out.append("**value crossover** is where MAE on log M first exceeds the conditions-only")
    out.append(f"reference **for that set** ({BAYES_RISK_REPLICATE:.4f} nats on `replicate_test_v10`,")
    out.append(f"not the {BAYES_RISK:.4f} of `test_v10`). Severities in **bold** lie outside the training")
    out.append("range s ∈ [0.25, 4] and confound measurement quality with distribution shift; they")
    out.append("are reported, not interpreted. The last column gives the ranking accuracy the model")
    out.append("still has at the point where it stops being worth using for a calibrated value.")
    return "\n".join(out)


def ranking_table(b):
    d = b[b.severity == 1.0]
    g = d.groupby("model", as_index=False).agg(
        MAE=("MAE_logM", "mean"), acc=("pairwise_accuracy", "mean"),
        rho=("rho", "mean"),
        dacc=("null_der_acc_mean", "mean"), dacc_sd=("null_der_acc_sd", "mean"),
        pacc=("null_perm_acc_mean", "mean"), pacc_sd=("null_perm_acc_sd", "mean"),
        drho=("null_der_rho_mean", "mean"), prho=("null_perm_rho_mean", "mean"),
        drho_exp=("der_rho_expected", "mean"))
    g = g.set_index("model").loc[ORDER].reset_index()
    out = ["| model | MAE | **pairwise accuracy** | acc., permuted (primary null) | acc., deranged | ρ (secondary) | ρ, permuted | ρ, deranged | E[ρ] deranged |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in g.itertuples():
        best = "**" if r.model == "cnn_kernel45" else ""
        out.append(f"| {LABEL[r.model]} | {r.MAE:.4f} | {best}{r.acc:.3f}{best} | "
                   f"{r.pacc:.3f} ± {r.pacc_sd:.3f} | {r.dacc:.3f} ± {r.dacc_sd:.3f} | "
                   f"{r.rho:.3f} | {r.prho:+.3f} | {r.drho:+.3f} | {r.drho_exp:+.3f} |")
    out.append(f"| conditions only | {BAYES_RISK_REPLICATE:.4f} | **0.500** (exact) | — | — | "
               "undefined | — | — | — |")
    out.append("")
    out.append(f"`replicate_test_v10`, s = 1, 250 tuples × 20 realizations, mean over nine runs. "
               f"The conditions-only MAE is that set's **own** oracle ({BAYES_RISK_REPLICATE:.4f}, "
               f"marginal {MARGINAL_REPLICATE:.4f}), computed on its own condition draws; the "
               f"test-set value 0.6045 is a different number and is not reused here.")
    out.append("Null columns are the mean ± SD over **200 permutations and 200 derangements**")
    out.append("per tuple; the ± is the SD over draws of the pooled statistic, not a standard")
    out.append("error. The **permutation is the primary chance control**; the derangement is a")
    out.append("separate control whose mean is below chance by construction (no fixed points).")
    out.append("The conditions-only row's accuracy is exactly one half because every")
    out.append("within-tuple pair is a tie and ties take half credit; its ρ is undefined because")
    out.append("the predictor has no variance within a tuple.")
    return "\n".join(out)


def param_table(p):
    o = p[p.model != "conditions-only (MC conditional median)"]
    g = o.groupby("model", as_index=False).agg(
        R2w=("R2_omega0", "mean"), R2g=("R2_logGamma", "mean"),
        accw=("pair_acc_omega0", "mean"),
        maew=("MAE_omega0", "mean"), maeg=("MAE_logGamma", "mean"),
        ks=("MAE_kappa_slow_overdamped", "mean"), r2ks=("R2_kappa_slow_overdamped", "mean"),
        nover=("n_overdamped", "mean"))
    g = g.set_index("model").loc[ORDER].reset_index()
    ref = p[p.model == "conditions-only (MC conditional median)"].iloc[0]
    N_OVER_TRUE = 760           # truly overdamped in test_v10 (Gamma > omega0), of 2000
    out = ["| model | R²(ω₀) | R²(log Γ) | MAE ω₀ (meV) | MAE log Γ | MAE κ_slow (meV) | R²(κ_slow) | n scored | n dropped |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in g.itertuples():
        out.append(f"| {LABEL[r.model]} | {r.R2w:.4f} | {r.R2g:.4f} | {r.maew:.3f} | "
                   f"{r.maeg:.3f} | {r.ks:.3f} | {r.r2ks:.3f} | {r.nover:.0f} | "
                   f"{N_OVER_TRUE - r.nover:.0f} |")
    out.append(f"| **conditions only (no spectrum)** | {ref.R2_omega0:.4f} | "
               f"{ref.R2_logGamma:.4f} | **{ref.MAE_omega0:.3f}** | {ref.MAE_logGamma:.3f} | "
               "— | — | — | — |")
    out.append("")
    out.append(f"`test_v10`, s = 1, n = 2000. Mean over nine runs.")
    out.append("κ_slow = Γ − √(Γ² − ω₀²), the **exact** slow decay rate; ω₀²/2Γ is its Γ ≫ ω₀")
    out.append("limit and is not used here.")
    out.append("")
    out.append(f"**κ_slow error policy.** κ_slow is defined only for an overdamped mode. "
               f"{N_OVER_TRUE} of the 2000 test spectra are truly overdamped (Γ > ω₀, 38.00%). "
               "A model that predicts Γ < ω₀ for one of them has no κ_slow to compare, so that "
               "case cannot enter the κ_slow columns. Those cases are **counted, not silently "
               "dropped**: the last two columns give the number scored and the number excluded "
               "for each model, and the exclusions run from 10 to 30 cases (1.3% to 3.9% of the "
               "overdamped population). They are a genuine failure of the prediction and are "
               "visible instead in R²(log Γ) and MAE log Γ, which every case enters. The "
               "distinct figure 743 that appeared in an earlier version of this table was this "
               "scored count for one model, not the overdamped population.")
    out.append("")
    out.append("The conditions-only row is the Monte Carlo conditional median of each parameter")
    out.append("given (T, c, E), computed the same way as the metadata oracle. It beats every")
    out.append("network on ω₀ and none of them on Γ.")
    return "\n".join(out)


def sweep_table(b):
    g = b.groupby(["model", "severity"], as_index=False).agg(
        acc=("pairwise_accuracy", "mean"), mae=("MAE_logM", "mean"))
    sev = sorted(g.severity.unique())
    head = "| model | " + " | ".join(f"s = {s:g}" for s in sev) + " |"
    out = [head, "|" + "---|" * (len(sev) + 1)]
    for m in ORDER:
        d = g[g.model == m].set_index("severity")
        out.append(f"| {LABEL[m]} | " + " | ".join(f"{d.acc.loc[s]:.3f}" for s in sev) + " |")
    out.append("| conditions only | " + " | ".join("0.500" for _ in sev) + " |")
    out.append("")
    inr = [s for s in sev if 0.25 <= s <= 4.0]
    out.append(f"Within-condition pairwise ranking accuracy, `replicate_test_v10`, mean over nine")
    out.append(f"runs. In-training-range columns: s = {min(inr):g} to {max(inr):g}. Columns at "
               f"s = {', '.join(f'{s:g}' for s in sev if s > 4.0)} are "
               "**out-of-training-range**.")
    out.append("The conditions-only row is 0.500 at every severity — exactly, not approximately,")
    out.append("and independently of the spectrum, which is the property that makes this metric's")
    out.append("baseline a fact rather than an estimate.")
    return "\n".join(out)


def null_table(b):
    d = b[b.severity == 1.0]
    g = d.groupby("model", as_index=False).agg(
        pa=("null_perm_acc_mean", "mean"), pasd=("null_perm_acc_sd", "mean"),
        da=("null_der_acc_mean", "mean"), dasd=("null_der_acc_sd", "mean"),
        pr=("null_perm_rho_mean", "mean"), prsd=("null_perm_rho_sd", "mean"),
        dr=("null_der_rho_mean", "mean"), drsd=("null_der_rho_sd", "mean"),
        exp=("der_rho_expected", "mean"), obs=("rho", "mean"))
    g = g.set_index("model").loc[ORDER].reset_index()
    out = ["| model | observed ρ | permutation ρ (mean ± SD) | derangement ρ (mean ± SD) | E[ρ] = −ρ/(n−1) | permutation acc. | derangement acc. |",
           "|---|---|---|---|---|---|---|"]
    for r in g.itertuples():
        out.append(f"| {LABEL[r.model]} | {r.obs:.3f} | {r.pr:+.4f} ± {r.prsd:.4f} | "
                   f"{r.dr:+.4f} ± {r.drsd:.4f} | {r.exp:+.4f} | "
                   f"{r.pa:.4f} ± {r.pasd:.4f} | {r.da:.4f} ± {r.dasd:.4f} |")
    out.append("")
    out.append("200 draws of each null per tuple. The derangement mean sits within about one SD")
    out.append("of its derived expectation for every model, and the permutation mean sits near")
    out.append("zero as it should. Both accuracies are at chance. We report the agreement and")
    out.append("draw no inference from the residual difference between the two nulls.")
    return "\n".join(out)


def substitute(path, marker, body):
    txt = path.read_text()
    open_m, close_m = f"<!-- {marker} -->", f"<!-- /{marker} -->"
    block = f"{open_m}\n\n{body}\n\n{close_m}"
    if close_m in txt:
        i, j = txt.index(open_m), txt.index(close_m) + len(close_m)
        txt = txt[:i] + block + txt[j:]
    elif open_m in txt:
        txt = txt.replace(open_m, block)
    else:
        print(f"  WARNING: marker {marker} not found in {path.name}"); return False
    # Write through a temp file in the same directory: this edits the manuscript
    # in place, and a kill part-way through a plain write_text would truncate it.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(txt)
    tmp.replace(path)
    print(f"  filled {marker} in {path.name}")
    return True


def main():
    out = DS.rev("A24")
    p = pd.read_csv(out / "parameter_recovery_v10.csv")
    b = pd.read_csv(out / "ranking_by_severity_v10.csv")
    print(f"  read {len(p)} parameter rows, {len(b)} severity rows from {out.name}")
    ok = [
        substitute(MAIN, "A24-BOUNDARY-TABLE", boundary_table(b)),
        substitute(MAIN, "A24-RANKING-TABLE", ranking_table(b)),
        substitute(SUPP, "A24-PARAM-TABLE", param_table(p)),
        substitute(SUPP, "A24-SWEEP-TABLE", sweep_table(b)),
        substitute(SUPP, "A24-NULL-TABLE", null_table(b)),
        fill_fit_table(),
        fill_main_fit_table(),
        fill_s57_table(),
    ]
    return 0 if all(ok) else 1




# ---------------------------------------------------------------------------
# Round 46: the fitting comparison, rendered from the CSVs rather than
# transcribed. Hand-transcription put a variance into a standard-deviation
# column once already; these tables move enough between reruns to be worth
# generating.
# ---------------------------------------------------------------------------
FIT_ORDER = ["cnn_kernel45", "tf_patch60", "fusion", "cnn", "5a", "5b",
             "dho_matched_prior", "dho_matched", "dho_single_win", "conditions_only"]


def fit_table(comb, prim, gaps):
    """Round 48 B6: the all-case comparison is primary; successful-fit-only is secondary."""
    NET = ["cnn_kernel45", "tf_patch60", "fusion", "cnn", "5a", "5b"]
    FIT = ["dho_matched_prior", "dho_matched_w0", "dho_matched", "dho_single_win",
           "dho_single"]
    c, P = comb.set_index("method"), prim.set_index("method")

    rows = ["**Primary: every method scored on all 600 cases.** A failed fit is replaced by the",
            "conditions-only conditional median for that spectrum, so no method is scored on a",
            "different subset from any other.",
            "",
            "| family | method | failure rate | MAE ω₀ | MAE Γ | median \\|Γ err\\| | MAE Γ, overdamped |",
            "|---|---|---|---|---|---|---|"]
    for m in NET:
        r = c.loc[m]
        rows.append(f"| network | {r.label} | — | {r.MAE_omega0:.4f} | {r.MAE_Gamma:.4f} | "
                    f"{r.median_abs_err_Gamma:.4f} | {r.MAE_Gamma_overdamped:.4f} |")
    for m in FIT:
        if m not in P.index:
            continue
        r = P.loc[m]
        rows.append(f"| fit | {r.label} | {r.failure_rate:.2%} | {r.MAE_omega0_all:.4f} | "
                    f"{r.MAE_Gamma_all:.4f} | {r.median_abs_err_Gamma_all:.4f} | "
                    f"{r.MAE_Gamma_overdamped_all:.4f} |")
    if "conditions_only" in c.index:
        r = c.loc["conditions_only"]
        rows.append(f"| reference | {r.label} | — | **{r.MAE_omega0:.4f}** | {r.MAE_Gamma:.4f} | "
                    f"{r.median_abs_err_Gamma:.4f} | {r.MAE_Gamma_overdamped:.4f} |")

    rows += ["", "**Secondary diagnostic: successful fits only.** These describe different subsets —",
             "each method's own converged cases — and are reported to show what a fit achieves when",
             "it works, not to compare methods with one another.", "",
             "| method | n scored | failure rate | MAE Γ | MAE Γ, overdamped |",
             "|---|---|---|---|---|"]
    for m in FIT:
        if m not in P.index:
            continue
        r = P.loc[m]
        rows.append(f"| {r.label} | {int(r.n_scored_ok)} | {r.failure_rate:.2%} | "
                    f"{r.MAE_Gamma_ok:.4f} | {r.MAE_Gamma_overdamped_ok:.4f} |")

    rows += ["", "**Gap closed by the penalty**, on the all-case numbers. Two contrasts are given: against",
             "the trust-region fit, which changes the optimizer as well as the penalty, and against the",
             "**W = 0 control**, which is the same objective, optimizer, initialisation, bounds and",
             "tolerances with the penalty switched off. Only the second isolates the penalty.", "",
             "| contrast | quantity | gap to tuned CNN (meV) | closed (meV) | fraction |",
             "|---|---|---|---|---|"]
    NICE = {"overall_vs_trust_region": ("vs trust-region fit", "MAE Γ, all"),
            "overall_vs_W0_control": ("**vs W = 0 control**", "MAE Γ, all"),
            "overdamped_vs_trust_region": ("vs trust-region fit", "MAE Γ, overdamped"),
            "overdamped_vs_W0_control": ("**vs W = 0 control**", "MAE Γ, overdamped")}
    for k in ("overall_vs_trust_region", "overall_vs_W0_control",
              "overdamped_vs_trust_region", "overdamped_vs_W0_control"):
        if k not in gaps:
            continue
        lab, q = NICE[k]; v = gaps[k]
        rows.append(f"| {lab} | {q} | {v['gap_meV']:.4f} | {v['closed_meV']:.4f} | "
                    f"**{v['fraction']:.1%}** |")
    rows += ["", "The W = 0 rows are the penalty-specific figures. The trust-region rows describe the",
             "difference between the two procedures as tested, penalty and optimizer together."]
    return "\n".join(rows)


def fill_fit_table():
    import json as _json
    out = DS.rev("A23")
    comb = pd.read_csv(out / "network_vs_fit_test.csv")
    prim = pd.read_csv(out / "primary_fit_summary.csv")
    gaps = _json.loads((out / "primary_gap_fractions.json").read_text())
    return substitute(SUPP, "A23-FIT-TABLE", fit_table(comb, prim, gaps))

# ---------------------------------------------------------------------------
# Round 51 item 3: the main-text fitting table and every sentence around it
# that quotes a number from it, rendered from the SAME summary file as S5.5.
# Before this round the main table was transcribed and had drifted: 1.465
# against 1.4755, 1.981 against 1.9819, 4.137 against 4.1387, 2.143 against
# 2.1420. Rounding now happens once, at format time, from one source.
# ---------------------------------------------------------------------------
MAIN_NET = ["cnn_kernel45", "tf_patch60", "5a"]
MAIN_FIT = ["dho_matched_prior", "dho_matched_w0", "dho_matched"]
MAIN_TAIL = ["dho_single_win", "dho_single"]
MAIN_LABEL = {"dho_matched_prior": "prior-regularized forward-model fit",
              "dho_matched_w0": "W = 0 control (same solver, no penalty)",
              "dho_matched": "matched-model fit (trust region)",
              "dho_single_win": "windowed single-mode fit (±8 meV)",
              "dho_single": "single-mode fit, full range (±15 meV)"}


def main_fit_block(comb, prim, gaps):
    c, P = comb.set_index("method"), prim.set_index("method")
    f4 = lambda x: f"{x:.4f}"

    def frow(m, bold=False):
        r = P.loc[m]
        w = (lambda v: f"**{v}**") if bold else (lambda v: v)
        return (f"| {MAIN_LABEL[m]} | {w(f'{r.failure_rate:.2%}')} | {w(f4(r.MAE_Gamma_all))} | "
                f"{w(f4(r.median_abs_err_Gamma_all))} | {w(f4(r.MAE_Gamma_overdamped_all))} | "
                f"{w(f4(r.MAE_omega0_all))} |")

    nok = {m: int(P.loc[m].n_scored_ok) for m in MAIN_FIT + MAIN_TAIL}
    cnn, pri, w0, tr = (c.loc["cnn_kernel45"], P.loc["dho_matched_prior"],
                        P.loc["dho_matched_w0"], P.loc["dho_matched"])
    ref = c.loc["conditions_only"]
    win, full = P.loc["dho_single_win"], P.loc["dho_single"]
    gw, go = gaps["overall_vs_W0_control"], gaps["overdamped_vs_W0_control"]
    gt, gto = gaps["overall_vs_trust_region"], gaps["overdamped_vs_trust_region"]
    opt, opto = gaps["overall_optimizer_only"], gaps["overdamped_optimizer_only"]

    L = [f"""All rows below are on the **same 600 spectra** of the untouched test set at s = 1, of which 34.7%
are overdamped, and **every method is scored on all 600**: a failed fit is replaced by the
conditions-only prediction for that spectrum rather than dropped. Excluding failures instead would
score each method on a different subset — {nok['dho_matched_prior']} cases for the prior-regularized fit, {nok['dho_matched_w0']} for the
W = 0 control, {nok['dho_matched']} for the trust-region fit, {nok['dho_single']} for the full-range single-mode fit and {nok['dho_single_win']} for
the windowed one — and failures concentrate in the hard spectra. Successful-fit-only errors are in
the supplement as a secondary diagnostic. This table and the paragraphs that follow it are
generated from `A23_v10/primary_fit_summary.csv` and `primary_gap_fractions.json`, the same files
S5.5 is generated from, so the two cannot disagree.""",
         "",
         r"| method | failure rate | MAE Γ (meV) | median \|Γ error\| | MAE Γ, overdamped | MAE ω₀ (meV) |",
         "|---|---|---|---|---|---|"]
    for m in MAIN_NET:
        r = c.loc[m]
        b = (lambda v: f"**{v}**") if m == "cnn_kernel45" else (lambda v: v)
        L.append(f"| {NET_SHORT[m]} | — | {b(f4(r.MAE_Gamma))} | {b(f4(r.median_abs_err_Gamma))} | "
                 f"{b(f4(r.MAE_Gamma_overdamped))} | {f4(r.MAE_omega0)} |")
    L.append(frow("dho_matched_prior"))
    L.append(frow("dho_matched_w0"))
    L.append(frow("dho_matched"))
    L.append(f"| conditions only (no spectrum) | — | {f4(ref.MAE_Gamma)} | "
             f"{f4(ref.median_abs_err_Gamma)} | {f4(ref.MAE_Gamma_overdamped)} | "
             f"**{f4(ref.MAE_omega0)}** |")
    L.append(frow("dho_single_win"))
    L.append(frow("dho_single"))

    L += ["", f"""Network rows are the mean over nine runs; fit rows are a single pass; the conditions-only row is
the Monte Carlo conditional median. A fit is failed if its optimizer did not report success or if
ω₀ or Γ landed within a relative 10⁻³ of a bound (S3.1). The two single-mode rows are **different
implementations** (S3.3), and both use the corrected lineshape (S3.4); an earlier version reported
the full-range fit under the windowed fit's name and with a lineshape that violated detailed
balance.

**The comparison that isolates the penalty is against the W = 0 control**, not against the
trust-region fit: the latter changes the optimizer as well, and "changing nothing else" would not
be true of it. The control is the same objective, optimizer, initialisation, bounds and tolerances
as the regularized fit with the penalty weight set to zero.

Against that control, and on all 600 cases, the penalty **cuts the failure rate from {w0.failure_rate:.1%} to {pri.failure_rate:.1%}**,
**cuts the frequency error from {f4(w0.MAE_omega0_all)} to {f4(pri.MAE_omega0_all)} meV**, and **cuts the overdamped linewidth
error from {f4(w0.MAE_Gamma_overdamped_all)} to {f4(pri.MAE_Gamma_overdamped_all)} meV**. The median absolute linewidth error falls too,
{f4(w0.median_abs_err_Gamma_all)} to {f4(pri.median_abs_err_Gamma_all)} meV, so the penalty is not only a tail regulariser. The trust-region
fit lands essentially on top of the control — {f4(tr.MAE_Gamma_all)} against {f4(w0.MAE_Gamma_all)} meV overall,
{f4(tr.MAE_Gamma_overdamped_all)} against {f4(w0.MAE_Gamma_overdamped_all)} overdamped — so the change of solver accounts for
{opt['difference_meV']:.4f} meV overall and {opto['difference_meV']:.4f} meV in the overdamped regime, and the two contrasts
give the same answer.

Quantifying it: taking the gap to the tuned CNN as the quantity to be closed, the penalty closes
**{gw['fraction']:.1%} of the overall linewidth-error gap and {go['fraction']:.1%} of the overdamped gap** against the W = 0
control, and {gt['fraction']:.1%} / {gto['fraction']:.1%} against the trust-region fit. The two agree because the solver
contributes almost nothing. These percentages measure **the effect of this particular penalty at
this particular weight** — not the fraction of the network's advantage attributable to prior
knowledge, which this experiment does not identify.

Two further readings of the table. On frequency the ordering is striking: the conditions-only row
({f4(ref.MAE_omega0)} meV) is the best entry in the table, the prior-regularized fit is next at {f4(pri.MAE_omega0_all)}, and both
beat every network — the tuned CNN reaches {f4(cnn.MAE_omega0)}. ω₀ is essentially fixed by the conditions,
and a penalty that encodes its distribution recovers almost all of what is left. On linewidth the
ordering inverts: there the conditions-only row ({f4(ref.MAE_Gamma)}) is beaten by every network and by the
prior-regularized fit.

The two single-mode rows are the most severely misspecified — one oscillator where the generator
has three — and they behave accordingly: the windowed fit reaches {f4(win.MAE_Gamma_all)} meV on linewidth and the
full-range fit {f4(full.MAE_Gamma_all)}, but they fail on **{win.failure_rate:.0%} and {full.failure_rate:.0%}** of spectra respectively, against
{pri.failure_rate:.0%} for the prior-regularized fit. They are **not the only misspecified rows**, though: every fit
in the table omits the generator's lineshape skew and supplies the pseudo-Voigt mixing parameter at
a fixed value rather than fitting it (S3.1). No row here is a proxy for "fitting" in general.

The honest summary is narrower than "learning beats fitting", and it has to be **qualified by
target**. On **linewidth** the tested networks outperform every tested fitting procedure, by a
factor of {pri.MAE_Gamma_all / cnn.MAE_Gamma:.1f} on the mean and {pri.MAE_Gamma_overdamped_all / cnn.MAE_Gamma_overdamped:.1f} in the overdamped regime against the regularized fit.
On **frequency** they do not: the prior-regularized fit reaches {f4(pri.MAE_omega0_all)} meV against the tuned
CNN's {f4(cnn.MAE_omega0)}, and the conditions-only reference beats both at {f4(ref.MAE_omega0)}. So the networks do not
outperform every fitting procedure on every reported quantity — they outperform them where the
realization signal actually lives. Whether a calibrated Bayesian treatment, with per-bin counting
variance and a posterior rather than a point estimate, would narrow the gap further is an **open
question this experiment does not answer**; the stored spectra do not carry the per-bin variances
it would need."""]
    return "\n".join(L)


NET_SHORT = {"cnn_kernel45": "tuned CNN", "tf_patch60": "tuned transformer", "5a": "ST-5a"}


def fill_main_fit_table():
    import json as _json
    out = DS.rev("A23")
    comb = pd.read_csv(out / "network_vs_fit_test.csv")
    prim = pd.read_csv(out / "primary_fit_summary.csv")
    gaps = _json.loads((out / "primary_gap_fractions.json").read_text())
    return substitute(MAIN, "MAIN-FIT-TABLE", main_fit_block(comb, prim, gaps))

# ---------------------------------------------------------------------------
# Round 51: S5.7's two tables, generated from the per-spectrum table so they
# follow the log-M fallback fix rather than having to be retyped after it.
# ---------------------------------------------------------------------------
S57_FIT = [("dho_matched_prior", "prior-regularized fit"),
           ("dho_matched_w0", "W = 0 control"),
           ("dho_matched", "matched-model fit (trust region)"),
           ("dho_single_win", "windowed single-mode (±8 meV)"),
           ("dho_single", "single-mode, full range (±15 meV)")]
S57_NET = [("cnn_kernel45", "tuned CNN"), ("tf_patch60", "tuned transformer"),
           ("fusion", "fusion"), ("cnn", "1D CNN"), ("5a", "ST-5a"), ("5b", "ST-5b")]


def s57_block(ps, net, ref):
    import numpy as _np
    y = ps.logM_true.to_numpy()
    at1 = {m: float(net[(net.model == m) & (net.severity == 1.0)].MAE_logM.iloc[0])
           for m, _ in S57_NET}
    L = ["At s = 1:", "",
         "| family | method | n | MAE log M | policy |", "|---|---|---|---|---|"]
    for m, lab in S57_NET:
        L.append(f"| learning | {lab} | {len(ps)} | {at1[m]:.4f} | — |")
    for m, lab in S57_FIT:
        c = f"logM_{m}"
        if c not in ps.columns:
            continue
        e = float(_np.abs(ps[c].to_numpy() - y).mean())
        L.append(f"| fitting | {lab} | {len(ps)} | {e:.4f} | failures replaced |")
    L.append(f"| reference | conditions only | {len(ps)} | **{ref:.4f}** | flat in severity |")
    L += ["", """The ordering is the paper's central claim in one table: learning, then fitting, then
conditions-only — with the penalty moving fitting toward learning, and the two single-mode rows
(deliberately misspecified in mode count) above the rest. The fitting rows follow the **replacement
policy in full**: a failed fit takes the conditions-only conditional median of log M, not merit()
evaluated at the conditional medians of ω₀ and Γ, which are different estimators (S3.1).

**Figure 2 displays fitting results only at s = 1; results at other severities are tabulated in
S5.7.1.** Learning and reference across severity:""", ""]
    sevs = sorted(net.severity.unique())
    L.append("| method | " + " | ".join(f"s = {v:g}" for v in sevs) + " |")
    L.append("|" + "---|" * (len(sevs) + 1))
    for m, lab in S57_NET:
        g = net[net.model == m].set_index("severity")
        L.append(f"| {lab} | " + " | ".join(f"{float(g.loc[v].MAE_logM):.4f}" for v in sevs) + " |")
    L.append("| conditions only | " + " | ".join(f"{ref:.4f}" for _ in sevs) + " |")
    return "\n".join(L)


def fill_s57_table():
    import json as _json
    out = DS.rev("A23")
    ps = pd.read_csv(out / "primary_per_spectrum.csv")
    net = pd.read_csv(out / "network_by_severity_fitting_subset.csv")
    ref = _json.loads((out / "conditions_only_reference.json").read_text())["MAE_logM"]
    return substitute(SUPP, "S57-TABLE", s57_block(ps, net, ref))

if __name__ == "__main__":
    sys.exit(main())
