"""Round 48 item D: the figure package, drawn from the tables the manuscript cites.

The previous module drew v8-era figure definitions. Running it on v10 inputs made
the figures *fresh* without making them *correct*: populations, metrics and methods
did not match what the text claimed. A freshness check cannot catch that, because it
verifies recency and input existence, not that a figure plots what the caption says.

Every figure here is drawn from the same tables the manuscript quotes, and each one
declares, in its CSV header, the population, the sample count, the checkpoint
collection, the aggregation and the failure policy.

Main sources (all v10):
  A23/primary_per_spectrum.csv   per-spectrum fits AND network predictions, 600 test cases
  A23/primary_fit_summary.csv    all-case and successful-fit-only fitting errors
  A23/severity_map_fits.csv      the fits across severity, if the map was restorable
  A22/test_eval_runs.csv         nine runs per architecture on the untouched test set
  A24/ranking_by_severity_v10.csv within-condition pairwise accuracy
  A10/holdout_table_v10.csv      hold-outs with the conditions-only reference
  A7/…                           channel ablation
  A17/variant_summary.csv        architecture sweep
  A9/…                           occlusion
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import dataset_paths as DS

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "results" / "revision"
OUT = ROOT / "figures"
CM = 1 / 2.54
matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 1.0, "font.size": 9,
    "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "axes.titlesize": 10})
C_META = "#6E6E6E"; C_CNN = "#1F4E79"; C_TUNED = "#0B2E4F"; C_TF = "#2E8B8B"
C_FUSE = "#7B5AA6"; C_FIT = "#A93226"; C_FIT2 = "#E08E79"; C_ORACLE = "#B8860B"
C_GRID = "#E8E8E8"; C_NEG = "#B2182B"; C_POS = "#2166AC"

NET_LABEL = {"cnn_kernel45": "tuned CNN", "tf_patch60": "tuned transformer",
             "fusion": "fusion", "cnn": "1D CNN", "5a": "ST-5a", "5b": "ST-5b"}
NET_ORDER = ["cnn_kernel45", "tf_patch60", "fusion", "cnn", "5a", "5b"]
# One colour + one marker + one dash pattern per architecture, used identically in
# Figures 2, 3 and S2 so a reader can carry the key between them, and so the six
# series stay separable in greyscale (round-50 review, item 6).
NET_STYLE = {"cnn_kernel45": (C_TUNED, "o", "-"),
             "tf_patch60": (C_TF, "s", (0, (5, 1.5))),
             "fusion": (C_FUSE, "^", (0, (1, 1))),
             "cnn": (C_CNN, "v", (0, (6, 1.5, 1, 1.5))),
             "5a": ("#4A7BA7", "D", (0, (3, 1.2, 1, 1.2))),
             "5b": ("#8C8C8C", "P", (0, (2, 2)))}
FIT_LABEL = {"dho_matched": "matched-model fit (trust region)",
             "dho_matched_w0": "W = 0 control",
             "dho_matched_prior": "prior-regularized fit",
             "dho_single_win": "windowed single-mode fit (±8 meV)",
             "dho_single": "single-mode fit, full range (±15 meV)"}


def rev(rel):
    head, _, tail = str(rel).partition("/")
    try:
        base = DS.rev(head)
    except KeyError:
        return REV / rel
    return base / tail if tail else base


def src(rel):
    return str(rev(rel).relative_to(ROOT))


def rd(rel):
    p = rev(rel)
    if not p.exists():
        raise FileNotFoundError(f"missing figure input: {p.relative_to(ROOT)}")
    return pd.read_csv(p)


def save(fig, stem, df, inputs, prov):
    """Write the PDF and a CSV whose header records the full provenance."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    hdr = [f"# figure: {stem}", f"# inputs: {'; '.join(inputs)}"]
    for k in ("population", "n", "checkpoints", "aggregation", "failure_policy", "note"):
        if prov.get(k) is not None:
            hdr.append(f"# {k}: {prov[k]}")
    with open(OUT / f"{stem}.csv", "w") as f:
        f.write("\n".join(hdr) + "\n")
        df.to_csv(f, index=False)
    print(f"  wrote {stem}.pdf/.csv", flush=True)


NINE = "mean over 9 runs (3 dataset realizations x 3 training seeds)"
CKPT = "A21_v10/checkpoints (ablation_v10_both_seed{0,1,2})"
REPLACED = "all cases; failed fits replaced by the conditions-only conditional median"


# --------------------------------------------------------------------------- #
# Figure 1 -- parameter recovery, four quantities, three methods, 600 cases
# --------------------------------------------------------------------------- #
def fig1():
    inp = [src("A23/primary_per_spectrum.csv")]
    d = rd("A23/primary_per_spectrum.csv")
    over = d.overdamped.to_numpy().astype(bool)
    methods = [("cnn_kernel45", "tuned CNN", C_TUNED),
               ("dho_matched_prior", "prior-regularized fit", C_FIT),
               ("dho_matched", "matched-model fit", C_FIT2)]
    quants = [("omega0", r"$\omega_0$ (meV)", None),
              ("Gamma", r"$\Gamma$ (meV)", None),
              ("ratio", r"$\Gamma/\omega_0$", None),
              ("logM", r"$\log M$ (nats)", None)]
    fig, axes = plt.subplots(2, 2, figsize=(16 * CM, 15 * CM))
    rows = []
    for ax, (q, lab, _) in zip(axes.ravel(), quants):
        if q == "ratio":
            t = d.Gamma_true / d.omega0_true
        else:
            t = d[f"{q}_true"]
        lo, hi = np.inf, -np.inf
        for m, ml, col in methods:
            v = (d[f"Gamma_{m}"] / d[f"omega0_{m}"]) if q == "ratio" else d[f"{q}_{m}"]
            ax.scatter(t, v, s=5, alpha=0.45, lw=0, color=col, label=ml, zorder=2)
            lo = min(lo, np.nanpercentile(v, 0.5), t.min())
            hi = max(hi, np.nanpercentile(v, 99.5), t.max())
            rows.append(pd.DataFrame({"quantity": q, "method": m, "label": ml,
                                      "idx": d.idx, "true": t, "pred": v,
                                      "overdamped": over}))
        ax.plot([lo, hi], [lo, hi], color="k", lw=0.8, ls="--", zorder=3)
        ax.set_xlabel(f"reference {lab}"); ax.set_ylabel(f"predicted {lab}")
        ax.grid(color=C_GRID, lw=0.5)
        if q in ("Gamma", "ratio"):
            ax.set_xscale("log"); ax.set_yscale("log")
    axes[0, 0].legend(loc="upper left", framealpha=0.9, markerscale=2.5)
    # no inset is drawn; the caption must not promise one
    fig.suptitle(f"{len(d)} spectra of the untouched test set, $s=1$ "
                 f"({over.mean():.1%} overdamped)", fontsize=9)
    fig.tight_layout()
    save(fig, "fig1_parameter_recovery", pd.concat(rows, ignore_index=True), inp,
         {"population": "test_v10, the 600-spectrum subset the fits use", "n": len(d),
          "checkpoints": CKPT, "aggregation": f"networks: {NINE}; fits: single pass",
          "failure_policy": REPLACED,
          "note": f"overdamped fraction on this population: {over.mean():.4f}"})


# --------------------------------------------------------------------------- #
# Figure 2 -- three-way recoverability, one population, one target
# --------------------------------------------------------------------------- #
def fig2():
    inp = [src("A23/primary_per_spectrum.csv"),
           src("A23/network_by_severity_fitting_subset.csv"),
           src("A23/conditions_only_reference.json")]
    d = rd("A23/primary_per_spectrum.csv")
    y = d.logM_true.to_numpy()
    oracle = json.loads(rev("A23/conditions_only_reference.json").read_text())["MAE_logM"]

    sev_path = rev("A23/severity_map_fits.csv")
    vpath = rev("A23/severity_map_verdict.json")
    verdict = json.loads(vpath.read_text())["verdict"] if vpath.exists() else "narrow"
    have_map = sev_path.exists() and verdict == "restore"

    net = rd("A23/network_by_severity_fitting_subset.csv")
    fig, ax = plt.subplots(figsize=(12.5 * CM, 8.5 * CM))
    rows = []
    for m in NET_ORDER:
        g = net[net.model == m].sort_values("severity")
        col, mk, ls = NET_STYLE[m]
        ax.errorbar(g.severity, g.MAE_logM, yerr=g.sd, fmt=mk, ls=ls, ms=4, lw=1.4,
                    capsize=2, color=col, label=NET_LABEL[m])
        rows.append(pd.DataFrame({"family": "learning", "method": m, "label": NET_LABEL[m],
                                  "severity": g.severity, "MAE_logM": g.MAE_logM,
                                  "sd_over_runs": g.sd, "n_runs": g.n_runs}))
    # All three matched procedures sit within 0.04 nats of one another at s = 1, so their
    # markers are drawn at a small horizontal offset to keep them separable. The offset is
    # cosmetic: every one of them is evaluated at s = 1 exactly, and the CSV records 1.0.
    fitcol = {"dho_matched_prior": (C_FIT, "*", 0.89), "dho_matched_w0": (C_FIT2, "X", 1.0),
              "dho_matched": ("#7B241C", "p", 1.12)}
    for m, (col, mk, xpos) in fitcol.items():
        if f"logM_{m}" not in d:
            continue
        e1 = float(np.abs(d[f"logM_{m}"] - y).mean())
        ax.plot([xpos], [e1], mk, ms=9, color=col, mec="black", mew=0.5,
                label=FIT_LABEL[m] + r" ($s=1$)")
        rows.append(pd.DataFrame({"family": "fitting", "method": m, "label": FIT_LABEL[m],
                                  "severity": [1.0], "MAE_logM": [e1],
                                  "sd_over_runs": [np.nan], "n_runs": [1],
                                  "plotted_at_x": [xpos]}))
    ax.axhline(oracle, color=C_ORACLE, lw=1.5, ls="-.")
    ax.text(0.03, 1.03 * oracle, f"conditions only ({oracle:.3f})", fontsize=7,
            color=C_ORACLE, ha="left", va="bottom", transform=ax.get_yaxis_transform())
    rows.append(pd.DataFrame({"family": "reference", "method": "conditions_only",
                              "label": "conditions only", "severity": [np.nan],
                              "MAE_logM": [oracle], "sd_over_runs": [np.nan], "n_runs": [1]}))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.4, 5.0)
    ax.set_xticks([0.5, 1, 2, 4]); ax.set_xticklabels(["0.5", "1", "2", "4"])
    ax.set_yticks([0.1, 0.2, 0.4, 0.7]); ax.set_yticklabels(["0.1", "0.2", "0.4", "0.7"])
    ax.minorticks_off()
    ax.set_xlabel("degradation severity $s$"); ax.set_ylabel(r"MAE $\log M$ (nats)")
    ax.grid(color=C_GRID, lw=0.5, which="major")
    # Legend outside the axes: inside it covered the fitting markers at s = 1 and the
    # conditions-only annotation (round-50 review, item 6).
    ax.legend(ncol=1, frameon=False, fontsize=6.8, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    save(fig, "fig2_recoverability_map", pd.concat(rows, ignore_index=True), inp,
         {"population": "test_v10, the 600-spectrum fitting subset -- EVERY series",
          "n": len(d), "checkpoints": CKPT,
          "aggregation": ("networks: per-run MAE, mean ± SD over 9 runs (no ensemble); "
                          "fits: single pass; reference: MC conditional median, 20000 draws"),
          "failure_policy": REPLACED,
          "note": (f"fits at s = 1 only (severity stop rule, verdict '{verdict}'); "
                   f"authoritative conditions-only reference {oracle:.6f}")})


# --------------------------------------------------------------------------- #
# Figure 3 -- the two boundaries, on pairwise ranking accuracy
# --------------------------------------------------------------------------- #
def fig3():
    inp = [src("A24/ranking_by_severity_v10.csv")]
    b = rd("A24/ranking_by_severity_v10.csv")
    g = b.groupby(["model", "severity"], as_index=False).agg(
        acc=("pairwise_accuracy", "mean"), acc_sd=("pairwise_accuracy", "std"),
        mae=("MAE_logM", "mean"))
    fig, ax = plt.subplots(figsize=(12.5 * CM, 8.5 * CM))
    for m in NET_ORDER:
        d = g[g.model == m].sort_values("severity")
        col, mk, ls = NET_STYLE[m]
        ax.plot(d.severity, d.acc, ls=ls, marker=mk, ms=3.5, lw=1.4, color=col,
                label=NET_LABEL[m])
    ax.axhline(0.5, color="k", lw=1.2, ls=":")
    ax.text(0.255, 0.462, "chance (0.5), exact for any conditions-only predictor",
            fontsize=7, color="k")
    for t in (0.85, 0.75, 0.65):
        ax.axhline(t, color=C_GRID, lw=0.8, zorder=0)
    ax.axvspan(4.0, 13.0, color="#F2F2F2", zorder=0)
    ax.text(5.0, 0.955, "out of training range", fontsize=7, color="#666666")
    ax.set_xscale("log"); ax.set_xticks([0.25, 0.5, 1, 2, 4, 8, 12])
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8", "12"])
    ax.minorticks_off()
    ax.set_xlabel("degradation severity $s$")
    ax.set_ylabel("within-condition pairwise ranking accuracy")
    ax.set_ylim(0.44, 1.0); ax.grid(color=C_GRID, lw=0.5, axis="y")
    # Outside the axes: inside, the legend sat on top of the chance annotation.
    ax.legend(ncol=1, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.0)
    fig.tight_layout()
    save(fig, "fig3_two_boundaries", g, inp,
         {"population": "replicate_test_v10, 250 tuples x 20 realizations", "n": 5000,
          "checkpoints": CKPT, "aggregation": f"per-run means, then {NINE}",
          "failure_policy": "n/a (no fitting in this figure)",
          "note": ("Spearman rho is reported in the supplement, not here; s > 4 shaded; "
                   "each architecture carries its own colour, marker and dash pattern")})


# --------------------------------------------------------------------------- #
# Figure 4 -- hold-outs, as advantage over the conditions-only oracle
# --------------------------------------------------------------------------- #
def fig4():
    inp = [src("A10/holdout_table_v10.csv")]
    h = rd("A10/holdout_table_v10.csv")
    order = ["val", "holdout_c", "holdout_c_extrap", "holdout_E", "holdout_T"]
    lab = {"val": "val\n(in-distribution)", "holdout_c": "holdout_c\n(c = 1%)",
           "holdout_c_extrap": "holdout_c_extrap\n(c = 2.5%)",
           "holdout_E": "holdout_E\n(4 kV/cm)", "holdout_T": "holdout_T\n(600 K)"}
    h = h[h.split.isin(order)].copy()
    h["adv"] = h.MAE_conditions_only - h.MAE_logM          # positive = model better
    M = h.pivot(index="model", columns="split", values="adv").loc[NET_ORDER, order]
    v = np.nanmax(np.abs(M.to_numpy()))
    fig, ax = plt.subplots(figsize=(13 * CM, 8 * CM))
    im = ax.imshow(M.to_numpy(), cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(len(order))); ax.set_xticklabels([lab[s] for s in order], fontsize=7)
    ax.set_yticks(range(len(NET_ORDER)))
    ax.set_yticklabels([NET_LABEL[m] for m in NET_ORDER], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            a = M.to_numpy()[i, j]
            ax.text(j, i, f"{a:+.3f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(a) > 0.6 * v else "black",
                    fontweight="bold" if a < 0 else "normal")
    cb = fig.colorbar(im, ax=ax, fraction=0.035)
    cb.set_label("advantage over the conditions-only oracle (nats)", fontsize=8)
    ax.set_title("positive = the spectrum helps;  negative = worse than conditions alone",
                 fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig4_holdout_heatmap", h[["split", "model", "MAE_logM", "MAE_conditions_only",
                                         "MAE_marginal", "adv"]], inp,
         {"population": "full_dataset_phase1_v10 hold-out splits + holdout_c_extrap_v10",
          "n": "1000 (val), 300 per hold-out", "checkpoints": CKPT, "aggregation": NINE,
          "failure_policy": "n/a",
          "note": "diverging scale centred at zero; negative cells in bold"})


# --------------------------------------------------------------------------- #
# Figure 5 -- channel ablation, absolute errors only
# --------------------------------------------------------------------------- #
def fig5():
    inp = [src("A7/oracle_per_setting.csv"), src("A7/rho_per_setting.csv")]
    o = rd("A7/oracle_per_setting.csv").groupby("setting", as_index=False).oracle_MAE.mean()
    r = rd("A7/rho_per_setting.csv")
    r5 = r[r.arch == "5a"].groupby("setting", as_index=False).agg(
        model_MAE=("MAE_logM", "mean"), rho=("rho_logM", "mean"))
    t = o.merge(r5, on="setting")
    order = ["both", "xi1_only", "alpha_per_mode", "xi2_only"]
    # Short labels broken over two lines: the one-line versions overlapped at
    # publication size (round-50 review, item 6).
    lab = {"both": "both\n" + r"($\xi_1,\xi_2$)",
           "xi1_only": r"$\xi_1$ only" + "\n(linewidth)",
           "alpha_per_mode": r"$\alpha$ per" + "\nmode",
           "xi2_only": r"$\xi_2$ only" + "\n(frequency)"}
    t = t.set_index("setting").loc[order].reset_index()
    x = np.arange(len(t)); w = 0.36
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16 * CM, 7.5 * CM))
    a1.bar(x - w / 2, t.oracle_MAE, w, color=C_ORACLE, label="conditions-only oracle")
    a1.bar(x + w / 2, t.model_MAE, w, color=C_TUNED, label="model (ST-5a)")
    for i, (a, b) in enumerate(zip(t.oracle_MAE, t.model_MAE)):
        a1.text(i - w / 2, a + 0.012, f"{a:.3f}", ha="center", fontsize=7)
        a1.text(i + w / 2, b + 0.012, f"{b:.3f}", ha="center", fontsize=7)
    a1.set_xticks(x); a1.set_xticklabels([lab[s] for s in t.setting], fontsize=7.5, linespacing=1.3)
    a1.set_ylabel(r"MAE $\log M$ (nats)"); a1.grid(color=C_GRID, lw=0.5, axis="y")
    a1.legend(fontsize=7.5); a1.set_title("absolute error", fontsize=9)
    a2.bar(x, t.rho, 0.5, color=C_TF)
    for i, v in enumerate(t.rho):
        a2.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=7)
    a2.set_xticks(x); a2.set_xticklabels([lab[s] for s in t.setting], fontsize=7.5, linespacing=1.3)
    a2.set_ylabel(r"within-condition $\rho(\log M)$"); a2.set_ylim(0, 1.0)
    a2.grid(color=C_GRID, lw=0.5, axis="y"); a2.set_title("ranking", fontsize=9)
    fig.tight_layout()
    save(fig, "fig5_channel_ablation", t, inp,
         {"population": "ablation_v10_{setting}_seed{0,1,2}, val split",
          "n": "1000 per realization", "checkpoints": "A7_v10/checkpoints",
          "aggregation": "mean over 3 realizations x 3 training seeds (ST-5a)",
          "failure_policy": "n/a",
          "note": ("absolute errors only; no percentage-improvement panel; "
                   "category labels broken over two lines")})


# --------------------------------------------------------------------------- #
# Figure 8 -- model families on the untouched test set
# --------------------------------------------------------------------------- #
def fig8():
    """Every series on the 600 fitting cases; networks as per-run mean ± SD."""
    inp = [src("A23/network_runs_on_fitting_subset.csv"),
           src("A23/primary_per_spectrum.csv"),
           src("A23/conditions_only_reference.json")]
    per = rd("A23/network_runs_on_fitting_subset.csv")
    per = per[per.severity == 1.0]
    g = per.groupby("model", as_index=False).agg(mean=("MAE_logM", "mean"),
                                                 sd=("MAE_logM", "std"),
                                                 n=("MAE_logM", "size"))
    d = rd("A23/primary_per_spectrum.csv"); y = d.logM_true.to_numpy()
    oracle = json.loads(rev("A23/conditions_only_reference.json").read_text())["MAE_logM"]
    rows = [{"model": NET_LABEL[m], "family": "learning",
             "MAE_logM": float(g[g.model == m]["mean"].iloc[0]),
             "sd": float(g[g.model == m]["sd"].iloc[0]),
             "n_runs": int(g[g.model == m]["n"].iloc[0])} for m in NET_ORDER]
    for m, lab in FIT_LABEL.items():
        if f"logM_{m}" in d:
            rows.append({"model": lab, "family": "fitting",
                         "MAE_logM": float(np.abs(d[f"logM_{m}"] - y).mean()),
                         "sd": np.nan, "n_runs": 1})
    rows.append({"model": "conditions only", "family": "reference",
                 "MAE_logM": oracle, "sd": np.nan, "n_runs": 1})
    df = pd.DataFrame(rows).sort_values("MAE_logM")
    col = {"learning": C_TUNED, "fitting": C_FIT, "reference": C_ORACLE}
    fig, ax = plt.subplots(figsize=(10.5 * CM, 9 * CM))
    yy = np.arange(len(df))
    ax.barh(yy, df.MAE_logM, xerr=df.sd.fillna(0), height=0.65, capsize=2,
            color=[col[f] for f in df.family])
    ax.set_yticks(yy); ax.set_yticklabels(df.model, fontsize=7.5); ax.invert_yaxis()
    ax.set_xlabel(r"MAE $\log M$ (nats), 600 test cases at $s=1$")
    ax.grid(color=C_GRID, lw=0.5, axis="x")
    ax.set_title("one population, one target; networks mean ± SD over 9 runs", fontsize=8)
    fig.tight_layout()
    save(fig, "fig8_model_family_bars", df, inp,
         {"population": "test_v10, the 600-spectrum fitting subset -- EVERY series",
          "n": 600, "checkpoints": CKPT,
          "aggregation": ("networks: per-run MAE, mean ± SD over 9 runs (no ensemble); "
                          "fits: single pass"),
          "failure_policy": REPLACED,
          "note": f"authoritative conditions-only reference {oracle:.6f}"})


# --------------------------------------------------------------------------- #
# Figure 7 -- fit overlays: two damping regimes x two fitting procedures
# --------------------------------------------------------------------------- #
def fig7():
    """Rendered from the stored optimized parameter vectors, not re-fitted here."""
    sys.path.insert(0, str(ROOT))
    import a3_fitting_baselines as A3
    from sbc.data.dataset import InsSpectraDataset
    fits = [("dho_matched", "matched-model fit", C_FIT2),
            ("dho_matched_prior", "prior-regularized fit", C_FIT)]
    inp = [src("A23/primary_per_spectrum.csv"), str(DS.TEST.relative_to(ROOT))] + \
          [src(f"A23/fit_parameters_{m}.csv") for m, _, _ in fits]
    d = rd("A23/primary_per_spectrum.csv")
    P = {m: rd(f"A23/fit_parameters_{m}.csv").set_index("idx") for m, _, _ in fits}
    ds = InsSpectraDataset(DS.TEST, "test", severity=1.0, as_torch=False)
    grid = ds._omega_grid
    ratio = (d.Gamma_true / d.omega0_true).to_numpy()

    def choose(mask):
        """Median-ratio case for which EVERY displayed fit reported success."""
        cand = [k for k in d.index[mask]
                if all(bool(P[m].loc[int(d.loc[k, "idx"]), "ok"]) for m, _, _ in fits)]
        if not cand:
            raise RuntimeError("no case in this regime has every displayed fit converged")
        r = ratio[cand]
        return int(cand[int(np.argsort(np.abs(r - np.median(r)))[0])])

    pick = [choose(ratio < 0.5), choose(ratio > 1.5)]
    pcols = [c for c in P[fits[0][0]].columns if c[:1] == "p" and c[1:2].isdigit()]
    fig, axes = plt.subplots(1, 2, figsize=(17 * CM, 7.6 * CM))
    rows, stats = [], []
    for ax, k in zip(axes, pick):
        r = d.loc[k]; i = int(r.idx)
        y = ds.get_augmented(i, 1.0).astype(float)
        T, E = float(ds._T_K[i]), float(ds._E_kVcm[i])
        peak = float(np.max(y))
        ax.plot(grid, y, lw=0.9, color="#444444", label="degraded spectrum", zorder=2)
        rows.append(pd.DataFrame({"panel": f"idx{i}", "series": "data",
                                  "omega": grid, "intensity": y}))
        reg = "overdamped" if r.overdamped else "underdamped"
        om_t, gm_t = float(r.omega0_true), float(r.Gamma_true)
        # The point of this figure is parameter error despite close spectral agreement,
        # so every panel carries the true and fitted (omega0, Gamma) and the RMS spectral
        # residual divided by the data peak (round-50 review, item 5).
        txt = [rf"true: $\omega_0$ = {om_t:.2f}, $\Gamma$ = {gm_t:.2f} meV"]
        for m, lab, col in fits:
            assert bool(P[m].loc[i, "ok"]), f"{m} did not converge on case {i}"
            vec = P[m].loc[i, pcols].to_numpy(dtype=float)
            curve = A3.model_matched(vec, grid, T, E)
            ax.plot(grid, curve, lw=1.3, color=col, label=lab, zorder=3)
            rows.append(pd.DataFrame({"panel": f"idx{i}", "series": m,
                                      "omega": grid, "intensity": curve}))
            om_f, gm_f = float(vec[0]), float(vec[1])
            rms = float(np.sqrt(np.mean((curve - y) ** 2)) / peak)
            txt.append(rf"{lab}: $\omega_0$ = {om_f:.2f}, $\Gamma$ = {gm_f:.2f}, "
                       rf"RMS/peak = {rms:.4f}")
            stats.append({"panel": f"idx{i}", "regime": reg, "idx": i, "method": m,
                          "label": lab, "omega0_true": om_t, "Gamma_true": gm_t,
                          "omega0_fit": om_f, "Gamma_fit": gm_f,
                          "abs_err_omega0": abs(om_f - om_t), "abs_err_Gamma": abs(gm_f - gm_t),
                          "rms_residual_over_peak": rms, "T_K": T, "E_kVcm": E,
                          "converged": True})
        ax.text(0.02, 0.98, "\n".join(txt), transform=ax.transAxes, va="top", ha="left",
                fontsize=6.4, linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CCCCCC", lw=0.6,
                          alpha=0.92), zorder=5)
        ax.set_title(rf"{reg}: $\Gamma/\omega_0$ = {gm_t / om_t:.2f}, "
                     rf"$T$ = {T:.0f} K", fontsize=8.5)
        ax.set_xlabel(r"$\omega$ (meV)"); ax.set_xlim(-15, 15); ax.grid(color=C_GRID, lw=0.5)
        ax.set_ylim(top=float(np.max(y)) * 1.65)
    axes[0].set_ylabel("intensity (counts)")
    # One shared legend, placed in the panel where it covers no curve.
    axes[1].legend(fontsize=7, loc="center right", frameon=False)
    fig.tight_layout()
    st = pd.DataFrame(stats)
    st.to_csv(OUT / "fig7_fit_overlay_parameters.csv", index=False)
    print("  wrote fig7_fit_overlay_parameters.csv", flush=True)
    print(st[["regime", "label", "omega0_true", "omega0_fit", "Gamma_true", "Gamma_fit",
              "rms_residual_over_peak"]].to_string(index=False), flush=True)
    save(fig, "fig7_fit_overlay", pd.concat(rows, ignore_index=True), inp,
         {"population": "test_v10, two cases from the 600-spectrum subset", "n": 2,
          "checkpoints": "n/a (fitting only)", "aggregation": "one spectrum per panel",
          "failure_policy": ("both displayed cases converged; the plotting code asserts it. "
                             "These are representative converged fits, NOT failures"),
          "note": ("each curve is the optimizer's fitted spectrum: the stored optimized "
                   "parameter vector pushed through model_matched(), the same forward "
                   "model used during fitting. True and fitted (omega0, Gamma) and the "
                   "RMS spectral residual over the data peak are in "
                   "fig7_fit_overlay_parameters.csv and printed on each panel")})


# --------------------------------------------------------------------------- #
# S1 -- local encoding scale
# --------------------------------------------------------------------------- #
def figS1():
    inp = [src("A17/variant_summary.csv")]
    v = rd("A17/variant_summary.csv")
    STEP = 0.050083
    WIDTH = {"cnn_kernel15": 15, "cnn_kernel21": 21, "cnn_kernel31": 31, "cnn_kernel45": 45,
             "tf_patch5": 5, "tf_patch20": 20, "tf_patch30": 30, "tf_patch40": 40,
             "tf_patch60": 60}
    BASE = {"cnn_base": 7, "tf_base": 10}
    v["bins"] = v.variant.map({**WIDTH, **BASE})
    v["meV"] = v.bins * STEP
    v["kind"] = np.where(v.variant.isin(WIDTH), "width sweep",
                         np.where(v.variant.isin(BASE), "family base", "depth / capacity"))
    fig, ax = plt.subplots(figsize=(11 * CM, 8.5 * CM))
    for fam, col, mark in (("cnn", C_TUNED, "o"), ("tf", C_TF, "s")):
        w = v[(v.family == fam) & (v.kind == "width sweep")].sort_values("meV")
        ax.errorbar(w.meV, w["mean"], yerr=w["std"], fmt=f"-{mark}", ms=4, lw=1.4,
                    capsize=2, color=col, label=f"{fam.upper()} width sweep (fixed capacity)")
        b = v[(v.family == fam) & (v.kind == "family base")]
        ax.errorbar(b.meV, b["mean"], yerr=b["std"], fmt=mark, ms=7, mfc="white",
                    mew=1.5, capsize=2, color=col, label=f"{fam.upper()} base (lower capacity)")
        o = v[(v.family == fam) & (v.kind == "depth / capacity")]
        if len(o):
            ax.errorbar(np.full(len(o), 0.30 if fam == "cnn" else 0.34), o["mean"],
                        yerr=o["std"], fmt="^", ms=5, capsize=2, color=col, alpha=0.65,
                        label=f"{fam.upper()} depth / capacity variants")
    ax.set_xlabel("local encoding scale (meV)")
    ax.set_ylabel(r"validation MAE $\log M$ (nats)")
    ax.grid(color=C_GRID, lw=0.5); ax.legend(fontsize=6.8, ncol=1)
    ax.set_title("mean over 3 training seeds ± SD (not the seed ensemble)", fontsize=8)
    fig.tight_layout()
    save(fig, "figS1_receptive_field", v, inp,
         {"population": "full_dataset_phase1_v10, val split", "n": 1000,
          "checkpoints": "A17_v10/checkpoints",
          "aggregation": "mean over 3 training seeds ± SD (per-run mean, NOT the ensemble)",
          "failure_policy": "n/a",
          "note": ("width-sweep points are connected and hold parameter count at "
                   "1.175-1.192 M; the family base and the depth/capacity variants are "
                   "separate markers because they change capacity as well")})


# --------------------------------------------------------------------------- #
# S3 -- occlusion, with the out-of-training-range label
# --------------------------------------------------------------------------- #
def figS3():
    inp = [src("A9/occlusion_sensitivity.csv")]
    d = rd("A9/occlusion_sensitivity.csv")
    sevs = sorted(d.severity.unique())
    fig, axes = plt.subplots(1, len(sevs), figsize=(4.4 * len(sevs) * CM, 7 * CM),
                             sharey=True)
    axes = np.atleast_1d(axes)
    col = {"cnn_kernel45": C_TUNED, "cnn": C_CNN, "5a": C_TF, "fusion": C_FUSE}
    for ax, s in zip(axes, sevs):
        g = d[d.severity == s]
        for m, sub in g.groupby("model"):
            sub = sub.sort_values("window_centre_meV")
            ax.plot(sub.window_centre_meV, sub.delta_MAE, lw=1.1,
                    color=col.get(m, "#999999"), label=NET_LABEL.get(m, m))
        ttl = f"$s={s:g}$"
        if s > 4.0:
            ttl += "\n(out of training range)"
            ax.set_facecolor("#FAFAFA")
        ax.set_title(ttl, fontsize=8)
        ax.set_xlabel(r"$\omega$ (meV)"); ax.grid(color=C_GRID, lw=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(5))
    axes[0].set_ylabel(r"$\Delta$MAE when occluded (nats)")
    axes[-1].legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, "figS3_occlusion", d, inp,
         {"population": "full_dataset_phase1_v10, stress_base", "n": 500,
          "checkpoints": CKPT, "aggregation": "mean over models per severity",
          "failure_policy": "n/a",
          "note": "s = 12 is explicitly labelled out of training range"})


# --------------------------------------------------------------------------- #
# Figures whose content the round-48 review did not fault are delegated to the
# previous module unchanged, so this file holds only what actually changed.
# --------------------------------------------------------------------------- #
def _delegate(name):
    import generate_figures_v8 as G8
    return getattr(G8, name)


def fig6():
    _delegate("fig6")()


def figS2():
    """One-factor sweeps on their own physical axes, with both background experiments.

    Round-50 review, item 3. The previous version labelled all three panels
    "level (s)" with ticks at 0.25 / 1 / 4, which describe the *combined* severity
    scale and not these sweeps: the source table's `level` column holds resolution
    FWHM in meV, counts at the peak, and background as a fraction of the peak. It
    also dropped `one_factor_background_shaped.csv`, which is the only background
    series that moves -- per-sample standardization removes a constant offset, so
    the constant-background control is flat by construction (tuned CNN 0.084309 at
    every level). Both are drawn, and the flat one is labelled as the control.
    """
    inp = [src("A8/one_factor_sweeps.csv"), src("A8/one_factor_background_shaped.csv")]
    d = rd("A8/one_factor_sweeps.csv")
    b = rd("A8/one_factor_background_shaped.csv")
    PANELS = [("resolution", d, "resolution FWHM (meV)", False,
               "resolution", [0.7, 1.0, 1.4, 2.0, 2.9]),
              ("counts", d, "counts at peak", True,
               "counts", [2500, 5000, 10000, 20000, 40000]),
              ("background", d, "background / peak", True,
               "background, constant (control)", [0.04, 0.08, 0.15, 0.3, 0.6]),
              ("background_shaped", b, "background / peak", True,
               "background, shaped", [0.04, 0.08, 0.15, 0.3, 0.6])]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(18.5 * CM, 6.6 * CM), sharey=True)
    rows = []
    for ax, (fac, tab, xlabel, logx, title, ticks) in zip(axes, PANELS):
        g = tab[tab.factor == fac]
        if g.empty:
            raise RuntimeError(f"figS2: no rows for factor {fac!r}")
        for m in NET_ORDER:
            sub = g[g.model == m].sort_values("level")
            if sub.empty:
                continue
            col, mk, ls = NET_STYLE[m]
            ax.plot(sub.level, sub.MAE_ensemble, ls=ls, marker=mk, ms=3.2, lw=1.2,
                    color=col, label=NET_LABEL[m])
            rows.append(pd.DataFrame({"factor": fac, "model": m, "label": NET_LABEL[m],
                                      "level": sub.level, "level_units": xlabel,
                                      "MAE_logM_ensemble": sub.MAE_ensemble}))
        if logx:
            ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels([(f"{t:g}" if t < 1000 else f"{t/1000:g}k") for t in ticks],
                           fontsize=7)
        ax.minorticks_off()
        ax.set_xlabel(xlabel, fontsize=8); ax.set_title(title, fontsize=8)
        ax.grid(color=C_GRID, lw=0.5)
    # Shared log y: the three left panels move by a few hundredths of a nat while the
    # shaped-background panel reaches 1.15, and a shared linear axis flattens the first
    # three into featureless lines.
    axes[0].set_yscale("log")
    axes[0].set_yticks([0.05, 0.1, 0.2, 0.4, 0.8])
    axes[0].set_yticklabels(["0.05", "0.1", "0.2", "0.4", "0.8"])
    for ax in axes:
        ax.minorticks_off()
    axes[0].set_ylabel(r"MAE $\log M$ (nats, ensemble)")
    axes[2].text(0.5, 0.97, "flat by construction:\nstandardization removes\na constant offset",
                 transform=axes[2].transAxes, ha="center", va="top", fontsize=6.2,
                 color="#666666")
    axes[-1].legend(ncol=1, frameon=False, fontsize=6.5, loc="upper left",
                    bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    save(fig, "figS2_one_factor", pd.concat(rows, ignore_index=True), inp,
         {"population": "test_v10, each degradation channel varied alone", "n": 500,
          "checkpoints": CKPT, "aggregation": "ensemble over the 9 runs",
          "failure_policy": "n/a (no fitting in this figure)",
          "note": ("x axes are the source table's physical `level` column -- resolution "
                   "FWHM in meV, counts at the peak, background as a fraction of the peak "
                   "-- NOT the combined severity s. Panel 3 is the constant-background "
                   "control (flat by construction under per-sample standardization); "
                   "panel 4 is the shaped background, which is the experiment that moves")})


def figS4():
    _delegate("figS4")()


def figS5():
    _delegate("figS5")()


def figS6():
    """The generator-correction control, with BOTH arms pinned to the v8 record.

    Round-51 review, item 2. The delegated version read the defective arm from
    the historical control and the corrected arm through the version resolver,
    which on v10 resolves to `A4_v10_extended` -- a different generator AND a
    different stopping rule. The figure was therefore comparing a v7-semantics
    defective run against a v10 extended-protocol run and describing the result
    as isolating the occupation-factor correction. It did not isolate anything.

    Both arms are now read from literal paths that the resolver cannot redirect:
    the defective runs from the control harness, and the corrected runs from the
    **v8, v7-protocol** arm that was the control's original comparator. Same
    protocol, same harness, same five seeds; the only difference is the
    occupation factor. That is the comparison the text claims.
    """
    # Literal, resolver-proof. DS.rev("A4") would map to A4_v10 under v10.
    pb = REV / "A4_control_buggy_bose" / "control_runs.csv"
    pc = REV / "A4" / "training_runs.csv"
    for q in (pb, pc):
        if not q.exists():
            raise FileNotFoundError(f"missing figS6 input: {q.relative_to(ROOT)}")
    inp = [str(pb.relative_to(ROOT)), str(pc.relative_to(ROOT))]
    b = pd.read_csv(pb)
    c = pd.read_csv(pc)
    c = c[c.protocol == "v7"]                       # the control's own protocol
    assert set(b.dataset_version.unique()) == {"v7_generator_semantics"}, \
        f"defective arm is not the v7-semantics control: {b.dataset_version.unique()}"
    assert set(c.dataset_version.unique()) == {"v8"}, \
        f"corrected arm is not v8: {c.dataset_version.unique()}"

    ARMS = [(b, "defective occupation factor", False),
            (c, "corrected (v8)", True)]
    fig, ax = plt.subplots(figsize=(9.5 * CM, 7.5 * CM))
    rows = []
    for k, (arch, lab, col) in enumerate((("5a", "learned-patch", C_TF),
                                          ("5b", "Fourier-feature", C_META))):
        for j, (tab, arm, filled) in enumerate(ARMS):
            v = tab[tab.arch == arch].best_val_MAE_logM.astype(float)
            ax.errorbar(k + (j - 0.5) * 0.24, v.mean(), yerr=v.std(ddof=1), fmt="o",
                        color=col, mfc=col if filled else "none", ms=6, capsize=3,
                        lw=1.2, label=arm if k == 0 else None)
            rows.append({"arch": arch, "architecture": lab, "arm": arm,
                         "dataset_version": tab[tab.arch == arch].dataset_version.iloc[0],
                         "protocol": "v7", "n_seeds": int(v.size),
                         "mean_val_MAE_logM": float(v.mean()),
                         "sd_val_MAE_logM": float(v.std(ddof=1)),
                         "seed_values": ";".join(f"{x:.6f}" for x in sorted(v))})
    ax.set_xticks([0, 1]); ax.set_xticklabels(["learned-patch", "Fourier-feature"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel(r"validation MAE $\log M$ (nats)")
    ax.set_title("v7 protocol, five seeds per arm; only the occupation factor differs",
                 fontsize=8)
    ax.legend(frameon=False, fontsize=7.5); ax.grid(color=C_GRID, lw=0.5, axis="y")
    fig.tight_layout()
    save(fig, "figS6_bose_control", pd.DataFrame(rows), inp,
         {"population": "v8 validation split (defective arm: same split, v7 generator "
                        "semantics)", "n": 1000,
          "checkpoints": "A4_control_buggy_bose/checkpoints and A4/checkpoints",
          "aggregation": "mean +- SD over 5 training seeds per arm",
          "failure_policy": "n/a (no fitting in this figure)",
          "note": ("BOTH arms are v7-protocol and pre-v9: the defective arm is the "
                   "pre-fix occupation factor, the corrected arm is v8. Paths are "
                   "literal so the version resolver cannot redirect them to v10. "
                   "This figure is the one deliberate pre-v10 item in the package")})


def figS7():
    _delegate("figS7")()


def figS8():
    _delegate("figS8")()


def figS9():
    """Energy-tilt sensitivity. Redrawn so the exported CSV carries EVERY plotted series.

    Round-51 review, item 4: the delegated version plotted the tilt-trained CNN
    but saved only the four sensitivity series, so the released plotting data
    could not reproduce the figure it accompanied.
    """
    inp = [src("A18b/energy_tilt_sensitivity.csv"),
           src("A18b_tilt_trained/tilt_trained_beta_grid.csv")]
    d = rd("A18b/energy_tilt_sensitivity.csv")
    t = rd("A18b_tilt_trained/tilt_trained_beta_grid.csv")
    fig, ax = plt.subplots(figsize=(9.5 * CM, 7 * CM))
    rows = []
    for m in ("cnn", "5a", "fusion", "5b"):
        g = d[d.model == m].sort_values("beta")
        if g.empty:
            continue
        col, mk, ls = NET_STYLE[m]
        ax.plot(g.beta, g.val_MAE_logM, ls=ls, marker=mk, ms=3.2, lw=1.1, color=col,
                label=NET_LABEL[m])
        rows.append(pd.DataFrame({"series": "tilt-evaluated", "model": m,
                                  "label": NET_LABEL[m], "beta": g.beta,
                                  "MAE_logM": g.val_MAE_logM,
                                  "trained_on": "untilted spectra"}))
    ax.plot(t.beta, t.tilt_trained_MAE, "s--", color=C_TUNED, ms=4, lw=1.5,
            label="CNN, tilt-augmented training")
    rows.append(pd.DataFrame({"series": "tilt-trained", "model": "cnn",
                              "label": "CNN, tilt-augmented training", "beta": t.beta,
                              "MAE_logM": t.tilt_trained_MAE,
                              "trained_on": "tilt-augmented spectra"}))
    ax.set_xlabel(r"efficiency tilt $\beta$"); ax.set_ylabel(r"MAE $\log M$ (nats)")
    ax.grid(color=C_GRID, lw=0.5)
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.0)
    fig.tight_layout()
    save(fig, "figS9_tilt_sensitivity", pd.concat(rows, ignore_index=True), inp,
         {"population": "full_dataset_phase1_v10 split stress_base, tilted by "
                        "g(omega) = 1 + beta * (omega / 15 meV)", "n": 500,
          "checkpoints": "A21_v10/checkpoints (evaluated) and A18b_v10_tilt/checkpoints (trained)",
          "aggregation": "mean over runs; tilt-trained arm is the mean over 3 runs",
          "failure_policy": "n/a (no fitting in this figure)",
          "note": ("all five plotted series are exported, including the tilt-trained CNN, "
                   "which an earlier version plotted but did not save")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-skips", action="store_true")
    ap.add_argument("--only", type=str, nargs="+", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    order = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8,
             figS1, figS2, figS3, figS4, figS5, figS6, figS7, figS8, figS9]
    if args.only:
        keep = set(args.only); order = [f for f in order if f.__name__ in keep]
    drawn, skipped = [], []
    for fn in order:
        try:
            fn(); drawn.append(fn.__name__)
        except Exception as e:
            skipped.append((fn.__name__, f"{type(e).__name__}: {e}"))
            print(f"  SKIP {fn.__name__}: {type(e).__name__}: {e}", flush=True)
    print(f"\n  drawn {len(drawn)}, skipped {len(skipped)}", flush=True)
    if skipped:
        print("  skipped figures:", ", ".join(n for n, _ in skipped), flush=True)
        if not args.allow_skips:
            raise SystemExit(f"{len(skipped)} figure(s) were not drawn; the files on disk "
                             f"are stale. Re-run with --allow-skips only if intended.")


if __name__ == "__main__":
    main()
