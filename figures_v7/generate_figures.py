"""Publication figures for Paper 1 (Session 8 results).

Generates three vector figures (SVG + PDF, no PNG) into manuscript_phase1/figures/:
  fig1_concept            — Phase 0 vs Phase 1 generator schematic
  fig2_main_result        — in-dist MAE_logM bar plot + 30% gate line
  fig3_severity_boundary  — MAE_logM vs severity, crossover highlighted

Data is read live from the Phase 1 acceptance artifacts so the figures
regenerate cleanly if the run is repeated:
  manuscript_phase1/results/phase1_diagnostic_report.floor.json
  manuscript_phase1/results/phase1_diagnostic_report.st_aggregated.csv

Run:  python manuscript_phase1/figures/generate_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]   # manuscript_phase1/
ART = ROOT / "results"
FLOOR_JSON = ART / "phase1_diagnostic_report.floor.json"
ST_CSV = ART / "phase1_diagnostic_report.st_aggregated.csv"
OUT = ROOT / "figures"

CM = 1.0 / 2.54  # cm -> inch

# ---- Global styling -------------------------------------------------------
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",     # keep text editable in SVG
    "pdf.fonttype": 42,         # TrueType in PDF
    "axes.linewidth": 1.0,      # spines
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.titlesize": 11,
})

C_NL = "#6E6E6E"      # NL-cond MLP (gray)
C_5A = "#1F4E79"      # ST-5a learned patches (deep blue)
C_5B = "#2E8B8B"      # ST-5b Fourier features (teal)
C_RED = "#A93226"     # gate / threshold / crossover (muted red)
C_GRID = "#E8E8E8"
C_BOX_GRAY = "#ECECEC"
C_BOX_WHITE = "#FFFFFF"


def _save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for fmt in ("svg", "pdf"):
        fig.savefig(OUT / f"{stem}.{fmt}", format=fmt, bbox_inches="tight")
    plt.close(fig)


# ---- Data -----------------------------------------------------------------

def load_data():
    floor = json.loads(FLOOR_JSON.read_text())
    df = pd.read_csv(ST_CSV)

    def get(arch, regime, sev):
        r = df[(df.arch == arch) & (df.regime == regime) & (df.severity == sev)]
        return float(r["mean"].iloc[0]), float(r["std"].iloc[0])

    data = {
        "floor_in_dist": float(floor["in_dist"]),
        "floor_stress": float(floor["stress_1.0"]),  # severity-independent
        "in_dist": {"5a": get("5a", "in_dist", 1.0), "5b": get("5b", "in_dist", 1.0)},
        "severities": [0.25, 0.5, 1.0, 2.0, 4.0],
        "stress": {
            "5a": [get("5a", "stress", s) for s in (0.25, 0.5, 1.0, 2.0, 4.0)],
            "5b": [get("5b", "stress", s) for s in (0.25, 0.5, 1.0, 2.0, 4.0)],
        },
    }
    return data


def print_values(d):
    print("=" * 60)
    print("Figure data (read from artifacts):")
    print("=" * 60)
    print(f"  NL-cond floor (in_dist) : {d['floor_in_dist']:.6f}")
    print(f"  NL-cond floor (stress)  : {d['floor_stress']:.6f}")
    gate = 0.70 * d["floor_in_dist"]
    print(f"  30% gate threshold      : {gate:.6f}  (0.70 x in_dist floor)")
    for arch in ("5a", "5b"):
        m, s = d["in_dist"][arch]
        gap = (d["floor_in_dist"] - m) / d["floor_in_dist"] * 100
        print(f"  ST-{arch} in_dist          : {m:.6f} ± {s:.6f}   gap {gap:+.1f}% vs floor")
    print("  Severity sweep (mean ± std):")
    for i, sev in enumerate(d["severities"]):
        m5a, s5a = d["stress"]["5a"][i]
        m5b, s5b = d["stress"]["5b"][i]
        print(f"    sev={sev:<4}  5a={m5a:.4f}±{s5a:.4f}  5b={m5b:.4f}±{s5b:.4f}  floor={d['floor_stress']:.4f}")
    print("=" * 60)


# ---- Figure 1: concept schematic -----------------------------------------

def _box(ax, cx, cy, w, h, text, fc, dashed=False, fontsize=8.5):
    ls = (0, (4, 2)) if dashed else "solid"
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.02",
        facecolor=fc, edgecolor="black", linewidth=1.0, linestyle=ls, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, zorder=3)


def _arrow(ax, p0, p1, label=None, loff=(0.015, 0.0), fontsize=8, ha="left"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=11, lw=1.2,
        color="black", shrinkA=3, shrinkB=3, zorder=1))
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx + loff[0], my + loff[1], label, fontsize=fontsize, ha=ha, va="center", zorder=3)


def _peak(ax, x0, x1, y0, y1, center, width, color):
    xs = np.linspace(-1, 1, 240)
    L = 1.0 / (1.0 + ((xs - center) / width) ** 2)
    L /= L.max()
    px = x0 + (xs + 1) / 2 * (x1 - x0)
    py = y0 + L * (y1 - y0)
    ax.plot(px, py, color=color, lw=1.3, zorder=3)


def fig1_concept(d):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(17.1 * CM, 8 * CM))
    for ax in (axL, axR):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---- Left: Phase 0 (deterministic) ----
    axL.set_title("Phase 0: deterministic generator", pad=6)
    _box(axL, 0.5, 0.90, 0.34, 0.085, r"$(T,\ c,\ E)$", C_BOX_GRAY)
    _arrow(axL, (0.5, 0.857), (0.5, 0.70), "deterministic", loff=(0.02, 0))
    _box(axL, 0.5, 0.645, 0.34, 0.085, r"$(\omega_0,\ \Gamma)$", C_BOX_WHITE)
    # split to M and to spectrum
    _arrow(axL, (0.40, 0.60), (0.28, 0.45), None)
    _arrow(axL, (0.60, 0.60), (0.72, 0.45), None)
    axL.text(0.255, 0.525, r"$M=f(\omega_0,\Gamma)$", fontsize=8, ha="right", va="center")
    axL.text(0.745, 0.525, r"spectrum $I(\omega)$", fontsize=8, ha="left", va="center")
    _box(axL, 0.27, 0.40, 0.15, 0.08, r"$M$", C_BOX_GRAY)
    _peak(axL, 0.60, 0.92, 0.22, 0.40, center=0.0, width=0.13, color=C_5A)
    axL.text(0.5, 0.075,
             "M is deterministic in (T, c, E).\nSpectrum is a noisy proxy for the same information.",
             fontsize=8, ha="center", va="center", style="italic")

    # ---- Right: Phase 1 (latent microphysics) ----
    axR.set_title("Phase 1: latent-microphysics generator", pad=6)
    _box(axR, 0.5, 0.93, 0.30, 0.072, r"$(T,\ c,\ E)$", C_BOX_GRAY)
    _arrow(axR, (0.5, 0.894), (0.5, 0.785), "baseline", loff=(0.02, 0))
    _box(axR, 0.5, 0.74, 0.32, 0.072, r"$(\omega_{0,\mathrm{base}},\ \Gamma_0)$", C_BOX_WHITE)
    # latent inputs (dashed) on a separate row, feeding the realized box
    _box(axR, 0.125, 0.595, 0.20, 0.095, "$\\xi_1$\n(linewidth)", C_BOX_WHITE, dashed=True, fontsize=8)
    _box(axR, 0.875, 0.595, 0.20, 0.095, "$\\xi_2$\n(frequency)", C_BOX_WHITE, dashed=True, fontsize=8)
    # baseline -> realized (central), with latents feeding in from the sides
    _arrow(axR, (0.5, 0.703), (0.5, 0.515), None)
    _arrow(axR, (0.225, 0.575), (0.36, 0.50), None)
    _arrow(axR, (0.775, 0.575), (0.64, 0.50), None)
    _box(axR, 0.5, 0.465, 0.36, 0.078, r"$(\omega_{0,\mathrm{real}},\ \Gamma_{\mathrm{real}})$", C_BOX_WHITE)
    # split to M and spectrum
    _arrow(axR, (0.40, 0.425), (0.27, 0.285), None)
    _arrow(axR, (0.60, 0.425), (0.73, 0.285), None)
    axR.text(0.255, 0.355, r"$M=\mathrm{merit}(\omega_{0,\mathrm{real}},\Gamma_{\mathrm{real}})$",
             fontsize=7.0, ha="right", va="center")
    axR.text(0.745, 0.355, r"spectrum $I(\omega)$", fontsize=8, ha="left", va="center")
    _box(axR, 0.24, 0.235, 0.15, 0.075, r"$M$", C_BOX_GRAY)
    _peak(axR, 0.62, 0.94, 0.085, 0.255, center=0.28, width=0.30, color=C_5B)
    axR.text(0.5, 0.03,
             r"M depends on realized $(\omega_0,\Gamma)$, which depend on latent $\xi_1,\xi_2$." "\n"
             "Conditions provide only the baseline; realized values vary at fixed conditions.",
             fontsize=7.5, ha="center", va="center", style="italic")

    _save(fig, "fig1_concept")


# ---- Figure 2: main result bar plot --------------------------------------

def fig2_main_result(d):
    fig, ax = plt.subplots(figsize=(8.3 * CM, 8 * CM))
    gate = 0.70 * d["floor_in_dist"]
    m5a, s5a = d["in_dist"]["5a"]
    m5b, s5b = d["in_dist"]["5b"]

    # order: NL-cond, ST-5b, ST-5a
    labels = ["NL-cond\nMLP", "ST-5b\n(Fourier)", "ST-5a\n(learned)"]
    xs = [0, 1, 2]
    means = [d["floor_in_dist"], m5b, m5a]
    errs = [None, s5b, s5a]
    colors = [C_NL, C_5B, C_5A]

    for x, m, e, col in zip(xs, means, errs, colors):
        ax.bar(x, m, width=0.6, color=col, edgecolor="black", linewidth=0.5, zorder=3)
        if e is not None:
            ax.errorbar(x, m, yerr=e, fmt="none", ecolor="black", capsize=4, capthick=1.0,
                        elinewidth=1.0, zorder=4)

    # numeric value labels above each bar
    for x, m, e in zip(xs, means, errs):
        top = m + (e if e else 0)
        ax.text(x, top + 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=9, zorder=5)

    # gate threshold line
    ax.axhline(gate, color=C_RED, lw=1.2, ls=(0, (5, 3)), zorder=2)
    ax.text(2.62, gate, "30% gate\nthreshold", color=C_RED, fontsize=9, ha="left", va="center")

    # gap annotations (bar top -> floor) for spectral bars, offset left of bar
    for x, m, gap in [(1, m5b, (d["floor_in_dist"] - m5b) / d["floor_in_dist"] * 100),
                      (2, m5a, (d["floor_in_dist"] - m5a) / d["floor_in_dist"] * 100)]:
        xo = x - 0.34
        ax.annotate("", xy=(xo, d["floor_in_dist"]), xytext=(xo, m),
                    arrowprops=dict(arrowstyle="<->", color=C_RED, lw=1.0), zorder=4)
        ax.text(xo - 0.04, (m + d["floor_in_dist"]) / 2, f"{gap:+.1f}%",
                color=C_RED, fontsize=9, ha="right", va="center", rotation=90)

    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel(r"MAE$_{\log M}$")
    ax.set_ylim(0, 0.75)
    ax.set_xlim(-0.6, 3.05)
    ax.set_yticks(np.arange(0, 0.76, 0.1))
    ax.yaxis.grid(True, color=C_GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    _save(fig, "fig2_main_result")


# ---- Figure 3: severity boundary -----------------------------------------

def fig3_severity_boundary(d):
    fig, ax = plt.subplots(figsize=(17.1 * CM, 8 * CM))
    sev = np.array(d["severities"])
    floor = d["floor_stress"]
    m5a = np.array([v[0] for v in d["stress"]["5a"]]); s5a = np.array([v[1] for v in d["stress"]["5a"]])
    m5b = np.array([v[0] for v in d["stress"]["5b"]]); s5b = np.array([v[1] for v in d["stress"]["5b"]])

    # crossover shading 2.0 -> 4.0
    ax.axvspan(2.0, 4.0, color=C_RED, alpha=0.15, zorder=0, lw=0)
    ax.text(np.sqrt(2.0 * 4.0), 1.04, "crossover region", color=C_RED, fontsize=9,
            ha="center", va="top")

    # NL-cond floor (horizontal dashed)
    ax.axhline(floor, color=C_NL, lw=1.5, ls=(0, (6, 3)), zorder=2, label="NL-cond MLP (floor)")

    # ST lines
    ax.errorbar(sev, m5a, yerr=s5a, color=C_5A, lw=1.5, marker="o", ms=5, capsize=3,
                capthick=1.0, elinewidth=1.0, zorder=4, label="ST-5a (learned patches)")
    ax.errorbar(sev, m5b, yerr=s5b, color=C_5B, lw=1.5, marker="s", ms=5, capsize=3,
                capthick=1.0, elinewidth=1.0, zorder=3, label="ST-5b (Fourier features)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(sev)
    ax.set_xticklabels([f"{s:g}" for s in sev])
    ax.set_xlabel(r"Augmentation severity ($\times$ nominal)")
    ax.set_ylabel(r"MAE$_{\log M}$")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0.22, 4.5)
    ax.set_yticks(np.arange(0, 1.11, 0.2))

    ax.yaxis.grid(True, color=C_GRID, lw=0.8, zorder=0)
    ax.xaxis.grid(True, color=C_GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    leg = ax.legend(loc="upper left", frameon=True, framealpha=1.0)
    leg.get_frame().set_edgecolor(C_GRID)
    leg.get_frame().set_facecolor("white")

    # crossover annotation: ST lines cross the floor between sev 2 and 4
    ax.annotate("ST crosses floor", xy=(2.83, floor), xytext=(1.15, 0.86),
                color=C_RED, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.0))

    _save(fig, "fig3_severity_boundary")


def main() -> int:
    d = load_data()
    print_values(d)
    fig1_concept(d)
    fig2_main_result(d)
    fig3_severity_boundary(d)
    print(f"\nWrote SVG+PDF for fig1_concept, fig2_main_result, fig3_severity_boundary -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
