"""Step F — Phase 1 acceptance gate + report (Session 8).

1. Train the NL-cond MLP on Phase 1 data -> live floor (per-severity + in_dist).
2. Read the Step E ST ablation per-seed metrics (5a, 5b) -> 3-seed mean +/- std.
3. Compute the live acceptance gate: spectrum-only ST in_dist MAE_logM must be
   >= 30% below the Phase 1 NL-cond floor (i.e., ST_mean <= 0.70 * floor).
4. Write artifacts/phase1_diagnostic_report.md with the full reporting spec,
   including a per-severity breakdown when the gate is a near-miss (10-20%).

Usage (run from manuscript_phase1/):
    python scripts/phase1_acceptance_report.py \
        --st-run-dir results/phase1_st_run \
        --data-path data/full_dataset_phase1/dataset.npz

All three arguments have sensible defaults pointing at the bundled
results/ and data/ directories, so a bare invocation reproduces the gate:
    python scripts/phase1_acceptance_report.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import InsSpectraDataset  # noqa: E402
from src.models.nonlinear_conditions_mlp import NonlinearConditionsMLPModel  # noqa: E402

PHASE0_FLOOR_REF = 0.0048           # historical reference (deterministic-M data)
GATE_FRACTION = 0.30                # ST must be >= 30% below the live floor
SEVERITIES = (0.25, 0.5, 1.0, 2.0, 4.0)
ARCHS = ("5a", "5b")
SEEDS = (42, 43, 44)


# --------------------------------------------------------------------- #
# NL-cond MLP floor on Phase 1                                           #
# --------------------------------------------------------------------- #

def _batch_from_split(data_path: Path, split: str, severity: float) -> dict:
    ds = InsSpectraDataset(data_path, split, severity=severity, as_torch=False)
    n = len(ds)
    T = np.empty(n, np.float32); c = np.empty(n, np.float32); E = np.empty(n, np.float32)
    M = np.empty(n, np.float32)
    for i in range(n):
        it = ds[i]
        T[i] = it["conditions"]["T_K"]; c[i] = it["conditions"]["c_pct"]; E[i] = it["conditions"]["E_kVcm"]
        M[i] = float(it["targets"]["M"])
    return {"T_K": T, "c_pct": c, "E_kVcm": E, "M": M}


def _mae_logM(model, batch) -> float:
    pred = np.asarray(model.predict_M(batch), np.float64)
    true = np.asarray(batch["M"], np.float64)
    return float(np.mean(np.abs(np.log(np.clip(pred, 1e-9, None)) - np.log(np.clip(true, 1e-9, None)))))


def compute_nlcond_floor(data_path: Path) -> dict:
    print(">>> training NL-cond MLP on Phase 1 ...")
    train_ds = InsSpectraDataset(data_path, "train", severity=1.0, as_torch=False)
    val_ds = InsSpectraDataset(data_path, "val", severity=1.0, as_torch=False)
    model = NonlinearConditionsMLPModel(lr=1e-3, n_epochs=1500, batch_size=256, weight_decay=1e-4, seed=42)
    model.fit(train_ds, val_ds)
    floor = {"in_dist": _mae_logM(model, _batch_from_split(data_path, "val", 1.0))}
    for sev in SEVERITIES:
        floor[f"stress_{sev}"] = _mae_logM(model, _batch_from_split(data_path, "stress_base", sev))
    print(f"    NL-cond floor in_dist = {floor['in_dist']:.4f}")
    return floor


# --------------------------------------------------------------------- #
# ST aggregation from Step E per-seed metrics                           #
# --------------------------------------------------------------------- #

def load_st_metrics(st_run_dir: Path) -> pd.DataFrame:
    frames = []
    for arch in ARCHS:
        for seed in SEEDS:
            csv = st_run_dir / f"{arch}_seed{seed}" / "metrics.csv"
            if not csv.exists():
                print(f"    WARNING: missing {csv}")
                continue
            df = pd.read_csv(csv)
            df["arch"] = arch
            df["seed"] = seed
            frames.append(df)
    if not frames:
        raise SystemExit(f"No ST per-seed metrics found under {st_run_dir}")
    return pd.concat(frames, ignore_index=True)


def aggregate_st(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["arch", "regime", "severity"])["MAE_logM"].agg(["mean", "std", "count"]).reset_index()
    return g


def _st_in_dist(agg: pd.DataFrame, arch: str) -> tuple[float, float]:
    row = agg[(agg["arch"] == arch) & (agg["regime"] == "in_dist") & (agg["severity"] == 1.0)]
    if row.empty:
        return float("nan"), float("nan")
    return float(row["mean"].iloc[0]), float(row["std"].iloc[0])


def _st_stress(agg: pd.DataFrame, arch: str, sev: float) -> float:
    row = agg[(agg["arch"] == arch) & (agg["regime"] == "stress") & (agg["severity"] == sev)]
    return float(row["mean"].iloc[0]) if not row.empty else float("nan")


# --------------------------------------------------------------------- #
# Report                                                                 #
# --------------------------------------------------------------------- #

def build_report(floor: dict, agg: pd.DataFrame, st_run_dir: Path, data_path: Path) -> str:
    floor_in = floor["in_dist"]
    threshold = (1.0 - GATE_FRACTION) * floor_in
    lines = []
    lines.append("# Phase 1 acceptance report (Step F)\n")
    lines.append(f"- Dataset: `{data_path}`")
    lines.append(f"- ST ablation run dir: `{st_run_dir}`")
    lines.append(f"- Phase 0 NL-cond floor (historical reference): **{PHASE0_FLOOR_REF:.4f}**")
    lines.append(f"- Phase 1 NL-cond floor (live, new gate basis): **{floor_in:.4f}**")
    lines.append(f"- Live acceptance threshold (0.70 x Phase 1 floor): **{threshold:.4f}**")
    lines.append(f"- Gate: spectrum-only ST in_dist MAE_logM <= threshold (>= {int(GATE_FRACTION*100)}% below floor)\n")

    lines.append("## In-distribution gate (val, severity=1)\n")
    lines.append("| Arch | ST MAE_logM (mean ± std, 3 seeds) | gap vs floor | pass |")
    lines.append("|------|-----------------------------------|--------------|------|")
    gate_status = {}
    for arch in ARCHS:
        mean, std = _st_in_dist(agg, arch)
        gap = (floor_in - mean) / floor_in if floor_in > 0 else float("nan")
        passed = mean <= threshold
        gate_status[arch] = {"mean": mean, "std": std, "gap": gap, "passed": passed}
        lines.append(f"| {arch} | {mean:.4f} ± {std:.4f} | {gap*100:+.1f}% | {'PASS' if passed else 'FAIL'} |")
    lines.append("")

    # Per-severity breakdown (ST mean per arch vs floor).
    lines.append("## Per-severity breakdown (ST mean MAE_logM vs NL-cond floor)\n")
    header = "| Severity | NL-cond floor | " + " | ".join(f"ST-{a}" for a in ARCHS) + " |"
    sep = "|" + "---|" * (2 + len(ARCHS))
    lines.append(header); lines.append(sep)
    for sev in SEVERITIES:
        fl = floor[f"stress_{sev}"]
        cells = " | ".join(f"{_st_stress(agg, a, sev):.4f}" for a in ARCHS)
        lines.append(f"| {sev} | {fl:.4f} | {cells} |")
    lines.append("")

    # Verdict + near-miss diagnostics.
    any_pass = any(gate_status[a]["passed"] for a in ARCHS)
    best_arch = min(ARCHS, key=lambda a: gate_status[a]["mean"])
    lines.append("## Verdict\n")
    if any_pass:
        lines.append(f"**GATE PASSED** — spectrum-only ST beats the Phase 1 NL-cond floor by "
                     f">= {int(GATE_FRACTION*100)}% (best: {best_arch}, "
                     f"gap {gate_status[best_arch]['gap']*100:+.1f}%). "
                     f"Option 3 (latent-microphysics) is validated; Phase 2 may proceed.")
    else:
        best_gap = gate_status[best_arch]["gap"]
        shortfall = GATE_FRACTION - best_gap   # how far below the 30% target
        lines.append(f"**GATE FAILED** — best arch {best_arch} gap {best_gap*100:+.1f}% "
                     f"(target >= {int(GATE_FRACTION*100)}%; shortfall {shortfall*100:.1f} pts).")
        if 0.10 <= shortfall <= 0.20:
            # Near-miss: classify high-severity-concentrated vs uniform.
            floor_vec = np.array([floor[f"stress_{s}"] for s in SEVERITIES])
            st_vec = np.array([_st_stress(agg, best_arch, s) for s in SEVERITIES])
            gap_vec = (floor_vec - st_vec) / np.clip(floor_vec, 1e-9, None)
            low_gap = float(np.mean(gap_vec[:2]))   # sev 0.25, 0.5
            high_gap = float(np.mean(gap_vec[-2:]))  # sev 2, 4
            lines.append(f"\nNear-miss (10-20%). Per-severity gap: low-sev "
                         f"{low_gap*100:+.1f}%, high-sev {high_gap*100:+.1f}%.")
            if high_gap < low_gap - 0.10:
                lines.append("Pattern: **concentrated at high severity** -> recommend the parameter "
                             "card's `phase_1_5_widening_option` for xi2 (range [-0.08, 0.10]) and re-run.")
            else:
                lines.append("Pattern: **uniform shortfall** -> ESCALATE to user (not a severity-localized "
                             "fix; likely a deeper representation or calibration question).")
        else:
            lines.append("\nShortfall outside the 10-20% near-miss band -> ESCALATE to user.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--st-run-dir", type=str, default="results/phase1_st_run")
    ap.add_argument("--data-path", type=str, default="data/full_dataset_phase1/dataset.npz")
    ap.add_argument("--out", type=str, default="results/phase1_diagnostic_report.md")
    args = ap.parse_args()

    data_path = ROOT / args.data_path if not Path(args.data_path).is_absolute() else Path(args.data_path)
    st_run_dir = ROOT / args.st_run_dir if not Path(args.st_run_dir).is_absolute() else Path(args.st_run_dir)

    floor = compute_nlcond_floor(data_path)
    st_df = load_st_metrics(st_run_dir)
    agg = aggregate_st(st_df)

    report = build_report(floor, agg, st_run_dir, data_path)
    print("\n" + report)

    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    # Persist machine-readable artifacts alongside.
    agg.to_csv(out_path.with_suffix(".st_aggregated.csv"), index=False)
    (out_path.with_suffix(".floor.json")).write_text(json.dumps(floor, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
