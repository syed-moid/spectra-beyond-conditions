"""Step 4' evaluation: per-model val metrics, gate statistics, paired CIs.

Round-4 decision 4: for every spectral model report improvement over the L1-MLP
metadata reference and over the A1 oracle, each with a paired-bootstrap 95% CI
over evaluation spectra on the seed-averaged prediction, the seed-level gate
pass count, and whether the CI of the improvement excludes 30%.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from a4_eval import load_a4_model  # noqa: E402
from sbc.data.dataset import InsSpectraDataset  # noqa: E402
import dataset_paths as DS

NPZ = DS.FULL
V9 = ROOT / "data" / "full_dataset_phase1_v9" / "dataset.npz"
A2_PRED = ROOT / "results" / "revision" / "A2" / "predictions_seedavg.csv"
A1_CSV = DS.rev("A1") / "per_tuple_conditional_logM.csv"
GATE_FRACTION = 0.30

MODELS = [("5a", "v7", (42, 43, 44, 45, 46)), ("5b", "v7", (42, 43, 44, 45, 46)),
          ("fusion", "v7", (42, 43, 44, 45, 46)), ("cnn", "v7", (42, 43, 44, 45, 46)),
          ("5a", "fixed1", (42, 43, 44)), ("5a", "low", (42, 43, 44)),
          ("fusion", "fixed1", (42, 43, 44))]


def boot_improvement(err_model, err_ref, n_boot, rng):
    """Bootstrap the fractional improvement 1 - MAE_model/MAE_ref over spectra."""
    n = len(err_model)
    idx = rng.integers(0, n, size=(n_boot, n))
    imp = 1.0 - err_model[idx].mean(axis=1) / err_ref[idx].mean(axis=1)
    point = 1.0 - err_model.mean() / err_ref.mean()
    return float(point), float(np.percentile(imp, 2.5)), float(np.percentile(imp, 97.5))


def paired_diff(a, b, n_boot, rng):
    d = a - b
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path,
                    default=DS.rev("A4_extended") / "checkpoints")
    ap.add_argument("--outdir", type=Path,
                    default=DS.rev("A4_extended"))
    ap.add_argument("--stopping-label", type=str, default="extended")
    ap.add_argument("--npz", type=Path, default=NPZ)
    ap.add_argument("--a1-dir", type=Path, default=DS.rev("A1"))
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    ds = InsSpectraDataset(args.npz, "val", severity=1.0, as_torch=False)
    X = np.stack([ds.get_augmented(i, 1.0) for i in range(len(ds))]).astype(np.float32)
    batch = {"spectrum": X, "T_K": ds._T_K.astype(float),
             "c_pct": ds._c_pct.astype(float), "E_kVcm": ds._E_kVcm.astype(float)}
    y = np.log(np.clip(ds._M.astype(float), 1e-9, None))

    a2 = pd.read_csv(A2_PRED)
    ref = a2[(a2.model == "mlp_l1") & (a2.split == "val")]["logM_pred_seedavg"].to_numpy()
    err_ref = np.abs(ref - y)
    ref_mae = float(err_ref.mean())
    a1 = pd.read_csv(ROOT / "results" / "revision" /
                     args.a1_dir / "per_tuple_conditional_logM.csv")
    orc = a1[a1.split == "val"]["logM_cond_median"].to_numpy()
    err_orc = np.abs(orc - y)
    orc_mae = float(err_orc.mean())
    threshold = (1.0 - GATE_FRACTION) * ref_mae
    print(f"metadata reference (mlp_l1) val MAE = {ref_mae:.4f} nats; "
          f"A1 oracle = {orc_mae:.4f}; gate threshold = {threshold:.4f}")

    rng = np.random.default_rng(args.seed)
    rows, pred_rows = [], []
    for arch, protocol, seeds in MODELS:
        preds, per_seed = [], []
        for s in seeds:
            m = load_a4_model(arch, protocol, s, ckpt_dir=args.ckpt_dir)
            p = np.log(np.clip(m.predict_M(batch), 1e-9, None))
            preds.append(p)
            per_seed.append(float(np.abs(p - y).mean()))
        pbar = np.mean(preds, axis=0)
        err = np.abs(pbar - y)
        imp_r, lo_r, hi_r = boot_improvement(err, err_ref, args.n_boot, rng)
        imp_o, lo_o, hi_o = boot_improvement(err, err_orc, args.n_boot, rng)
        d_r, dlo_r, dhi_r = paired_diff(err, err_ref, args.n_boot, rng)
        rows.append({
            "dataset_version": DS.VERSION, "stopping_protocol": args.stopping_label,
            "arch": arch, "train_severity_protocol": protocol, "n_seeds": len(seeds),
            "MAE_logM_mean": float(np.mean(per_seed)),
            "MAE_logM_sd": float(np.std(per_seed, ddof=1)),
            "MAE_logM_ensemble": float(err.mean()),
            "metadata_reference_MAE": ref_mae, "oracle_MAE": orc_mae,
            "improvement_vs_reference": imp_r,
            "improvement_vs_reference_lo": lo_r, "improvement_vs_reference_hi": hi_r,
            "improvement_vs_oracle": imp_o,
            "improvement_vs_oracle_lo": lo_o, "improvement_vs_oracle_hi": hi_o,
            "paired_diff_vs_reference": d_r,
            "paired_diff_vs_reference_lo": dlo_r, "paired_diff_vs_reference_hi": dhi_r,
            "gate_threshold": threshold,
            "seeds_passing_gate": int(sum(v <= threshold for v in per_seed)),
            "improvement_CI_excludes_30pct": bool(lo_r > GATE_FRACTION),
            "beats_reference_CI_excludes_zero": bool(dhi_r < 0.0),
        })
        pred_rows.append(pd.DataFrame({"dataset_version": DS.VERSION, "arch": arch,
                                       "protocol": protocol, "split": "val",
                                       "logM_true": y, "logM_pred_seedavg": pbar}))
        print(f"  {arch:7s}/{protocol:7s} MAE {np.mean(per_seed):.4f} +/- "
              f"{np.std(per_seed, ddof=1):.4f} | improvement vs ref "
              f"{imp_r*100:5.1f}% [{lo_r*100:.1f}, {hi_r*100:.1f}] | "
              f"gate {rows[-1]['seeds_passing_gate']}/{len(seeds)} | "
              f"CI>30%: {rows[-1]['improvement_CI_excludes_30pct']}")

    # pairwise architecture comparisons under the same training protocol
    pairs = []
    ens = {(r["arch"], r["train_severity_protocol"]): None for r in rows}
    allp = pd.concat(pred_rows, ignore_index=True)
    for a1_, p1 in ens:
        for a2_, p2 in ens:
            if (a1_, p1) >= (a2_, p2):
                continue
            e1 = np.abs(allp[(allp.arch == a1_) & (allp.protocol == p1)]
                        ["logM_pred_seedavg"].to_numpy() - y)
            e2 = np.abs(allp[(allp.arch == a2_) & (allp.protocol == p2)]
                        ["logM_pred_seedavg"].to_numpy() - y)
            d, lo, hi = paired_diff(e1, e2, args.n_boot, rng)
            pairs.append({"dataset_version": DS.VERSION, "model_a": f"{a1_}/{p1}",
                          "model_b": f"{a2_}/{p2}", "diff_a_minus_b": d,
                          "lo": lo, "hi": hi, "significant": bool(hi < 0 or lo > 0)})

    pd.DataFrame(rows).to_csv(args.outdir / "gate_and_improvement.csv", index=False)
    pd.DataFrame(pairs).to_csv(args.outdir / "pairwise_model_comparisons.csv", index=False)
    allp.to_csv(args.outdir / "val_predictions_seedavg.csv", index=False)
    (args.outdir / "run_sidecar_gate.json").write_text(json.dumps({
        "task": "A4_gate_eval", "dataset_version": DS.VERSION,
        "stopping_protocol": args.stopping_label, "n_boot": args.n_boot, "seed": args.seed,
        "metadata_reference": "A2 mlp_l1 seed-averaged, val",
        "oracle": "A1 conditional median, val",
        "date_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"\nwrote {args.outdir}")


if __name__ == "__main__":
    main()
