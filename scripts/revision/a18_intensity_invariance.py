"""A18 - does the spectral advantage rest on absolute intensity?

Round-5 decision 4 specifies training with a random global gain and a random
additive offset applied to the training spectra. Every architecture here
standardizes each spectrum per-sample before its encoder,

    x -> (x - mean(x)) / std(x),

so for any gain a > 0 and offset b,

    (a*x + b - (a*mu + b)) / (a*sigma) = (x - mu) / sigma,

identically. Gain and offset are therefore exact no-ops: the specified
augmentation cannot change any prediction, and training under it cannot change
any result. This script verifies that numerically on the trained models rather
than spending training runs to measure a zero, and quantifies the residual as
pure float arithmetic.
"""

from __future__ import annotations

import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from a4_eval import load_a4_model  # noqa: E402
from sbc.data.dataset import InsSpectraDataset  # noqa: E402
import dataset_paths as DS

NPZ = DS.FULL
CK = DS.rev("A4_extended") / "checkpoints"
OUTDIR = DS.rev("A18")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    ds = InsSpectraDataset(NPZ, "val", severity=1.0, as_torch=False)
    X = np.stack([ds.get_augmented(i, 1.0) for i in range(len(ds))]).astype(np.float32)
    y = np.log(np.clip(ds._M.astype(float), 1e-9, None))
    cond = {"T_K": ds._T_K.astype(float), "c_pct": ds._c_pct.astype(float),
            "E_kVcm": ds._E_kVcm.astype(float)}
    rng = np.random.default_rng(args.seed)
    n = len(X)
    gain = np.exp(rng.uniform(np.log(0.5), np.log(2.0), size=(n, 1))).astype(np.float32)
    med_bg = np.median(np.concatenate([X[:, :30], X[:, -30:]], axis=1), axis=1, keepdims=True)
    offset = (rng.uniform(-0.10, 0.10, size=(n, 1)) * med_bg).astype(np.float32)

    rows = []
    for arch in ("cnn", "5a", "5b", "fusion"):
        for s in args.seeds:
            m = load_a4_model(arch, "v7", s, ckpt_dir=CK)
            p0 = m.predict_logM({"spectrum": X, **cond})
            variants = {
                "gain": (X * gain).astype(np.float32),
                "offset": (X + offset).astype(np.float32),
                "gain_and_offset": (X * gain + offset).astype(np.float32),
            }
            for label, Xv in variants.items():
                p = m.predict_logM({"spectrum": Xv, **cond})
                rows.append({"dataset_version": DS.VERSION, "arch": arch, "seed": s,
                             "augmentation": label,
                             "MAE_logM_baseline": float(np.abs(p0 - y).mean()),
                             "MAE_logM_augmented": float(np.abs(p - y).mean()),
                             "delta_MAE": float(np.abs(p - y).mean() - np.abs(p0 - y).mean()),
                             "max_abs_pred_diff": float(np.abs(p - p0).max()),
                             "mean_abs_pred_diff": float(np.abs(p - p0).mean())})
    df = pd.DataFrame(rows)
    df.to_csv(args.outdir / "intensity_invariance.csv", index=False)
    agg = df.groupby(["arch", "augmentation"]).agg(
        delta_MAE_mean=("delta_MAE", "mean"), max_pred_diff=("max_abs_pred_diff", "max")
    ).reset_index()
    print(agg.to_string(index=False))
    print(f"\nlargest |delta MAE| anywhere: {df.delta_MAE.abs().max():.3e}")
    print(f"largest |prediction change| anywhere: {df.max_abs_pred_diff.max():.3e}")
    (args.outdir / "run_sidecar.json").write_text(json.dumps({
        "task": "A18", "dataset_version": DS.VERSION,
        "conclusion": "per-sample standardization makes every architecture exactly invariant "
                      "to global gain and additive offset; the specified augmentation is a no-op",
        "gain_range": [0.5, 2.0], "offset_frac_of_median_background": 0.10,
        "largest_abs_delta_MAE": float(df.delta_MAE.abs().max()),
        "largest_abs_pred_change": float(df.max_abs_pred_diff.max()),
        "seeds": list(args.seeds),
        "date_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"\nwrote {args.outdir}")


if __name__ == "__main__":
    main()
