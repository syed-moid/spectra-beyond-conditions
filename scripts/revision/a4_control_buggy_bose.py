"""Control: this harness, but with the PRE-FIX (buggy) Bose factor.

Why this is the only valid control. `InsSpectraDataset.get_augmented`
regenerates every spectrum from (T, c, E, latent) using whatever generator code
is imported at run time; the stored `spectra_clean` array is returned only at
severity <= 0 and is never seen by a training run. Loading
`full_dataset_phase1/dataset.npz` (v7) therefore produces *v8 spectra* under the
corrected generator - the two dataset files differ only in an array nothing
consumes.

So "v7 dataset vs v8 dataset" is not the comparison that matters. The comparison
is "buggy bose() vs corrected bose()", and it is made by monkeypatching the
factor back to its pre-fix form here. `dho()` resolves `bose` as a module
global, and `augmentations.py` imports the name but never calls it, so patching
the one attribute is sufficient and is verified below before any training runs.

The tracked source file is NOT modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sbc.data.spectrum_generator as G  # noqa: E402

OUTDIR = ROOT / "results" / "revision" / "A4_control_buggy_bose"


def install_buggy_bose():
    """Restore the pre-fix expression and recalibrate F^2 exactly as import did."""
    def bose_buggy(omega, T):
        return 1.0 / (np.exp(G.HBAR * np.abs(omega) / (G.K_B * T)) - 1.0 + 1e-12)
    G.bose = bose_buggy
    G._bose = bose_buggy
    G._MODES["soft"]["F2"] = 1.0
    G._MODES["acoustic"]["F2"] = 0.8
    G._MODES["optical"]["F2"] = 0.3
    G._calibrate_F2()
    return bose_buggy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--archs", type=str, nargs="+", default=["5a", "5b"])
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    f2_fixed = G._MODES["soft"]["F2"]
    install_buggy_bose()
    f2_buggy = G._MODES["soft"]["F2"]
    # verify the patch actually reaches the lineshape
    grid = np.array([-5.0, 5.0])
    S = G.dho(grid, 5.0, 1.0, 0.0, 300.0, 1.0)
    implied = float(S[1] / S[0])
    expect_buggy = float(np.exp(G.HBAR * 5.0 / (G.K_B * 300.0)))
    expect_fixed = float(np.exp(5.0 / (G.K_B * 300.0)))
    print(f"F2 soft: corrected {f2_fixed:.2f} -> buggy {f2_buggy:.2f} "
          f"(ratio {f2_buggy/f2_fixed:.6f})")
    print(f"detailed-balance ratio now {implied:.6f}; buggy expects {expect_buggy:.6f}, "
          f"fixed expects {expect_fixed:.6f}")
    assert abs(implied - expect_buggy) < 1e-9, "monkeypatch did not reach dho()"

    # import AFTER patching so the training module sees the patched generator
    from a4_train_v8 import NPZ, prepare, train_one  # noqa: E402
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    sp = prepare(NPZ, "v7")

    rows = []
    for arch in args.archs:
        for seed in args.seeds:
            r = train_one(arch, "v7", seed, sp, device, args.outdir / "checkpoints",
                          log_every=False)
            r.update(control="buggy_bose", dataset_version="v7_generator_semantics")
            rows.append(r)
            print(f"  {arch} seed {seed}: val MAE_logM {r['best_val_MAE_logM']:.4f} "
                  f"@ epoch {r['best_epoch']}/{r['epochs_trained']}", flush=True)
            pd.DataFrame(rows).to_csv(args.outdir / "control_runs.csv", index=False)

    df = pd.DataFrame(rows)
    agg = df.groupby("arch")["best_val_MAE_logM"].agg(["mean", "std", "count"]).reset_index()
    agg.to_csv(args.outdir / "control_summary.csv", index=False)
    print("\n" + agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    (args.outdir / "run_sidecar.json").write_text(json.dumps({
        "task": "A4_control_buggy_bose", "script": str(Path(__file__).relative_to(ROOT)),
        "note": "harness identical to A4; bose() monkeypatched to the pre-fix expression; "
                "F2 recalibrated; tracked source unmodified",
        "F2_soft_corrected": f2_fixed, "F2_soft_buggy": f2_buggy,
        "seeds": list(args.seeds), "archs": list(args.archs),
        "date_utc": datetime.now(timezone.utc).isoformat()}, indent=2))


if __name__ == "__main__":
    main()
