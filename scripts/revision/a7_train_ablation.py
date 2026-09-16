"""A7 - train on one channel-ablation dataset (one setting, one dataset seed).

ST-5a is trained for every setting; the base CNN additionally for `both` and
`alpha_per_mode` (the shared-alpha comparison). Extended protocol, cap 200, v7
severity distribution. Results append incrementally.
"""

from __future__ import annotations

import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import a4_train_v8 as A4  # noqa: E402
import dataset_paths as DS  # noqa: E402

REV = DS.rev("A7")
DATA = ROOT / "data"
SEEDS = (42, 43, 44)
CNN_SETTINGS = ("both", "alpha_per_mode")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True)
    ap.add_argument("--dataset-seed", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()
    REV.mkdir(parents=True, exist_ok=True)
    npz = DS.ablation(args.setting, args.dataset_seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    sp = A4.prepare(npz, "v7")
    archs = ["5a"] + (["cnn"] if args.setting in CNN_SETTINGS else [])
    out = REV / "training_runs.csv"
    ck = REV / "checkpoints" / f"{args.setting}_seed{args.dataset_seed}"
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        done = {(r.setting, int(r.dataset_seed), r.arch, int(r.seed)) for r in prev.itertuples()}
    for arch in archs:
        for s in args.seeds:
            if (args.setting, args.dataset_seed, arch, s) in done:
                print(f"  skip {args.setting} ds{args.dataset_seed} {arch} seed{s} (done)",
                      flush=True)
                continue
            t0 = time.time()
            r = A4.train_one(arch, "v7", s, sp, device, ck, log_every=False,
                             stopping="extended")
            r.update(setting=args.setting, dataset_seed=args.dataset_seed,
                     dataset=str(npz.relative_to(ROOT)),
                     run_id=f"a7_{args.setting}_ds{args.dataset_seed}",
                     date_utc=datetime.now(timezone.utc).isoformat())
            pd.DataFrame([r]).to_csv(out, mode="a", header=not out.exists(), index=False)
            print(f"  {args.setting} ds{args.dataset_seed} {arch} seed{s}: "
                  f"{r['best_val_MAE_logM']:.4f} @ {r['best_epoch']}/{r['epochs_trained']} "
                  f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
