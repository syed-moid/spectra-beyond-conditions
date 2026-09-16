"""Main-text models trained on 3 dataset realizations x 3 training seeds (round 42/43 Part 3.1).

The 5-seed single-dataset protocol is retired. Each of the six main-text models is
trained on three independent realizations of the standard generator
(`ablation_v10_both_seed{0,1,2}` -- the `both` setting is the unmodified Phase 1
latent model, so these are three draws of the standard dataset, not an ablation),
with three training seeds each: 9 runs per model, 54 in total.

Selection uses each realization's own `val` split. Headline evaluation happens
later on the untouched `test_v10` / `replicate_test_v10` sets, which no run in
this script ever sees.

Appends incrementally and resumes on entry.
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
from a17_arch_sweep import build_variant, VARIANTS  # noqa: E402
import dataset_paths as DS  # noqa: E402

# model -> (kind, arch/variant name). "base" trains through A4.PLAN archs;
# "variant" swaps A4.build_model for the tuned configuration.
MODELS = [
    ("5a",           "base",    "5a"),
    ("5b",           "base",    "5b"),
    ("fusion",       "base",    "fusion"),
    ("cnn",          "base",    "cnn"),
    ("cnn_kernel45", "variant", "cnn_kernel45"),
    ("tf_patch60",   "variant", "tf_patch60"),
]
DATASET_SEEDS = (0, 1, 2)
TRAIN_SEEDS = (42, 43, 44)
KEY = ["model", "dataset_seed", "seed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--dataset-seeds", type=int, nargs="+", default=list(DATASET_SEEDS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(TRAIN_SEEDS))
    ap.add_argument("--models", type=str, nargs="+", default=[m[0] for m in MODELS])
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--cap", type=int, default=None,
                    help="override the stopping rule's max_epochs (protocol-health check)")
    args = ap.parse_args()

    out = args.outdir or DS.rev("A21")
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "training_runs.csv"
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    if args.cap:
        A4.STOPPING["extended"]["max_epochs"] = int(args.cap)
        print(f"  cap override: max_epochs = {args.cap}")

    done, rows = set(), []
    if csv.exists():
        prev = pd.read_csv(csv)
        rows = prev.to_dict("records")
        done = {(r.model, int(r.dataset_seed), int(r.seed)) for r in prev.itertuples()}
        print(f"  resuming: {len(done)} runs already recorded", flush=True)

    wanted = [m for m in MODELS if m[0] in set(args.models)]
    t0 = time.time()
    for ds_seed in args.dataset_seeds:
        npz = DS.ablation("both", ds_seed)
        if not npz.exists():
            print(f"  MISSING dataset realization {npz}"); continue
        sp = A4.prepare(npz, "v7")
        for name, kind, arch in wanted:
            for seed in args.seeds:
                if (name, ds_seed, seed) in done:
                    print(f"  skip {name} ds{ds_seed} seed{seed} (done)", flush=True)
                    continue
                ck = out / "checkpoints" / f"ds{ds_seed}"
                t1 = time.time()
                if kind == "variant":
                    orig = A4.build_model
                    A4.build_model = lambda a, s, _n=arch: build_variant(_n, s)
                    try:
                        r = A4.train_one(name, "v7", seed, sp, device, ck,
                                         log_every=False, stopping="extended")
                    finally:
                        A4.build_model = orig
                else:
                    r = A4.train_one(arch, "v7", seed, sp, device, ck,
                                     log_every=False, stopping="extended")
                r.update(model=name, dataset_seed=ds_seed,
                         dataset=str(npz.relative_to(ROOT)),
                         dataset_version=DS.VERSION,
                         run_id=f"a21_{name}_ds{ds_seed}_s{seed}",
                         date_utc=datetime.now(timezone.utc).isoformat())
                rows.append(r)
                pd.DataFrame(rows).to_csv(csv, index=False)
                print(f"  {name:14s} ds{ds_seed} seed{seed}: {r['best_val_MAE_logM']:.4f} "
                      f"@ {r['best_epoch']}/{r['epochs_trained']} ({time.time()-t1:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    if len(df):
        agg = (df.groupby("model")["best_val_MAE_logM"].agg(["mean", "std", "count"])
                 .reset_index().sort_values("mean"))
        agg.to_csv(out / "training_summary.csv", index=False)
        print("\n" + agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    (out / "run_sidecar.json").write_text(json.dumps({
        "task": "A21_multirealization",
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "dataset_version": DS.VERSION,
        "dataset_realizations": [str(DS.ablation("both", s).relative_to(ROOT))
                                 for s in args.dataset_seeds],
        "realization_note": "the `both` ablation setting is the unmodified Phase 1 latent "
                            "model; these are three draws of the standard dataset",
        "dataset_seeds": list(args.dataset_seeds), "training_seeds": list(args.seeds),
        "selection_split": "val (per realization)",
        "headline_eval": "test_v10 / replicate_test_v10 (untouched, never seen here)",
        "stopping_protocol": "extended",
        "cap_override": args.cap,
        "wall_time_sec": round(time.time() - t0, 1),
        "date_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
