"""A5 negative control - label permutation.

Train ST-5a on v8 with the log M training target randomly permuted, 2 seeds.
The auxiliary omega0 / log Gamma targets are left intact, which is the
conservative choice: the shared representation can still learn real spectral
structure, so any residual skill on log M would be a genuine leak. Test MAE on
log M must land at or above the marginal-median predictor, which is what a model
with no usable signal converges to.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import a4_train_v8 as A4  # noqa: E402
from a4_train_v8 import prepare, train_one  # noqa: E402
from sbc.models.spectral_transformer import unstandardize_outputs  # noqa: E402
import dataset_paths as DS  # noqa: E402

NPZ = DS.FULL
OUTDIR = DS.rev("A5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--permute-seed", type=int, default=20260905)
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))

    sp = prepare(NPZ, "v7")
    rng = np.random.default_rng(args.permute_seed)

    # Permute the log M column of BOTH train and val, and permute the natural-units
    # val_M used for checkpoint selection identically. Selecting the best epoch on the
    # TRUE validation labels would leak exactly the association this control destroys:
    # the first version of this script did that and scored BELOW the marginal-median
    # predictor as a result. Selection is now blind to the true targets; the selected
    # checkpoint is scored against them only afterwards.
    perm_tr = rng.permutation(len(sp.train_y))
    perm_va = rng.permutation(len(sp.val_y))
    y_tr = sp.train_y.copy(); y_tr[:, 0] = y_tr[perm_tr, 0]
    y_va = sp.val_y.copy(); y_va[:, 0] = y_va[perm_va, 0]
    sp_perm = replace(sp, train_y=y_tr, val_y=y_va, val_M=sp.val_M[perm_va])

    log_true = np.log(np.clip(sp.val_M, 1e-9, None))
    marginal_median = float(np.abs(log_true - np.median(log_true)).mean())
    marginal_mean = float(np.abs(log_true - log_true.mean()).mean())

    rows = []
    for seed in args.seeds:
        r = train_one("5a", "v7", seed, sp_perm, device,
                      args.outdir / "checkpoints_label_perm", log_every=False,
                      stopping="extended", tag_suffix="_labelperm")
        ck = torch.load(args.outdir / "checkpoints_label_perm" /
                        f"5a_v7_seed{seed}_labelperm.pt", map_location="cpu",
                        weights_only=False)
        net = A4.build_model("5a", seed); net.load_state_dict(ck["state_dict"]); net.eval()
        with torch.no_grad():
            out = np.concatenate([net(torch.from_numpy(sp.val_x[i:i + 2048])).numpy()
                                  for i in range(0, len(sp.val_x), 2048)], 0)
        pred = np.log(np.clip(unstandardize_outputs(out, sp.stats)["M"], 1e-9, None))
        true_mae = float(np.abs(pred - log_true).mean())
        r.update(dataset_version=DS.VERSION, control="label_permutation",
                 permute_seed=args.permute_seed,
                 val_MAE_on_permuted_labels=r["best_val_MAE_logM"],
                 val_MAE_on_true_labels=true_mae,
                 marginal_median_MAE_logM=marginal_median,
                 marginal_mean_MAE_logM=marginal_mean,
                 ratio_to_marginal_median=true_mae / marginal_median)
        rows.append(r)
        print(f"  seed {seed}: MAE on TRUE labels {true_mae:.4f}  (marginal-median "
              f"{marginal_median:.4f}, marginal-mean {marginal_mean:.4f}, "
              f"ratio {true_mae / marginal_median:.3f})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.outdir / "label_permutation.csv", index=False)
    (args.outdir / "run_sidecar_label_perm.json").write_text(json.dumps({
        "task": "A5_label_permutation", "script": str(Path(__file__).relative_to(ROOT)),
        "dataset_version": DS.VERSION, "seeds": list(args.seeds), "permute_seed": args.permute_seed,
        "note": "log M training column permuted; omega0/log Gamma aux targets left intact",
        "marginal_median_MAE_logM": marginal_median,
        "marginal_mean_MAE_logM": marginal_mean,
        "date_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"\nwrote {args.outdir}")


if __name__ == "__main__":
    main()
