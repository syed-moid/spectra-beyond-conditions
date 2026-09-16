"""Network errors on exactly the 600 fitting cases (round 49 item 7).

Figures 1, 2 and 8 put networks, fits and the conditions-only reference on one
axis. Until now the network series came from the full 2000-spectrum test set
while the fits came from a 600-spectrum subset of it, so the series did not
describe the same population.

This evaluates every architecture on **exactly the 600 cases the fits use**, at
each severity, and reports **per-run** errors so the manuscript's stated
aggregation -- mean over the nine runs, with SD -- can be formed without an
ensemble anywhere.

Evaluation only; no training.
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import a4_train_v8 as A4
from a17_arch_sweep import build_variant
from sbc.data.dataset import InsSpectraDataset
from sbc.models.spectral_transformer import TargetStats, unstandardize_outputs
from sbc.models.nonlinear_conditions_mlp import normalize_conditions
import dataset_paths as DS

MODELS = [("cnn_kernel45", "variant"), ("tf_patch60", "variant"), ("fusion", "base"),
          ("cnn", "base"), ("5a", "base"), ("5b", "base")]
DS_SEEDS, TR_SEEDS = (0, 1, 2), (42, 43, 44)
SEVS = (0.5, 1.0, 2.0, 4.0)


def load(model, kind, ds, sd, root):
    ck = torch.load(root / f"ds{ds}" / f"{model}_v7_seed{sd}.pt", map_location="cpu",
                    weights_only=False)
    net = build_variant(model, sd) if kind == "variant" else A4.build_model(ck["arch"], sd)
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net, TargetStats.from_dict(ck["target_stats"])


@torch.no_grad()
def pred(net, stats, X, cond=None, chunk=2048):
    o = []
    for i in range(0, len(X), chunk):
        xb = torch.from_numpy(X[i:i + chunk].astype(np.float32))
        o.append(net(xb).numpy() if cond is None else
                 net(xb, torch.from_numpy(cond[i:i + chunk].astype(np.float32))).numpy())
    return unstandardize_outputs(np.concatenate(o, 0), stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--severities", type=float, nargs="+", default=list(SEVS))
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()
    out = args.outdir or DS.rev("A23"); out.mkdir(parents=True, exist_ok=True)

    d = InsSpectraDataset(DS.TEST, "test", severity=1.0, as_torch=False)
    idx = np.linspace(0, len(d) - 1, min(args.n, len(d))).astype(int)   # identical to A23
    y = np.log(np.clip(d._M[idx].astype(float), 1e-9, None))
    cond = normalize_conditions(d._T_K[idx], d._c_pct[idx], d._E_kVcm[idx])
    root = DS.rev("A21") / "checkpoints"
    print(f"  {len(idx)} cases (the fitting subset), severities {args.severities}")

    rows = []
    for sev in args.severities:
        X = np.stack([d.get_augmented(int(i), sev) for i in idx]).astype(np.float32)
        for model, kind in MODELS:
            for ds_ in DS_SEEDS:
                for sd in TR_SEEDS:
                    net, st = load(model, kind, ds_, sd, root)
                    P = pred(net, st, X, cond if model == "fusion" else None)
                    lm = np.log(np.clip(P["M"], 1e-9, None))
                    rows.append({"model": model, "severity": sev, "dataset_seed": ds_,
                                 "seed": sd, "n": len(idx),
                                 "MAE_logM": float(np.abs(lm - y).mean())})
        print(f"  s={sev:<4g} done", flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(out / "network_runs_on_fitting_subset.csv", index=False)

    agg = per.groupby(["model", "severity"], as_index=False).agg(
        MAE_logM=("MAE_logM", "mean"), sd=("MAE_logM", "std"), n_runs=("MAE_logM", "size"))
    agg.to_csv(out / "network_by_severity_fitting_subset.csv", index=False)
    print("\n" + agg.pivot(index="model", columns="severity",
                           values="MAE_logM").round(4).to_string())
    (out / "run_sidecar_network_subset.json").write_text(json.dumps({
        "task": "A34_network_on_fitting_subset", "dataset_version": DS.VERSION,
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "eval_set": str(DS.TEST.relative_to(ROOT)),
        "subset": "np.linspace(0, N-1, 600) of the test split; identical to A23",
        "aggregation": "per-run MAE; mean and SD over the 9 runs. No ensemble.",
        "training": "none",
        "date_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"  wrote {out}/network_by_severity_fitting_subset.csv")


if __name__ == "__main__":
    main()
