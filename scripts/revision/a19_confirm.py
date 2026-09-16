"""A19 held-out confirmation + tilt-trained CNN evaluation.

A19: the per-family best variants were selected on val, so they are confirmed on
the stress base (s in {0.5,1,2,4}) and on replicate_eval_v8 (MAE and within-tuple
rho), neither of which was used for selection.

A18b decision 5: the tilt-trained CNN is evaluated for (a) in-distribution cost
vs the untilted base with a paired CI, (b) the full beta grid, (c) replicate MAE
and rho at s = 1.
"""

from __future__ import annotations

import json, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from a4_eval import load_a4_model  # noqa: E402
from a17_arch_sweep import build_variant  # noqa: E402
from replicate_io import block_key, load_replicate  # noqa: E402
from sbc.data.dataset import InsSpectraDataset  # noqa: E402
from sbc.models.spectral_transformer import TargetStats, unstandardize_outputs  # noqa: E402
import dataset_paths as DS

NPZ = DS.FULL
REV = ROOT / "results" / "revision"
A4E = DS.rev("A4_extended") / "checkpoints"
OUT = DS.rev("A19")
SEEDS = (42, 43, 44)
SEVS = (0.5, 1.0, 2.0, 4.0)
BETAS = (-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20)


class VariantModel:
    def __init__(self, name, seed, ckpt_dir, tag=""):
        ck = torch.load(ckpt_dir / f"{name}{tag}.pt", map_location="cpu", weights_only=False)
        self.net = build_variant(name.rsplit("_v7_seed", 1)[0], seed) if "_v7_seed" in name else None
        raise RuntimeError


def load_variant(name, seed, ckpt_dir=None):
    """Load a tuned variant. ckpt_dir is ignored unless given explicitly: the
    tuned variants live in A19 on v9 and in A17 on v10, so the location is
    resolved centrally (dataset_paths.tuned_ckpt) rather than assumed here."""
    path = (ckpt_dir / f"{name}_v7_seed{seed}.pt") if ckpt_dir else DS.tuned_ckpt(name, seed)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = build_variant(name, seed); net.load_state_dict(ck["state_dict"]); net.eval()
    return net, TargetStats.from_dict(ck["target_stats"])


@torch.no_grad()
def predict(net, stats, X, chunk=2048):
    o = np.concatenate([net(torch.from_numpy(X[i:i + chunk].astype(np.float32))).numpy()
                        for i in range(0, len(X), chunk)], 0)
    return np.log(np.clip(unstandardize_outputs(o, stats)["M"], 1e-9, None))


def tuple_rho(tid, y, p):
    out = []
    for t in np.unique(tid):
        m = tid == t
        r = spearmanr(y[m], p[m]).statistic if np.ptp(p[m]) > 0 else 0.0
        out.append(0.0 if not np.isfinite(r) else float(r))
    return float(np.mean(out))


def paired(a, b, rng, n=1000):
    d = a - b; i = rng.integers(0, len(d), size=(n, len(d))); m = d[i].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260906)
    stress = InsSpectraDataset(NPZ, "stress_base", severity=1.0, as_torch=False)
    ys = np.log(np.clip(stress._M.astype(float), 1e-9, None))
    conds = {"T_K": stress._T_K.astype(float), "c_pct": stress._c_pct.astype(float),
             "E_kVcm": stress._E_kVcm.astype(float)}
    Xs = {s: np.stack([stress.get_augmented(i, s) for i in range(len(stress))]).astype(np.float32)
          for s in SEVS}
    clean, blocks = load_replicate()
    tid = clean["tuple_id"]
    yr = np.log(np.clip(clean["M"].astype(float), 1e-9, None))
    condr = {"T_K": clean["T_K"].astype(float), "c_pct": clean["c_pct"].astype(float),
             "E_kVcm": clean["E_kVcm"].astype(float)}

    tuned = ("cnn_kernel45", "tf_patch60")
    rows = []
    for name in tuned:
        nets = [load_variant(name, s) for s in SEEDS]
        for s in SEVS:
            per = [np.abs(predict(n, st, Xs[s]) - ys).mean() for n, st in nets]
            p = np.mean([predict(n, st, Xs[s]) for n, st in nets], axis=0)
            rows.append({"dataset_version": DS.VERSION, "model": name, "set": "stress_base",
                         "severity": s, "MAE_mean": float(np.mean(per)),
                         "MAE_sd": float(np.std(per, ddof=1)),
                         "MAE_ensemble": float(np.abs(p - ys).mean()), "rho": np.nan})
        for s in SEVS:
            Xr = blocks[block_key(s)]
            per = [np.abs(predict(n, st, Xr) - yr).mean() for n, st in nets]
            p = np.mean([predict(n, st, Xr) for n, st in nets], axis=0)
            rows.append({"dataset_version": DS.VERSION, "model": name, "set": "replicate_eval",
                         "severity": s, "MAE_mean": float(np.mean(per)),
                         "MAE_sd": float(np.std(per, ddof=1)),
                         "MAE_ensemble": float(np.abs(p - yr).mean()),
                         "rho": tuple_rho(tid, yr, p)})
            print(f"  {name:14s} replicate s={s}: MAE {np.mean(per):.4f} rho {rows[-1]['rho']:.4f}",
                  flush=True)
    # base models on the same axes
    for arch in ("cnn", "5a", "fusion", "5b"):
        ms = [load_a4_model(arch, "v7", s, ckpt_dir=A4E) for s in SEEDS]
        for s in SEVS:
            p = np.mean([m.predict_logM({"spectrum": Xs[s], **conds}) for m in ms], axis=0)
            rows.append({"dataset_version": DS.VERSION, "model": arch, "set": "stress_base",
                         "severity": s, "MAE_mean": np.nan, "MAE_sd": np.nan,
                         "MAE_ensemble": float(np.abs(p - ys).mean()), "rho": np.nan})
            Xr = blocks[block_key(s)]
            pr = np.mean([m.predict_logM({"spectrum": Xr, **condr}) for m in ms], axis=0)
            rows.append({"dataset_version": DS.VERSION, "model": arch, "set": "replicate_eval",
                         "severity": s, "MAE_mean": np.nan, "MAE_sd": np.nan,
                         "MAE_ensemble": float(np.abs(pr - yr).mean()),
                         "rho": tuple_rho(tid, yr, pr)})
    pd.DataFrame(rows).to_csv(OUT / "confirmation_heldout.csv", index=False)
    print("\n" + pd.DataFrame(rows).pivot_table(index="model", columns=["set", "severity"],
                                                values="MAE_ensemble")
          .to_string(float_format=lambda x: f"{x:.4f}"))

    # ---- tilt-trained CNN ---------------------------------------------------
    tdir = DS.rev("A18b_tilt_trained") / "checkpoints"
    val = InsSpectraDataset(NPZ, "val", severity=1.0, as_torch=False)
    Xv = np.stack([val.get_augmented(i, 1.0) for i in range(len(val))]).astype(np.float32)
    yv = np.log(np.clip(val._M.astype(float), 1e-9, None))
    condv = {"T_K": val._T_K.astype(float), "c_pct": val._c_pct.astype(float),
             "E_kVcm": val._E_kVcm.astype(float)}
    grid = val._omega_grid
    tms = []
    for s in SEEDS:
        ck = torch.load(tdir / f"cnn_v7_seed{s}_tilt.pt", map_location="cpu", weights_only=False)
        from a4_train_v8 import build_model
        net = build_model("cnn", s); net.load_state_dict(ck["state_dict"]); net.eval()
        tms.append((net, TargetStats.from_dict(ck["target_stats"])))
    bms = [load_a4_model("cnn", "v7", s, ckpt_dir=A4E) for s in SEEDS]

    pt = np.mean([predict(n, st, Xv) for n, st in tms], axis=0)
    pb = np.mean([bm.predict_logM({"spectrum": Xv, **condv}) for bm in bms], axis=0)
    d, lo, hi = paired(np.abs(pt - yv), np.abs(pb - yv), rng)
    trows = [{"dataset_version": DS.VERSION, "quantity": "val_MAE_tilt_trained",
              "value": float(np.abs(pt - yv).mean())},
             {"dataset_version": DS.VERSION, "quantity": "val_MAE_base", "value": float(np.abs(pb - yv).mean())},
             {"dataset_version": DS.VERSION, "quantity": "paired_diff_tilt_minus_base", "value": d},
             {"dataset_version": DS.VERSION, "quantity": "ci_lo", "value": lo},
             {"dataset_version": DS.VERSION, "quantity": "ci_hi", "value": hi}]
    print(f"\ntilt-trained CNN val {np.abs(pt-yv).mean():.4f} vs base {np.abs(pb-yv).mean():.4f}; "
          f"paired {d:+.4f} [{lo:+.4f},{hi:+.4f}]")
    grid_rows = []
    for b in BETAS:
        tl = (1.0 + b * (grid / 15.0)).astype(np.float32)
        p_t = np.mean([predict(n, st, (Xv * tl).astype(np.float32)) for n, st in tms], axis=0)
        p_b = np.mean([bm.predict_logM({"spectrum": (Xv * tl).astype(np.float32), **condv})
                       for bm in bms], axis=0)
        grid_rows.append({"dataset_version": DS.VERSION, "beta": b,
                          "tilt_trained_MAE": float(np.abs(p_t - yv).mean()),
                          "base_MAE": float(np.abs(p_b - yv).mean())})
        print(f"  beta={b:+.2f}: tilt-trained {grid_rows[-1]['tilt_trained_MAE']:.4f}  "
              f"base {grid_rows[-1]['base_MAE']:.4f}")
    Xr1 = blocks[block_key(1.0)]
    prt = np.mean([predict(n, st, Xr1) for n, st in tms], axis=0)
    trows.append({"dataset_version": DS.VERSION, "quantity": "replicate_MAE_s1",
                  "value": float(np.abs(prt - yr).mean())})
    trows.append({"dataset_version": DS.VERSION, "quantity": "replicate_rho_s1",
                  "value": tuple_rho(tid, yr, prt)})
    print(f"  replicate s=1: MAE {trows[-2]['value']:.4f}  rho {trows[-1]['value']:.4f}")
    outd = DS.rev("A18b_tilt_trained")
    outd.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trows).to_csv(outd / "tilt_trained_summary.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(outd / "tilt_trained_beta_grid.csv", index=False)
    (outd / "run_sidecar_eval.json").write_text(json.dumps({
        "task": "A18b_tilt_eval", "seeds": list(SEEDS), "betas": list(BETAS),
        "date_utc": datetime.now(timezone.utc).isoformat()}, indent=2))


if __name__ == "__main__":
    main()
