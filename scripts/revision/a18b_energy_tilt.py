"""A18b - sensitivity to an energy-dependent efficiency tilt (evaluation-only).

A18 established that every model is exactly invariant to a global gain and an
additive offset, because each standardizes per-sample. That invariance does NOT
extend to an energy-dependent gain, which deforms the spectrum rather than
rescaling it. A real detector's efficiency and self-shielding vary with energy
transfer, so this is the concrete boundary of the normalization-robustness
statement.

Tilt applied after augmentation and before standardization:

    g(omega) = 1 + beta * (omega / 15 meV),  beta in {+-0.05, +-0.10, +-0.20}

Reports delta MAE on val and delta within-tuple rho on the replicate set.
"""

from __future__ import annotations

import argparse, json, sys
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
A4E = DS.rev("A4_extended") / "checkpoints"
OUTDIR = DS.rev("A18b")
BETAS = (-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20)
SEEDS = (42, 43, 44)


class VariantModel:
    """Wrapper for an A17/A19 variant checkpoint, same interface as A4Model."""

    def __init__(self, name, seed, ckpt_dir):
        ck = torch.load(ckpt_dir / f"{name}_v7_seed{seed}.pt", map_location="cpu",
                        weights_only=False)
        self.net = build_variant(name, seed)
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.stats = TargetStats.from_dict(ck["target_stats"])

    @torch.no_grad()
    def predict_logM(self, batch, chunk=2048):
        X = np.asarray(batch["spectrum"], dtype=np.float32)
        out = [self.net(torch.from_numpy(X[i:i + chunk])).numpy() for i in range(0, len(X), chunk)]
        o = np.concatenate(out, 0)
        return np.log(np.clip(unstandardize_outputs(o, self.stats)["M"], 1e-9, None))


def per_tuple_rho(tid, y, pred):
    out = []
    for t in np.unique(tid):
        m = tid == t
        if np.ptp(pred[m]) == 0 or np.ptp(y[m]) == 0:
            out.append(0.0)
        else:
            r = spearmanr(y[m], pred[m]).statistic
            out.append(0.0 if not np.isfinite(r) else float(r))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    ap.add_argument("--tuned", type=str, nargs="*", default=None,
                    help="tuned variant names to include (from A17/A19)")
    ap.add_argument("--tuned-dir", type=Path, default=None)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    ds = InsSpectraDataset(NPZ, "val", severity=1.0, as_torch=False)
    X = np.stack([ds.get_augmented(i, 1.0) for i in range(len(ds))]).astype(np.float32)
    y = np.log(np.clip(ds._M.astype(float), 1e-9, None))
    cond = {"T_K": ds._T_K.astype(float), "c_pct": ds._c_pct.astype(float),
            "E_kVcm": ds._E_kVcm.astype(float)}
    grid = ds._omega_grid

    clean, blocks = load_replicate()
    Xr = blocks[block_key(1.0)]
    tid = clean["tuple_id"]
    yr = np.log(np.clip(clean["M"].astype(float), 1e-9, None))
    condr = {"T_K": clean["T_K"].astype(float), "c_pct": clean["c_pct"].astype(float),
             "E_kVcm": clean["E_kVcm"].astype(float)}

    models = {}
    for arch in ("cnn", "5a", "5b", "fusion"):
        models[arch] = [load_a4_model(arch, "v7", s, ckpt_dir=A4E) for s in SEEDS]
    for name in (args.tuned or []):
        models[name] = [VariantModel(name, s, args.tuned_dir) for s in SEEDS]

    rows = []
    for label, ms in models.items():
        for beta in BETAS:
            tilt = (1.0 + beta * (grid / 15.0)).astype(np.float32)
            pv = np.mean([m.predict_logM({"spectrum": (X * tilt).astype(np.float32), **cond})
                          for m in ms], axis=0)
            pr = np.mean([m.predict_logM({"spectrum": (Xr * tilt).astype(np.float32), **condr})
                          for m in ms], axis=0)
            rows.append({"dataset_version": DS.VERSION, "model": label, "beta": beta,
                         "val_MAE_logM": float(np.abs(pv - y).mean()),
                         "replicate_rho": per_tuple_rho(tid, yr, pr)})
            print(f"  {label:18s} beta={beta:+.2f}  val MAE {rows[-1]['val_MAE_logM']:.4f}  "
                  f"rho {rows[-1]['replicate_rho']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    base = df[df.beta == 0.0].set_index("model")
    df["delta_MAE"] = df.apply(lambda r: r.val_MAE_logM - base.loc[r.model, "val_MAE_logM"], axis=1)
    df["delta_rho"] = df.apply(lambda r: r.replicate_rho - base.loc[r.model, "replicate_rho"], axis=1)
    df.to_csv(args.outdir / "energy_tilt_sensitivity.csv", index=False)
    print("\n" + df.pivot_table(index="model", columns="beta", values="delta_MAE")
          .to_string(float_format=lambda x: f"{x:+.4f}"))
    (args.outdir / "run_sidecar.json").write_text(json.dumps({
        "task": "A18b", "dataset_version": DS.VERSION, "betas": list(BETAS), "seeds": list(SEEDS),
        "tilt": "g(w) = 1 + beta*(w/15 meV), applied after augmentation before standardization",
        "date_utc": datetime.now(timezone.utc).isoformat()}, indent=2))


if __name__ == "__main__":
    main()
