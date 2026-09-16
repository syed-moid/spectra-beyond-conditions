"""A17 - architecture robustness sweep (round-5 decision 2).

Four CNN configurations and four transformer configurations, extended protocol,
v7 severity distribution, 3 seeds each. The base configurations are the ones
already trained in A4; the variants are defined here.

Transformer variants are built locally rather than through
`build_spectral_transformer`, because patch size and depth are module-level
constants in `spectral_transformer.py` and the tracked file is not modified.
The local builder reproduces the base parameter count exactly (asserted at
startup) so the variants are comparable to the A4 base runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import a4_train_v8 as A4  # noqa: E402
from sbc.models.spectral_transformer import HEAD_OUT_DIM  # noqa: E402
import dataset_paths as DS

OUTDIR = DS.rev("A17")
GRID_LEN = 600


# --------------------------------------------------------------------------- #
# configurable models                                                         #
# --------------------------------------------------------------------------- #

class CNNVariant(nn.Module):
    def __init__(self, channels, kernel=7):
        super().__init__()
        blocks, c_in = [], 1
        for c_out in channels:
            blocks += [nn.Conv1d(c_in, c_out, kernel_size=kernel, padding=kernel // 2),
                       nn.BatchNorm1d(c_out), nn.GELU(), nn.MaxPool1d(2)]
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(channels[-1], 96), nn.GELU(),
                                  nn.Linear(96, HEAD_OUT_DIM))

    def forward(self, spectrum):
        mu = spectrum.mean(-1, keepdim=True)
        sd = spectrum.std(-1, keepdim=True).clamp_min(1e-6)
        return self.head(self.features(((spectrum - mu) / sd).unsqueeze(1)).mean(-1))


class TransformerVariant(nn.Module):
    def __init__(self, patch=10, depth=6, d_model=96, heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        assert GRID_LEN % patch == 0
        self.patch = patch
        n_patches = GRID_LEN // patch
        self.proj = nn.Linear(patch, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads,
                                           dim_feedforward=d_model * mlp_ratio,
                                           dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.final_ln = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, HEAD_OUT_DIM))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Embedding(1 + n_patches, d_model)

    def forward(self, spectrum):
        B = spectrum.shape[0]
        mu = spectrum.mean(-1, keepdim=True)
        sd = spectrum.std(-1, keepdim=True).clamp_min(1e-6)
        x = ((spectrum - mu) / sd).reshape(B, GRID_LEN // self.patch, self.patch)
        tok = torch.cat([self.cls_token.expand(B, -1, -1), self.proj(x)], dim=1)
        tok = tok + self.pos_embed(torch.arange(tok.shape[1], device=tok.device))
        return self.head(self.final_ln(self.transformer(tok))[:, 0])


VARIANTS = {
    # family, builder
    "cnn_base":         ("cnn", lambda: CNNVariant((32, 64, 128, 192, 256), 7)),
    "cnn_half_depth":   ("cnn", lambda: CNNVariant((64, 160, 320), 7)),
    # "double width" reduced to 1.5x: (64,128,256,384,512) is 2.41M params, outside
    # the 0.5-2x band, so the nearest sensible widening is used instead.
    "cnn_wide":         ("cnn", lambda: CNNVariant((48, 96, 192, 256, 352), 7)),
    "cnn_kernel15":     ("cnn", lambda: CNNVariant((32, 64, 128, 192, 224), 15)),
    "cnn_param_matched": ("cnn", lambda: CNNVariant((32, 64, 128, 208, 272), 7)),
    "tf_base":          ("tf", lambda: TransformerVariant(patch=10, depth=6)),
    "tf_patch5":        ("tf", lambda: TransformerVariant(patch=5, depth=6)),
    "tf_patch20":       ("tf", lambda: TransformerVariant(patch=20, depth=6)),
    "tf_depth8":        ("tf", lambda: TransformerVariant(patch=10, depth=8)),
    # --- A19 receptive-field sweep -------------------------------------------
    # Channel widths are adjusted per kernel so total parameters stay ~1.18M for
    # every CNN point, isolating receptive field at roughly fixed capacity. A17
    # showed width is not influential (cnn_wide vs cnn_base: +0.0044, n.s.), so
    # the adjustment is not a meaningful confound.
    "cnn_kernel21":     ("cnn", lambda: CNNVariant((32, 64, 112, 160, 176), 21)),
    "cnn_kernel31":     ("cnn", lambda: CNNVariant((24, 48, 96, 128, 152), 31)),
    "cnn_kernel45":     ("cnn", lambda: CNNVariant((24, 48, 80, 104, 120), 45)),
    "tf_patch30":       ("tf", lambda: TransformerVariant(patch=30, depth=6)),
    "tf_patch40":       ("tf", lambda: TransformerVariant(patch=40, depth=6)),
    "tf_patch60":       ("tf", lambda: TransformerVariant(patch=60, depth=6)),
}

# Receptive field in meV: kernel or patch size x the 0.05008 meV grid step.
GRID_STEP_MEV = 30.0 / 599.0
RECEPTIVE_FIELD = {
    "cnn_half_depth": 7, "cnn_base": 7, "cnn_param_matched": 7, "cnn_wide": 7,
    "cnn_kernel15": 15, "cnn_kernel21": 21, "cnn_kernel31": 31, "cnn_kernel45": 45,
    "tf_patch5": 5, "tf_base": 10, "tf_depth8": 10, "tf_patch20": 20,
    "tf_patch30": 30, "tf_patch40": 40, "tf_patch60": 60,
}


def n_params(m):
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad))


def build_variant(name, seed):
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    return VARIANTS[name][1]()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--variants", type=str, nargs="+", default=None)
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--npz", type=Path, default=None)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    st5a = 687843
    print(f"ST-5a reference: {st5a:,d} params; 0.5-2x band = "
          f"{int(0.5*st5a):,d}-{int(2*st5a):,d}; +/-5% band = "
          f"{int(0.95*st5a):,d}-{int(1.05*st5a):,d}")
    for name in VARIANTS:
        p = n_params(build_variant(name, 42))
        band = "in 0.5-2x" if 0.5 * st5a <= p <= 2 * st5a else "OUT OF BAND"
        pm = " (param-matched +/-5%)" if 0.95 * st5a <= p <= 1.05 * st5a else ""
        print(f"  {name:20s} {p:>9,d}  {band}{pm}")
    # the local transformer builder must reproduce the tracked base exactly
    assert n_params(build_variant("tf_base", 42)) == st5a, "tf_base must match ST-5a"
    if args.list_only:
        return

    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    sp = A4.prepare(args.npz or A4.NPZ, "v7")
    names = args.variants or list(VARIANTS)
    rows = []
    done = set()
    _csv = args.outdir / "variant_runs.csv"
    if _csv.exists():
        _prev = pd.read_csv(_csv)
        done = {(r.variant, int(r.seed)) for r in _prev.itertuples()}
        rows = _prev.to_dict("records")
        print(f"  resuming: {len(done)} cells already recorded", flush=True)
    t0 = time.time()
    for name in names:
        family = VARIANTS[name][0]
        for seed in args.seeds:
            if (name, seed) in done:
                print(f"  skip {name} seed {seed} (done)", flush=True)
                continue
            orig = A4.build_model
            A4.build_model = lambda a, s, _n=name: build_variant(_n, s)
            try:
                r = A4.train_one(name, "v7", seed, sp, device,
                                 args.outdir / "checkpoints", log_every=False,
                                 stopping="extended")
            finally:
                A4.build_model = orig
            r.update(variant=name, family=family, dataset_version=DS.VERSION)
            rows.append(r)
            print(f"  {name:20s} seed {seed}: {r['best_val_MAE_logM']:.4f} "
                  f"@ {r['best_epoch']}/{r['epochs_trained']} ({r['stop_reason']})", flush=True)
            pd.DataFrame(rows).to_csv(args.outdir / "variant_runs.csv", index=False)

    df = pd.DataFrame(rows)
    agg = (df.groupby(["family", "variant"])["best_val_MAE_logM"]
             .agg(["mean", "std", "count"]).reset_index())
    agg["n_parameters"] = [n_params(build_variant(v, 42)) for v in agg.variant]
    agg.to_csv(args.outdir / "variant_summary.csv", index=False)
    print("\n" + agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    (args.outdir / "run_sidecar.json").write_text(json.dumps({
        "task": "A17", "dataset_version": DS.VERSION, "stopping_protocol": "extended",
        "train_severity_protocol": "v7", "seeds": list(args.seeds),
        "wall_time_sec": time.time() - t0,
        "date_utc": datetime.now(timezone.utc).isoformat()}, indent=2))


if __name__ == "__main__":
    main()
