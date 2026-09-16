"""Step 4 - training block on v8: ST-5a, ST-5b, fusion, 1D CNN, plus A16 protocols.

Architectures
    5a      v7 learned-patch Spectral Transformer (spectrum only)
    5b      v7 fixed-RFF Spectral Transformer (spectrum only)
    fusion  5a with the normalized (T, c, E) vector concatenated to the pooled
            CLS representation before the regression head
    cnn     1D convolutional encoder, spectrum only, parameter count within
            +/-20% of 5a

Severity protocols (A16)
    v7        per-sample severity ~ log-uniform[0.25, 4]   (the v7 protocol)
    fixed1    every training sample at s = 1
    low       per-sample severity ~ log-uniform[0.25, 1]

Everything else - three-output head (log M, omega0 std, log Gamma std), loss
weights 1.0/0.1/0.1, AdamW lr 1e-3 wd 0.01, batch 64, 100 epochs max, linear
warmup 1000 steps then cosine, grad clip 1.0, early stop patience 10 on val
MAE_logM - is unchanged from v7 and imported from the v7 training script where
possible so the two cannot drift apart.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sbc.data.dataset import InsSpectraDataset  # noqa: E402
from sbc.models.nonlinear_conditions_mlp import normalize_conditions  # noqa: E402
from sbc.models.spectral_transformer import (  # noqa: E402
    EMBED_DIM, HEAD_OUT_DIM, N_PATCHES, OMEGA_GRID_LEN, PATCH_SIZE,
    SpectralTransformer, TargetStats, build_spectral_transformer,
    count_trainable_parameters, standardize_targets, unstandardize_outputs,
)
import dataset_paths as DS  # noqa: E402

NPZ = DS.FULL
OUTDIR = DS.rev("A4_extended")

# Frozen v7 hyperparameters (mirrored from scripts/train_spectral_transformer.py:77-108).
BATCH_SIZE, N_EPOCHS, LR, WEIGHT_DECAY = 64, 100, 1e-3, 0.01
WARMUP_STEPS, GRAD_CLIP, EARLY_STOP_PATIENCE = 1000, 1.0, 10
LOSS_W = (1.0, 0.1, 0.1)
SEVERITY_SAMPLING_SEED = 0          # fixed across training seeds (round-3 decision 1)

# Stopping rules. "v7" is the original: patience 10 on the raw val metric, cap 100,
# no minimum - which produced 9/29 runs truncated at the cap while still improving
# and 4/29 stopped at epochs 20-33. "extended" (round-4 decision 2) is the primary
# v8 protocol: minimum 30 epochs, cap 200, patience 20 on an EMA (alpha 0.3) of the
# val metric, best checkpoint still chosen on the RAW val MAE_logM.
STOPPING = {
    "v7":       dict(min_epochs=1,  max_epochs=100, patience=10, ema_alpha=None),
    "extended": dict(min_epochs=30, max_epochs=200, patience=20, ema_alpha=0.3),
}

PROTOCOLS = {
    "v7":     ("loguniform", 0.25, 4.0),
    "fixed1": ("fixed", 1.0, 1.0),
    "low":    ("loguniform", 0.25, 1.0),
}


# --------------------------------------------------------------------------- #
# models                                                                      #
# --------------------------------------------------------------------------- #

class FusionTransformer(nn.Module):
    """ST-5a whose pooled CLS token is concatenated with normalized (T, c, E)."""

    def __init__(self, backbone: SpectralTransformer):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(EMBED_DIM + 3, EMBED_DIM), nn.GELU(),
            nn.Linear(EMBED_DIM, HEAD_OUT_DIM),
        )

    def forward(self, spectrum: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        B = spectrum.shape[0]
        mu = spectrum.mean(dim=-1, keepdim=True)
        sd = spectrum.std(dim=-1, keepdim=True).clamp_min(1e-6)
        x = (spectrum - mu) / sd
        patches = x.reshape(B, N_PATCHES, PATCH_SIZE)
        tokens = torch.cat([b.cls_token.expand(B, -1, -1), b.patch_encoder(patches)], dim=1)
        tokens = tokens + b.pos_embed(torch.arange(tokens.shape[1], device=tokens.device))
        out = b.final_ln(b.transformer(tokens))[:, 0]
        return self.head(torch.cat([out, conditions], dim=-1))


class CNN1D(nn.Module):
    """Spectrum-only 1D CNN: 5 conv blocks, global average pool, same 3-output head."""

    CHANNELS = (32, 64, 128, 192, 256)

    def __init__(self):
        super().__init__()
        blocks = []
        c_in = 1
        for c_out in self.CHANNELS:
            blocks += [nn.Conv1d(c_in, c_out, kernel_size=7, padding=3),
                       nn.BatchNorm1d(c_out), nn.GELU(), nn.MaxPool1d(2)]
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(self.CHANNELS[-1], EMBED_DIM), nn.GELU(),
            nn.Linear(EMBED_DIM, HEAD_OUT_DIM),
        )

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        mu = spectrum.mean(dim=-1, keepdim=True)
        sd = spectrum.std(dim=-1, keepdim=True).clamp_min(1e-6)
        x = ((spectrum - mu) / sd).unsqueeze(1)
        return self.head(self.features(x).mean(dim=-1))


def build_model(arch: str, seed: int):
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if arch in ("5a", "5b"):
        return build_spectral_transformer(arch, seed)
    if arch == "fusion":
        return FusionTransformer(build_spectral_transformer("5a", seed))
    if arch == "cnn":
        return CNN1D()
    raise ValueError(arch)


def uses_conditions(arch: str) -> bool:
    return arch == "fusion"


# --------------------------------------------------------------------------- #
# data                                                                        #
# --------------------------------------------------------------------------- #

@dataclass
class Splits:
    train_x: np.ndarray
    train_cond: np.ndarray
    train_y: np.ndarray
    train_sev: np.ndarray
    val_x: np.ndarray
    val_cond: np.ndarray
    val_y: np.ndarray
    val_M: np.ndarray
    stats: TargetStats


def _severities(protocol: str, n: int) -> np.ndarray:
    kind, lo, hi = PROTOCOLS[protocol]
    if kind == "fixed":
        return np.full(n, lo, dtype=np.float32)
    rng = np.random.default_rng(SEVERITY_SAMPLING_SEED)
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size=n)).astype(np.float32)


def prepare(npz: Path, protocol: str) -> Splits:
    tr = InsSpectraDataset(npz, "train", severity=1.0, as_torch=False)
    va = InsSpectraDataset(npz, "val", severity=1.0, as_torch=False)
    sev = _severities(protocol, len(tr))
    tx = np.stack([tr.get_augmented(i, severity=float(sev[i])) for i in range(len(tr))]
                  ).astype(np.float32)
    vx = np.stack([va.get_augmented(i, severity=1.0) for i in range(len(va))]).astype(np.float32)
    stats = TargetStats.from_train(tr._omega_Q.astype(float), tr._Gamma_Q.astype(float))
    return Splits(
        train_x=tx,
        train_cond=normalize_conditions(tr._T_K, tr._c_pct, tr._E_kVcm),
        train_y=standardize_targets(tr._M.astype(float), tr._omega_Q.astype(float),
                                    tr._Gamma_Q.astype(float), stats),
        train_sev=sev,
        val_x=vx,
        val_cond=normalize_conditions(va._T_K, va._c_pct, va._E_kVcm),
        val_y=standardize_targets(va._M.astype(float), va._omega_Q.astype(float),
                                  va._Gamma_Q.astype(float), stats),
        val_M=va._M.astype(np.float64), stats=stats,
    )


# --------------------------------------------------------------------------- #
# training                                                                    #
# --------------------------------------------------------------------------- #

def multitask_loss(out, tgt):
    l = [nn.functional.mse_loss(out[:, i], tgt[:, i]) for i in range(3)]
    return LOSS_W[0] * l[0] + LOSS_W[1] * l[1] + LOSS_W[2] * l[2]


def lr_lambda(total_steps, warmup):
    def f(step):
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
    return f


def forward(model, arch, x, cond):
    return model(x, cond) if uses_conditions(arch) else model(x)


def train_one(arch, protocol, seed, sp: Splits, device, out_dir: Path, max_epochs=None,
              log_every=True, stopping="v7", tag_suffix="", train_tilt=0.0):
    """train_tilt > 0 applies a random energy-dependent efficiency tilt
    g(w) = 1 + beta*(w/15 meV), beta ~ U[-train_tilt, +train_tilt], per sample per
    epoch, to the TRAINING spectra only (A18b). Evaluation data is untouched."""
    rule = dict(STOPPING[stopping])
    if max_epochs is not None:
        rule["max_epochs"] = max_epochs
    max_epochs = rule["max_epochs"]
    model = build_model(arch, seed).to(device)
    n_par = count_trainable_parameters(model)
    Xtr = torch.from_numpy(sp.train_x).to(device)
    Ctr = torch.from_numpy(sp.train_cond).to(device)
    Ytr = torch.from_numpy(sp.train_y).to(device)
    Xva = torch.from_numpy(sp.val_x).to(device)
    Cva = torch.from_numpy(sp.val_cond).to(device)
    Yva = torch.from_numpy(sp.val_y).to(device)
    log_M_true = np.log(np.clip(sp.val_M, 1e-9, None))

    n = Xtr.shape[0]
    if train_tilt > 0.0:
        omega_norm = torch.linspace(-1.0, 1.0, Xtr.shape[1], device=device)
    steps = max(1, n // BATCH_SIZE) * rule["max_epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(steps, WARMUP_STEPS))
    g = torch.Generator().manual_seed(seed)

    best, best_ep, patience = float("inf"), 0, rule["patience"]
    best_ema, ema = float("inf"), None
    best_state = None
    stop_reason = "cap"
    history = []
    t0 = time.time()
    epoch_times = []
    for ep in range(1, max_epochs + 1):
        te = time.time()
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        tot = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = Xtr[idx]
            if train_tilt > 0.0:
                beta = (torch.rand(xb.shape[0], 1, device=device) * 2.0 - 1.0) * train_tilt
                xb = xb * (1.0 + beta * omega_norm)
            out = forward(model, arch, xb, Ctr[idx])
            loss = multitask_loss(out, Ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            sched.step()
            tot += float(loss.detach().cpu()) * len(idx)
        model.eval()
        with torch.no_grad():
            vo = forward(model, arch, Xva, Cva)
            vloss = float(multitask_loss(vo, Yva).cpu())
            nat = unstandardize_outputs(vo.cpu().numpy(), sp.stats)
            vmae = float(np.mean(np.abs(np.log(np.clip(nat["M"], 1e-9, None)) - log_M_true)))
        epoch_times.append(time.time() - te)
        # Best checkpoint always tracks the RAW metric.
        if vmae < best - 1e-5:
            best, best_ep = vmae, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        # Stopping tracks either the raw metric (v7) or its EMA (extended).
        if rule["ema_alpha"] is None:
            monitor = vmae
        else:
            a = rule["ema_alpha"]
            ema = vmae if ema is None else a * vmae + (1.0 - a) * ema
            monitor = ema
        history.append({"epoch": ep, "train_loss": tot / n, "val_loss": vloss,
                        "val_MAE_logM": vmae, "val_MAE_logM_ema": ema,
                        "lr": sched.get_last_lr()[0]})
        if monitor < best_ema - 1e-5:
            best_ema, patience = monitor, rule["patience"]
        else:
            patience -= 1
        if log_every and (ep % 10 == 0 or ep == 1):
            print(f"    ep {ep:3d} train {tot/n:.4f} val {vloss:.4f} "
                  f"val_MAE_logM {vmae:.4f} (best {best:.4f} @ {best_ep})", flush=True)
        if patience <= 0 and ep >= rule["min_epochs"]:
            stop_reason = "patience"
            break

    model.load_state_dict(best_state)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{arch}_{protocol}_seed{seed}{tag_suffix}"
    torch.save({"arch": arch, "protocol": protocol, "seed": seed,
                "state_dict": best_state, "target_stats": sp.stats.to_dict(),
                "epoch": best_ep, "val_MAE_logM": best, "name": tag,
                "stopping": stopping, "dataset_version": DS.VERSION}, out_dir / f"{tag}.pt")
    pd.DataFrame(history).to_csv(out_dir / f"{tag}_history.csv", index=False)
    cap_hit = stop_reason == "cap"
    return {"arch": arch, "protocol": protocol, "seed": seed, "dataset_version": DS.VERSION,
            "stopping_protocol": stopping, "min_epochs": rule["min_epochs"],
            "max_epochs": rule["max_epochs"], "patience": rule["patience"],
            "ema_alpha": rule["ema_alpha"], "train_tilt": train_tilt,
            "stop_epoch": len(history),
            "stop_reason": stop_reason, "cap_hit": bool(cap_hit),
            "best_within_5_of_cap": bool(cap_hit and (rule["max_epochs"] - best_ep) <= 5),
            "n_parameters": n_par, "epochs_trained": len(history), "best_epoch": best_ep,
            "best_val_MAE_logM": best, "wall_time_sec": time.time() - t0,
            "mean_epoch_sec": float(np.mean(epoch_times)),
            "severity_sampling_seed": SEVERITY_SAMPLING_SEED,
            "train_severity_protocol": protocol,
            "train_severity_range": list(PROTOCOLS[protocol][1:]),
            "device": str(device)}


# --------------------------------------------------------------------------- #

def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def gen_sha():
    p = ROOT / "src" / "data" / "spectrum_generator.py"
    return hashlib.sha256(p.read_bytes()).hexdigest()


PLAN = (
    [("5a", "v7", s) for s in (42, 43, 44, 45, 46)]
    + [("5b", "v7", s) for s in (42, 43, 44, 45, 46)]
    + [("fusion", "v7", s) for s in (42, 43, 44, 45, 46)]
    + [("cnn", "v7", s) for s in (42, 43, 44, 45, 46)]
    + [("5a", "fixed1", s) for s in (42, 43, 44)]
    + [("5a", "low", s) for s in (42, 43, 44)]
    + [("fusion", "fixed1", s) for s in (42, 43, 44)]
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, default=NPZ)
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--timing-only", action="store_true",
                    help="one-epoch timing run per architecture, print the plan estimate, exit")
    ap.add_argument("--only", type=str, nargs="+", default=None,
                    help="restrict to entries 'arch:protocol' e.g. fusion:v7")
    ap.add_argument("--stopping", type=str, default="v7", choices=tuple(STOPPING),
                    help="stopping rule; 'extended' is the primary v8 protocol")
    ap.add_argument("--tag-suffix", type=str, default="")
    ap.add_argument("--cap", type=int, default=None,
                    help="override the stopping rule's max_epochs")
    ap.add_argument("--runs", type=str, nargs="+", default=None,
                    help="explicit 'arch:protocol:seed' triples")
    ap.add_argument("--train-tilt", type=float, default=0.0,
                    help="A18b: random energy tilt amplitude applied to training spectra")
    args = ap.parse_args()
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    args.outdir.mkdir(parents=True, exist_ok=True)
    ckpt = args.outdir / "checkpoints"

    plan = PLAN
    if args.only:
        keep = {tuple(o.split(":")) for o in args.only}
        plan = [p for p in plan if (p[0], p[1]) in keep]
    if args.runs:
        plan = [(a, pr, int(sd)) for a, pr, sd in (r.split(":") for r in args.runs)]
    if args.cap:
        STOPPING[args.stopping]["max_epochs"] = int(args.cap)

    protocols = sorted({p[1] for p in plan})
    print(f"device={device}  runs={len(plan)}  protocols={protocols}")

    if args.timing_only:
        sp = prepare(args.npz, "v7")
        print(f"\nparameter counts (target: 5a +/-20% = "
              f"{0.8*687843:.0f}-{1.2*687843:.0f}):")
        est = {}
        for arch in ("5a", "5b", "fusion", "cnn"):
            m = build_model(arch, 42)
            print(f"  {arch:7s} {count_trainable_parameters(m):>9,d}")
            r = train_one(arch, "v7", 42, sp, device, ckpt / "_timing", max_epochs=1,
                          log_every=False, stopping=args.stopping)
            est[arch] = r["mean_epoch_sec"]
            print(f"          one epoch: {r['mean_epoch_sec']:.1f} s")
        cap = STOPPING[args.stopping]["max_epochs"]
        total = sum(est[a] * cap for a, _, _ in plan)
        print(f"\nplan: {len(plan)} runs, stopping='{args.stopping}', worst case "
              f"{cap} epochs/run")
        print(f"projected worst-case training wall time: {total/60:.0f} min "
              f"({total/3600:.2f} h)")
        print(f"plus {len(protocols)} x train-set materialization")
        return

    materialized = {}
    metas = []
    done = set()
    _csv = args.outdir / "training_runs.csv"
    if _csv.exists():
        _prev = pd.read_csv(_csv)
        done = {(r.arch, r.protocol, int(r.seed)) for r in _prev.itertuples()}
        metas = _prev.to_dict("records")
    for arch, protocol, seed in plan:
        if (arch, protocol, seed) in done:
            print(f"  skip {arch}/{protocol}/seed{seed} (done)", flush=True)
            continue
        if protocol not in materialized:
            t = time.time()
            materialized[protocol] = prepare(args.npz, protocol)
            print(f"materialized protocol={protocol} in {time.time()-t:.0f}s "
                  f"(severity median "
                  f"{float(np.median(materialized[protocol].train_sev)):.3f})", flush=True)
        print(f"  == {arch} / {protocol} / seed {seed} ==", flush=True)
        meta = train_one(arch, protocol, seed, materialized[protocol], device, ckpt,
                         stopping=args.stopping, tag_suffix=args.tag_suffix,
                         train_tilt=args.train_tilt)
        meta["cap_override"] = args.cap
        meta.update(git_commit=git_sha(), generator_file_sha256=gen_sha(),
                    dataset=str(Path(args.npz).resolve().relative_to(ROOT)),
                    date_utc=datetime.now(timezone.utc).isoformat(),
                    torch=torch.__version__)
        (ckpt / f"{arch}_{protocol}_seed{seed}{args.tag_suffix}_run_meta.json").write_text(
            json.dumps(meta, indent=2))
        metas.append(meta)
        print(f"     -> best val MAE_logM {meta['best_val_MAE_logM']:.4f} "
              f"@ epoch {meta['best_epoch']} / {meta['epochs_trained']} "
              f"({meta['stop_reason']}), {meta['wall_time_sec']:.0f}s", flush=True)
        pd.DataFrame(metas).to_csv(args.outdir / "training_runs.csv", index=False)

    df = pd.DataFrame(metas)
    agg = (df.groupby(["arch", "protocol"])["best_val_MAE_logM"]
             .agg(["mean", "std", "count"]).reset_index())
    agg["dataset_version"] = DS.VERSION
    agg.to_csv(args.outdir / "training_summary.csv", index=False)
    print("\n" + agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
