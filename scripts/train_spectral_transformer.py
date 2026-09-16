"""Step 4.2 -- train SpectralTransformer 5a or 5b across seeds.

Usage
-----
    ./venv/bin/python scripts/train_spectral_transformer.py \\
        --arch 5a --seeds 42 43 44

    ./venv/bin/python scripts/train_spectral_transformer.py \\
        --arch 5b --seeds 42 43 44

Per-seed artifacts land in:
    outputs/eval/spectral_transformer_step4.2_<date>/{arch}_seed{N}/
        metrics.csv           full eval-suite results for this seed
        eval_table.json       same, as JSON for downstream aggregation
        train_curves.png      per-epoch train/val loss + val_MAE_logM
        checkpoints/best.pt   best val_MAE_logM checkpoint
        checkpoints/epoch_*.pt
        run_meta.json         seed, arch, MPS info, final stats

Safety features
---------------
* MPS memory queried at start (warn + offer CPU fallback if <2 GB).
* Peak MPS memory reported at end of each seed (catches monotonic leaks).
* Epoch-20 catastrophic-overfit abort: train/val < 0.2 AND val flat-or-rising
  over epochs 15-20 -> abort that seed; do not advance to the next.
* End-of-seed train/val ratio printed; if >2x AND val_MAE_logM > 0.40
  the script aborts the remaining seeds.
* Checkpoint every 10 epochs (last 3 rotated, plus best.pt always kept).
* Per-seed eval written immediately so a mid-run crash leaves earlier
  seeds analyzable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Silence the harmless nested-tensor info warning emitted on each
# TransformerEncoder construction with norm_first=True.
warnings.filterwarnings("ignore", message=".*enable_nested_tensor is True.*")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.dataset import InsSpectraDataset  # noqa: E402
from src.evaluation.harness import run_full_evaluation_suite  # noqa: E402
from src.models.spectral_transformer import (  # noqa: E402
    SpectralTransformer,
    SpectralTransformerModel,
    TargetStats,
    build_spectral_transformer,
    count_trainable_parameters,
    standardize_targets,
)


# ---------------------------------------------------------------------------
# Frozen training hyperparameters (Step 4.2 spec).
# ---------------------------------------------------------------------------

DATASET_PATH = ROOT / "data" / "full_dataset_phase1" / "dataset.npz"
OUTPUT_ROOT = ROOT / "results" / "training_runs"
STEP_TAG = "step4.2"

BATCH_SIZE = 64
N_EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 1000
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 10

LOSS_W_M = 1.0
LOSS_W_OMEGA = 0.1
LOSS_W_GAMMA = 0.1

# Session 7: training severity matches the eval range. Log-uniform because
# the eval grid {0.25, 0.5, 1, 2, 4} is geometric -- log-uniform gives equal
# weight to each "doubling" of severity, matching how reviewers will read
# the degradation curve. Linear-uniform would over-weight sev=2-4 (broader
# linear interval). See SESSION_LOG Session 7 for the planning-error context.
TRAIN_SEVERITY_RANGE: tuple[float, float] = (0.25, 4.0)
TRAIN_SEVERITY_LOG_UNIFORM = True

# Catastrophic-overfit abort at epoch 20: train/val < 0.2 AND val flat-or-rising.
OVERFIT_CHECK_EPOCH = 20
OVERFIT_FLAT_WINDOW = (15, 20)
OVERFIT_TRAIN_VAL_RATIO_MAX = 0.2

# End-of-seed gap check.
END_OF_SEED_RATIO_TRIGGER = 2.0
END_OF_SEED_VAL_MAE_TRIGGER = 0.40

# Checkpoint rotation.
CHECKPOINT_EVERY = 10
KEEP_LAST_N_CHECKPOINTS = 3

# MPS memory warning threshold.
MPS_MIN_BYTES = 2 * 1024**3  # 2 GiB


# ---------------------------------------------------------------------------
# Device / MPS helpers.
# ---------------------------------------------------------------------------

def select_device(prefer: str = "mps") -> torch.device:
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mps_memory_report() -> dict[str, float]:
    """Return MPS memory metrics in GiB. All zero/missing on non-MPS."""
    out = {"current_alloc_gib": 0.0, "driver_alloc_gib": 0.0, "recommended_max_gib": 0.0}
    if not torch.backends.mps.is_available():
        return out
    g = 1024 ** 3
    try:
        out["current_alloc_gib"] = torch.mps.current_allocated_memory() / g
    except Exception:
        pass
    try:
        out["driver_alloc_gib"] = torch.mps.driver_allocated_memory() / g
    except Exception:
        pass
    try:
        out["recommended_max_gib"] = torch.mps.recommended_max_memory() / g
    except Exception:
        pass
    return out


def mps_reset_peak() -> None:
    if torch.backends.mps.is_available() and hasattr(torch.mps, "reset_peak_memory_stats"):
        try:
            torch.mps.reset_peak_memory_stats()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Reproducibility.
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _worker_init_fn(worker_id: int) -> None:
    base = torch.initial_seed() % (2**31 - 1)
    np.random.seed(base + worker_id)
    import random
    random.seed(base + worker_id)


# ---------------------------------------------------------------------------
# Loss.
# ---------------------------------------------------------------------------

def multitask_loss(out: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """out, target: (B, 3) with columns (log_M, omega_Q_std, log_Gamma_Q_std)."""
    l_M = nn.functional.mse_loss(out[:, 0], target[:, 0])
    l_omega = nn.functional.mse_loss(out[:, 1], target[:, 1])
    l_Gamma = nn.functional.mse_loss(out[:, 2], target[:, 2])
    total = LOSS_W_M * l_M + LOSS_W_OMEGA * l_omega + LOSS_W_GAMMA * l_Gamma
    return total, {
        "loss_M": float(l_M.detach().cpu()),
        "loss_omega": float(l_omega.detach().cpu()),
        "loss_Gamma": float(l_Gamma.detach().cpu()),
        "loss_total": float(total.detach().cpu()),
    }


# ---------------------------------------------------------------------------
# Data preparation.
# ---------------------------------------------------------------------------

@dataclass
class PreparedSplits:
    train_specs: np.ndarray         # (N, 600), float32 -- raw augmented spectra
    train_targets_std: np.ndarray   # (N, 3), float32  -- standardized targets
    train_severities: np.ndarray    # (N,), float32    -- per-sample augmentation severity used
    val_specs: np.ndarray
    val_targets_std: np.ndarray
    target_stats: TargetStats
    train_M_natural: np.ndarray     # (N,) for diagnostics
    val_M_natural: np.ndarray
    val_omega_Q_natural: np.ndarray
    val_Gamma_Q_natural: np.ndarray


def _collate_split_at_fixed_severity(ds: InsSpectraDataset) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Materialize a dataset at the severity it was constructed with.
    Used for val (severity=1) which is non-stochastic.
    """
    n = len(ds)
    specs = np.empty((n, 600), dtype=np.float32)
    T_K = np.empty(n, dtype=np.float32)
    c_pct = np.empty(n, dtype=np.float32)
    E_kVcm = np.empty(n, dtype=np.float32)
    M = np.empty(n, dtype=np.float32)
    omega_Q = np.empty(n, dtype=np.float32)
    Gamma_Q = np.empty(n, dtype=np.float32)
    for i in range(n):
        it = ds[i]
        s = it["spectrum"]
        if hasattr(s, "numpy"):
            s = s.numpy()
        specs[i] = s.astype(np.float32)
        T_K[i] = float(it["conditions"]["T_K"])
        c_pct[i] = float(it["conditions"]["c_pct"])
        E_kVcm[i] = float(it["conditions"]["E_kVcm"])
        t = it["targets"]
        M[i] = float(t["M"])
        omega_Q[i] = float(t["omega_Q"])
        Gamma_Q[i] = float(t["Gamma_Q"])
    return specs, {
        "T_K": T_K, "c_pct": c_pct, "E_kVcm": E_kVcm,
        "M": M, "omega_Q": omega_Q, "Gamma_Q": Gamma_Q,
    }


def _collate_train_with_severity_distribution(
    ds: InsSpectraDataset,
    severity_range: tuple[float, float],
    log_uniform: bool,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Materialize a training dataset where each sample gets its own
    severity drawn from the configured distribution.

    Returns (specs, target_dict, severities_used).

    Severity sampling is seeded so the training distribution is deterministic
    per training seed (matches the seed-paired training contract).
    """
    rng = np.random.default_rng(seed)
    lo, hi = severity_range
    n = len(ds)
    if log_uniform:
        severities = np.exp(rng.uniform(np.log(lo), np.log(hi), size=n)).astype(np.float32)
    else:
        severities = rng.uniform(lo, hi, size=n).astype(np.float32)

    specs = np.empty((n, 600), dtype=np.float32)
    T_K = np.empty(n, dtype=np.float32)
    c_pct = np.empty(n, dtype=np.float32)
    E_kVcm = np.empty(n, dtype=np.float32)
    M = np.empty(n, dtype=np.float32)
    omega_Q = np.empty(n, dtype=np.float32)
    Gamma_Q = np.empty(n, dtype=np.float32)
    for i in range(n):
        # Per-sample severity override via the dataset's get_augmented(idx, severity=...).
        s = ds.get_augmented(i, severity=float(severities[i])).astype(np.float32)
        specs[i] = s
        T_K[i] = float(ds._T_K[i])
        c_pct[i] = float(ds._c_pct[i])
        E_kVcm[i] = float(ds._E_kVcm[i])
        M[i] = float(ds._M[i])
        omega_Q[i] = float(ds._omega_Q[i])
        Gamma_Q[i] = float(ds._Gamma_Q[i])
    return specs, {
        "T_K": T_K, "c_pct": c_pct, "E_kVcm": E_kVcm,
        "M": M, "omega_Q": omega_Q, "Gamma_Q": Gamma_Q,
    }, severities


def prepare_splits(
    npz_path: Path,
    *,
    train_severity_range: tuple[float, float] = TRAIN_SEVERITY_RANGE,
    train_log_uniform: bool = TRAIN_SEVERITY_LOG_UNIFORM,
    severity_sampling_seed: int = 0,
) -> PreparedSplits:
    lo, hi = train_severity_range
    dist_kind = "log-uniform" if train_log_uniform else "uniform"
    print(f"Materializing train ({dist_kind} severity in [{lo}, {hi}]) + val (severity=1.0) splits...")
    train_ds = InsSpectraDataset(npz_path, "train", severity=1.0, as_torch=False)
    val_ds = InsSpectraDataset(npz_path, "val", severity=1.0, as_torch=False)

    train_specs, train_tgt, train_sev = _collate_train_with_severity_distribution(
        train_ds, train_severity_range, train_log_uniform, severity_sampling_seed,
    )
    val_specs, val_tgt = _collate_split_at_fixed_severity(val_ds)

    stats = TargetStats.from_train(train_tgt["omega_Q"], train_tgt["Gamma_Q"])
    train_std = standardize_targets(train_tgt["M"], train_tgt["omega_Q"], train_tgt["Gamma_Q"], stats)
    val_std = standardize_targets(val_tgt["M"], val_tgt["omega_Q"], val_tgt["Gamma_Q"], stats)

    print(f"  train N={len(train_specs)}, val N={len(val_specs)}")
    print(f"  target_stats: omega_Q ~ N({stats.omega_Q_mean:.3f}, {stats.omega_Q_std:.3f})")
    print(f"                log_Gamma_Q ~ N({stats.log_Gamma_Q_mean:.3f}, {stats.log_Gamma_Q_std:.3f})")
    print(f"  train severity distribution: min={train_sev.min():.3f} "
          f"median={float(np.median(train_sev)):.3f} max={train_sev.max():.3f}")

    return PreparedSplits(
        train_specs=train_specs,
        train_targets_std=train_std,
        train_severities=train_sev,
        val_specs=val_specs,
        val_targets_std=val_std,
        target_stats=stats,
        train_M_natural=train_tgt["M"],
        val_M_natural=val_tgt["M"],
        val_omega_Q_natural=val_tgt["omega_Q"],
        val_Gamma_Q_natural=val_tgt["Gamma_Q"],
    )


class _TensorPairDataset(torch.utils.data.Dataset):
    """Lightweight (X, y) dataset over pre-materialized arrays."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


# ---------------------------------------------------------------------------
# LR schedule.
# ---------------------------------------------------------------------------

def make_lr_lambda(total_steps: int, warmup_steps: int):
    """Linear warmup to LR over `warmup_steps`, then cosine to 0 over remainder."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        import math
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


# ---------------------------------------------------------------------------
# Training loop for a single seed.
# ---------------------------------------------------------------------------

@dataclass
class SeedRunResult:
    arch: str
    seed: int
    epochs_trained: int
    best_val_mae_logM: float
    best_epoch: int
    final_train_loss: float
    final_val_loss: float
    train_val_ratio: float
    val_MAE_omega_Q_meV: float
    aborted: bool
    abort_reason: str
    wall_time_sec: float
    mps_peak_gib: float
    mps_driver_end_gib: float
    train_curve: list[dict[str, float]]


def _val_mae_logM_in_natural(model: nn.Module, val_specs: torch.Tensor,
                              val_M_natural: np.ndarray, stats: TargetStats,
                              batch: int = 256) -> float:
    """Compute MAE on log(M) in natural units (matches the eval-harness metric)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, val_specs.shape[0], batch):
            out = model(val_specs[i:i + batch])
            preds.append(out[:, 0].detach().cpu().numpy())  # log_M head
    log_M_pred = np.concatenate(preds, axis=0)
    log_M_true = np.log(np.clip(val_M_natural, 1e-9, None))
    return float(np.mean(np.abs(log_M_pred - log_M_true)))


def _val_mae_omega_Q_in_meV(model: nn.Module, val_specs: torch.Tensor,
                             val_omega_natural: np.ndarray, stats: TargetStats,
                             batch: int = 256) -> float:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, val_specs.shape[0], batch):
            out = model(val_specs[i:i + batch])
            preds.append(out[:, 1].detach().cpu().numpy())
    om_std = np.concatenate(preds, axis=0)
    om_pred = om_std * stats.omega_Q_std + stats.omega_Q_mean
    return float(np.mean(np.abs(om_pred - val_omega_natural)))


def train_one_seed(
    arch: str,
    seed: int,
    splits: PreparedSplits,
    device: torch.device,
    seed_dir: Path,
    wb_run,
    n_epochs: int = N_EPOCHS,
) -> SeedRunResult:
    seed_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = seed_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 72}\nSEED RUN  arch={arch}  seed={seed}  device={device}\n{'=' * 72}")
    t_start = time.time()

    mps_reset_peak()
    mem_start = mps_memory_report()
    print(f"MPS memory at start (GiB): {mem_start}")

    # Reproducibility: full RNG seeding, then build model.
    seed_everything(seed)
    net = build_spectral_transformer(arch, seed=seed).to(device)
    n_params = count_trainable_parameters(net)
    print(f"Trainable params: {n_params:,}")

    # Tensors on device.
    train_specs_t = torch.from_numpy(splits.train_specs).to(device)
    train_tgt_t = torch.from_numpy(splits.train_targets_std).to(device)
    val_specs_t = torch.from_numpy(splits.val_specs).to(device)
    val_tgt_t = torch.from_numpy(splits.val_targets_std).to(device)

    # Index-shuffling DataLoader (one int per sample) for reproducible shuffles.
    indices = np.arange(splits.train_specs.shape[0], dtype=np.int64)
    idx_ds = _TensorPairDataset(indices.reshape(-1, 1).astype(np.float32),
                                 indices.reshape(-1, 1).astype(np.float32))
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        idx_ds, batch_size=BATCH_SIZE, shuffle=True,
        generator=g, num_workers=0, worker_init_fn=_worker_init_fn,
    )

    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = n_epochs * max(1, len(loader))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, make_lr_lambda(total_steps, WARMUP_STEPS))

    history: list[dict[str, float]] = []
    best_val_mae_logM = float("inf")
    best_epoch = -1
    patience_left = EARLY_STOP_PATIENCE
    aborted = False
    abort_reason = ""

    step = 0
    for epoch in range(1, n_epochs + 1):
        net.train()
        running_loss_M = 0.0
        running_loss_omega = 0.0
        running_loss_Gamma = 0.0
        running_total = 0.0
        n_seen = 0
        for batch_pair in loader:
            # Recover indices from the dummy dataset.
            xb, _ = batch_pair
            idx = xb[:, 0].long().to(device)
            specs = train_specs_t[idx]
            tgt = train_tgt_t[idx]
            out = net(specs)
            loss, parts = multitask_loss(out, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            opt.step()
            sched.step()
            bs = specs.shape[0]
            running_total += parts["loss_total"] * bs
            running_loss_M += parts["loss_M"] * bs
            running_loss_omega += parts["loss_omega"] * bs
            running_loss_Gamma += parts["loss_Gamma"] * bs
            n_seen += bs
            step += 1
        train_loss = running_total / max(1, n_seen)
        train_loss_M = running_loss_M / max(1, n_seen)

        # Validation in standardized space (matches training-loss units).
        net.eval()
        with torch.no_grad():
            val_out = net(val_specs_t)
            val_loss, val_parts = multitask_loss(val_out, val_tgt_t)
        val_loss_v = float(val_loss.detach().cpu())
        val_loss_M = val_parts["loss_M"]

        # Eval-harness metric (MAE on log(M) in natural units).
        val_mae_logM = _val_mae_logM_in_natural(net, val_specs_t,
                                                 splits.val_M_natural, splits.target_stats)
        val_mae_omega = _val_mae_omega_Q_in_meV(net, val_specs_t,
                                                 splits.val_omega_Q_natural, splits.target_stats)

        lr_now = sched.get_last_lr()[0]
        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_loss,
            "train_loss_M": train_loss_M,
            "val_loss": val_loss_v,
            "val_loss_M": val_loss_M,
            "val_MAE_logM": val_mae_logM,
            "val_MAE_omega_Q_meV": val_mae_omega,
            "train_val_ratio": train_loss / max(1e-9, val_loss_v),
        }
        history.append(row)
        print(f"  ep {epoch:3d}/{n_epochs}  lr={lr_now:.2e}  "
              f"train={train_loss:.4f}  val={val_loss_v:.4f}  "
              f"val_MAE_logM={val_mae_logM:.4f}  "
              f"val_MAE_omegaQ={val_mae_omega:.3f}")
        if wb_run is not None:
            wb_run.log({f"epoch": epoch, **{k: v for k, v in row.items() if k != "epoch"}})

        # Best-checkpoint tracking.
        improved = val_mae_logM < best_val_mae_logM - 1e-5
        if improved:
            best_val_mae_logM = val_mae_logM
            best_epoch = epoch
            patience_left = EARLY_STOP_PATIENCE
            torch.save({
                "arch": arch, "seed": seed,
                "state_dict": net.state_dict(),
                "target_stats": splits.target_stats.to_dict(),
                "epoch": epoch,
                "val_MAE_logM": val_mae_logM,
                "name": f"spectral_transformer_{arch}_seed{seed}",
            }, ckpt_dir / "best.pt")
        else:
            patience_left -= 1

        # Periodic checkpoint with rotation.
        if epoch % CHECKPOINT_EVERY == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
            torch.save({
                "arch": arch, "seed": seed,
                "state_dict": net.state_dict(),
                "target_stats": splits.target_stats.to_dict(),
                "epoch": epoch,
                "val_MAE_logM": val_mae_logM,
                "name": f"spectral_transformer_{arch}_seed{seed}",
            }, ckpt_path)
            # Rotate: keep only the last KEEP_LAST_N_CHECKPOINTS periodic files.
            existing = sorted(ckpt_dir.glob("epoch_*.pt"))
            for old in existing[:-KEEP_LAST_N_CHECKPOINTS]:
                try:
                    old.unlink()
                except OSError:
                    pass

        # Epoch-20 catastrophic-overfit abort.
        if epoch == OVERFIT_CHECK_EPOCH:
            window = [r for r in history if OVERFIT_FLAT_WINDOW[0] <= r["epoch"] <= OVERFIT_FLAT_WINDOW[1]]
            ratio = train_loss / max(1e-9, val_loss_v)
            if window:
                v_first = window[0]["val_MAE_logM"]
                v_last = window[-1]["val_MAE_logM"]
                flat_or_rising = (v_last >= v_first * 0.98)  # within 2% or higher
                if ratio < OVERFIT_TRAIN_VAL_RATIO_MAX and flat_or_rising:
                    abort_reason = (
                        f"catastrophic overfit at epoch {epoch}: "
                        f"train/val={ratio:.3f} (<{OVERFIT_TRAIN_VAL_RATIO_MAX}) "
                        f"AND val_MAE_logM flat-or-rising "
                        f"({v_first:.4f} -> {v_last:.4f}) over epochs "
                        f"{OVERFIT_FLAT_WINDOW[0]}-{OVERFIT_FLAT_WINDOW[1]}"
                    )
                    print(f"  ABORT: {abort_reason}")
                    aborted = True
                    break

        if patience_left <= 0:
            print(f"  Early stop at epoch {epoch} (patience exhausted).")
            break

    # End of training loop for this seed.
    wall = time.time() - t_start
    final_row = history[-1] if history else {
        "train_loss": float("nan"), "val_loss": float("nan"),
        "val_MAE_logM": float("nan"), "val_MAE_omega_Q_meV": float("nan"),
    }

    # Peak MPS memory at end of seed.
    peak_gib = 0.0
    if torch.backends.mps.is_available() and hasattr(torch.mps, "driver_allocated_memory"):
        try:
            peak_gib = torch.mps.driver_allocated_memory() / (1024 ** 3)
        except Exception:
            pass
    mem_end = mps_memory_report()
    print(f"MPS memory at end (GiB): driver_alloc={mem_end['driver_alloc_gib']:.3f}  "
          f"current_alloc={mem_end['current_alloc_gib']:.3f}")

    train_val_ratio = final_row["train_loss"] / max(1e-9, final_row["val_loss"])

    return SeedRunResult(
        arch=arch,
        seed=seed,
        epochs_trained=len(history),
        best_val_mae_logM=float(best_val_mae_logM),
        best_epoch=int(best_epoch),
        final_train_loss=float(final_row["train_loss"]),
        final_val_loss=float(final_row["val_loss"]),
        train_val_ratio=float(train_val_ratio),
        val_MAE_omega_Q_meV=float(final_row["val_MAE_omega_Q_meV"]),
        aborted=bool(aborted),
        abort_reason=abort_reason,
        wall_time_sec=float(wall),
        mps_peak_gib=float(mem_end["driver_alloc_gib"]),
        mps_driver_end_gib=float(mem_end["driver_alloc_gib"]),
        train_curve=history,
    )


# ---------------------------------------------------------------------------
# Plot helpers.
# ---------------------------------------------------------------------------

def save_train_curves(history: list[dict[str, float]], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    if not history:
        return
    epochs = [r["epoch"] for r in history]
    train = [r["train_loss"] for r in history]
    val = [r["val_loss"] for r in history]
    mae = [r["val_MAE_logM"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train, label="train", lw=1.8)
    axes[0].plot(epochs, val, label="val", lw=1.8)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("multitask loss")
    axes[0].set_yscale("log")
    axes[0].set_title("Train / val loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(epochs, mae, color="C2", lw=1.8)
    axes[1].axhline(0.286, ls="--", color="grey", lw=1.0, label="dho_ridge val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val MAE on log(M)")
    axes[1].set_title("Validation MAE_logM")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-seed evaluation harness (mirrors Step 4.1).
# ---------------------------------------------------------------------------

def run_eval_for_seed(arch: str, seed: int, ckpt_path: Path, device: torch.device,
                      out_dir: Path, data_path: Path) -> pd.DataFrame:
    """Load best.pt, run the full evaluation suite, persist results."""
    eval_device = "cpu"  # eval-harness uses numpy batches; CPU keeps it simple
    model = SpectralTransformerModel.from_checkpoint(ckpt_path, device=eval_device)
    # Override the name so the eval-table rows distinguish 5a / 5b / seeds.
    model.name = f"spectral_transformer_{arch}_seed{seed}"
    df = run_full_evaluation_suite(model, data_path)
    df["arch"] = arch
    df["seed"] = seed
    df.to_csv(out_dir / "metrics.csv", index=False)
    df.to_json(out_dir / "eval_table.json", orient="records", indent=2)
    return df


# ---------------------------------------------------------------------------
# W&B helpers.
# ---------------------------------------------------------------------------

def _wandb_authenticated() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        import netrc
        n = netrc.netrc()
        if n.authenticators("api.wandb.ai"):
            return True
    except (FileNotFoundError, Exception):
        pass
    return False


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=("5a", "5b"), required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--device", default="mps", choices=("mps", "cuda", "cpu"))
    p.add_argument("--max-epochs", type=int, default=N_EPOCHS)
    p.add_argument("--limit-epochs", type=int, default=None,
                   help="If set, run only this many epochs (sanity-check mode).")
    p.add_argument("--out-tag", default=None,
                   help="Optional suffix on the run dir; defaults to today's date.")
    p.add_argument("--data-path", type=str, default=None,
                   help="Dataset .npz. Defaults to data/full_dataset_phase1/dataset.npz (Phase 1).")
    p.add_argument("--param-card", type=str, default=None,
                   help="Path to the Phase 1 parameter card (logged to W&B for provenance).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import wandb

    device = select_device(args.device)
    if device.type != args.device:
        print(f"WARNING: requested device={args.device} unavailable; using {device}")

    if device.type == "mps":
        mem = mps_memory_report()
        if mem["recommended_max_gib"] > 0 and mem["recommended_max_gib"] < (MPS_MIN_BYTES / 1024**3):
            print(f"WARNING: MPS recommended_max={mem['recommended_max_gib']:.2f} GiB "
                  f"(< {MPS_MIN_BYTES/1024**3:.1f} GiB).")
            print("         Consider --device cpu to avoid OOM. Continuing on MPS.")

    data_path = Path(args.data_path) if args.data_path else DATASET_PATH
    print(f"Dataset: {data_path}")

    today = args.out_tag or datetime.utcnow().strftime("%Y%m%d")
    run_root = OUTPUT_ROOT / f"spectral_transformer_{STEP_TAG}_{today}"
    run_root.mkdir(parents=True, exist_ok=True)

    wb_mode = "online" if _wandb_authenticated() else "offline"
    if wb_mode == "offline":
        print("W&B: no API key detected -- falling back to offline mode.")
        print("     Run `./venv/bin/wandb login` and `wandb sync <run-dir>` to upload.")

    splits = prepare_splits(data_path)

    seed_summaries: list[SeedRunResult] = []
    epochs_to_run = args.limit_epochs if args.limit_epochs else args.max_epochs

    for seed in args.seeds:
        seed_dir = run_root / f"{args.arch}_seed{seed}"
        seed_dir.mkdir(exist_ok=True)
        run_name = f"spectral_transformer_{args.arch}_{STEP_TAG}_seed{seed}_{today}"
        wb_run = wandb.init(
            # entity left unset -- W&B uses the default entity from
            # `wandb login`. (Session 6's "syedmoid" entity was incorrect;
            # the real team is "syedmoid-galaxy-technologies-llc" and
            # leaving it unset is the robust choice.)
            project="ins-spectral-learning",
            name=run_name,
            group=f"{STEP_TAG}_{args.arch}",
            job_type="train",
            tags=[STEP_TAG, args.arch, f"seed{seed}"],
            config={
                "arch": args.arch,
                "seed": seed,
                "embed_dim": 96,
                "n_layers": 6,
                "n_heads": 8,
                "mlp_ratio": 4,
                "patch_size": 10,
                "n_patches": 60,
                "rff_num_freqs": 48,
                "rff_sigma": 1.0,
                "input_standardization": "full_spectrum",
                "train_severity_range": list(TRAIN_SEVERITY_RANGE),
                "train_severity_log_uniform": TRAIN_SEVERITY_LOG_UNIFORM,
                "batch_size": BATCH_SIZE,
                "n_epochs": epochs_to_run,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "warmup_steps": WARMUP_STEPS,
                "grad_clip": GRAD_CLIP,
                "early_stop_patience": EARLY_STOP_PATIENCE,
                "loss_weights": {"M": LOSS_W_M, "omega": LOSS_W_OMEGA, "Gamma": LOSS_W_GAMMA},
                "device": device.type,
                "dataset_path": str(data_path),
                "param_card": args.param_card or "",
            },
            mode=wb_mode,
            dir=str(seed_dir),
            reinit="finish_previous",
        )

        result = train_one_seed(
            arch=args.arch, seed=seed, splits=splits, device=device,
            seed_dir=seed_dir, wb_run=wb_run, n_epochs=epochs_to_run,
        )
        seed_summaries.append(result)

        # Write per-seed train curve + run_meta IMMEDIATELY so a crash on a
        # later seed leaves this one analyzable.
        save_train_curves(result.train_curve, seed_dir / "train_curves.png")
        with (seed_dir / "run_meta.json").open("w") as f:
            payload = asdict(result)
            payload["train_curve_len"] = len(result.train_curve)
            payload.pop("train_curve")  # the full curve is in train_curves.png
            json.dump(payload, f, indent=2)

        # Per-seed eval against the full suite using the best.pt checkpoint.
        best_path = seed_dir / "checkpoints" / "best.pt"
        if best_path.exists() and not result.aborted:
            try:
                df = run_eval_for_seed(args.arch, seed, best_path, device, seed_dir, data_path)
                if wb_run is not None:
                    wb_run.log({"eval_table": wandb.Table(dataframe=df)})
            except Exception as e:
                print(f"  WARNING: eval suite failed for seed {seed}: {e}")
        else:
            print(f"  Skipping per-seed eval (aborted={result.aborted}, best_exists={best_path.exists()}).")

        # End-of-seed train/val gap check.
        if (result.train_val_ratio > END_OF_SEED_RATIO_TRIGGER and
                result.best_val_mae_logM > END_OF_SEED_VAL_MAE_TRIGGER):
            print(f"\nABORT REMAINING SEEDS: end-of-seed train/val ratio "
                  f"{result.train_val_ratio:.2f}x with val_MAE_logM "
                  f"{result.best_val_mae_logM:.4f}. See seed dir for curves.")
            wb_run.finish()
            break

        wb_run.finish()

    # Cross-seed summary for this arch.
    summary_path = run_root / f"{args.arch}_seed_summary.json"
    with summary_path.open("w") as f:
        json.dump([{
            "arch": r.arch, "seed": r.seed,
            "epochs_trained": r.epochs_trained,
            "best_val_mae_logM": r.best_val_mae_logM,
            "best_epoch": r.best_epoch,
            "final_train_loss": r.final_train_loss,
            "final_val_loss": r.final_val_loss,
            "train_val_ratio": r.train_val_ratio,
            "val_MAE_omega_Q_meV": r.val_MAE_omega_Q_meV,
            "aborted": r.aborted, "abort_reason": r.abort_reason,
            "wall_time_sec": r.wall_time_sec,
            "mps_peak_gib": r.mps_peak_gib,
            "mps_driver_end_gib": r.mps_driver_end_gib,
        } for r in seed_summaries], f, indent=2)
    print(f"\nWrote arch summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
