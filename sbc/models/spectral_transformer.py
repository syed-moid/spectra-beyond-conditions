"""Spectral Transformer 5a / 5b for Step 4.2.

Two architectures share everything except the patch encoder:

    5a (learned linear patch projection)
        x -> nn.Linear(10, 96, bias=True)              # learned, 1056 params
             -> + learned pos_embed -> transformer

    5b (fixed Random Fourier Feature patch encoding)
        x -> per-patch standardize
             -> B @ x                                   # B is (48, 10) buffer, fixed
             -> [cos(2*pi y), sin(2*pi y)]              # 96-dim
             -> + learned pos_embed -> transformer

The headline ablation tests content representation, not positional
encoding (Design.md secondary claim: "frequency-domain attention
captures spectral physics"). Both arches use the same learned positional
embedding so position is the controlled variable.

Seed-paired init protocol (also documented in build_spectral_transformer
docstring, repeated here for navigation):

    At fixed `seed`, the SHARED components (transformer blocks, MLP head,
    learned positional embedding, CLS token, final LayerNorm) must be
    byte-identical between arch="5a" and arch="5b". Only the patch encoder
    differs.

    RNG sequence:
      1. torch.manual_seed(seed) — sets default generator.
      2. Construct SHARED modules in fixed order. Each module's init
         consumes RNG state from the default generator. Order:
           a. transformer blocks (6 x nn.TransformerEncoderLayer)
           b. final LayerNorm
           c. MLP head (96 -> 96 -> 3)
           d. CLS token (nn.Parameter, randn(1, 1, 96))
           e. learned positional embedding (nn.Embedding(61, 96))
         After Phase 1 the default generator's state is identical for
         5a and 5b builds at this seed.
      3. Construct ARCH-SPECIFIC patch encoder:
           5a: nn.Linear(10, 96) -- consumes the NEXT chunk of the
               default generator's stream.
           5b: B matrix initialized from a SEPARATE Generator seeded
               with `seed + RFF_BUFFER_SEED_OFFSET` (= seed + 10_000).
               Isolating B onto a separate generator makes the invariant
               "shared Phase-1 weights identical across 5a/5b at fixed
               seed" trivially true regardless of how 5a's patch_proj
               consumes the default stream.

Auxiliary head outputs are produced in *standardized* space; the
SpectralTransformerModel wrapper un-standardizes at inference using
target_stats stashed at fit time.

References
----------
* Tancik et al. 2020 "Fourier features let networks learn high frequency
  functions in low dimensional domains" — RFF formulation used here.
* Vaswani et al. 2017 — base Transformer encoder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    _TORCH_OK = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_OK = False

from .base import BaseModel


# ---------------------------------------------------------------------------
# Constants -- frozen Step 4.2 spec.
# ---------------------------------------------------------------------------

OMEGA_GRID_LEN = 600
PATCH_SIZE = 10
N_PATCHES = OMEGA_GRID_LEN // PATCH_SIZE          # 60
SEQ_LEN = 1 + N_PATCHES                            # 61 (CLS + patches)
EMBED_DIM = 96
N_LAYERS = 6
N_HEADS = 8
MLP_RATIO = 4
FFN_HIDDEN = EMBED_DIM * MLP_RATIO                 # 384
DROPOUT = 0.1

RFF_NUM_FREQS = 48                                 # output dim = 2 * 48 = 96
# Session 7: sigma dropped from 10 -> 1 after switching to full-spectrum
# standardization in SpectralTransformer.forward. With unit-variance input
# bins after full-spectrum standardization, a 10-bin patch has E[||x||^2] = 10
# so ||x|| ~ sqrt(10) ~ 3.2 for a generic patch and ~10 for a strong peak.
# sigma=1 keeps 2*pi*B@x within ~20-60 rad even on peak patches (a few cycles,
# the regime where Tancik features carry usable gradient). The original
# sigma=10 with per-patch standardization aliased catastrophically; see
# SESSION_LOG Session 7 for the diagnosis.
RFF_SIGMA = 1.0
RFF_BUFFER_SEED_OFFSET = 10_000

HEAD_OUT_DIM = 3                                   # (log_M, omega_Q_std, log_Gamma_Q_std)


# ---------------------------------------------------------------------------
# Patch encoders.
# ---------------------------------------------------------------------------

class LearnedPatchProjection(nn.Module):
    """5a: learnable linear projection of each 10-bin patch into 96-dim."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(PATCH_SIZE, EMBED_DIM, bias=True)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        # patches: (B, N_PATCHES, PATCH_SIZE)
        return self.proj(patches)


class RFFPatchEncoder(nn.Module):
    """5b: fixed Random Fourier Feature encoder.

    Session 7 redesign: the patch encoder is now a pure RFF transform with
    NO internal standardization. Full-spectrum standardization happens in
    SpectralTransformer.forward (applied uniformly to both 5a and 5b so the
    ablation differs ONLY in the patch encoder). This preserves relative
    amplitude variation BETWEEN patches within a spectrum (a peak patch has
    a larger norm than a tail patch), which is the spectral information M
    depends on.

    sigma=1 is now the default (was sigma=10 in the buggy v1). With unit-
    variance bins after full-spectrum standardization, a typical patch has
    ||x|| ~ sqrt(10) ~ 3.2 and a strong peak patch ~10. 2*pi*B@x at sigma=1
    has std ~ 2*pi*||x|| ~ 20-60 rad => a few cycles across the input
    range, which is the regime where Tancik features carry gradient.
    """

    def __init__(self, B: torch.Tensor):
        super().__init__()
        if B.shape != (RFF_NUM_FREQS, PATCH_SIZE):
            raise ValueError(f"RFF B must be ({RFF_NUM_FREQS}, {PATCH_SIZE}); got {tuple(B.shape)}")
        # Buffer => fixed at construction, included in state_dict, NOT in .parameters().
        self.register_buffer("rff_B", B)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        # patches: (B, N_PATCHES, PATCH_SIZE) -- already full-spectrum-standardized
        # by the SpectralTransformer.forward caller. No further normalization here.
        y = patches @ self.rff_B.T                                     # (B, N, 48)
        two_pi_y = (2.0 * math.pi) * y
        return torch.cat([torch.cos(two_pi_y), torch.sin(two_pi_y)], dim=-1)  # (B, N, 96)


# ---------------------------------------------------------------------------
# Transformer encoder body (shared between 5a/5b).
# ---------------------------------------------------------------------------

def _make_transformer_encoder() -> nn.TransformerEncoder:
    """Pre-LN Transformer encoder, deterministically initialized under the
    currently-active default generator."""
    layer = nn.TransformerEncoderLayer(
        d_model=EMBED_DIM,
        nhead=N_HEADS,
        dim_feedforward=FFN_HIDDEN,
        dropout=DROPOUT,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=N_LAYERS)


def _make_mlp_head() -> nn.Sequential:
    """96 -> 96 -> 3 with GELU. Outputs are in standardized space."""
    return nn.Sequential(
        nn.Linear(EMBED_DIM, EMBED_DIM),
        nn.GELU(),
        nn.Linear(EMBED_DIM, HEAD_OUT_DIM),
    )


class SpectralTransformer(nn.Module):
    """The shared encoder. `patch_encoder` is swapped between 5a and 5b.

    Forward contract:
      input  spectrum: (B, OMEGA_GRID_LEN) raw spectrum values
      output: (B, 3) head output in STANDARDIZED target space
              (log_M, omega_Q_std, log_Gamma_Q_std)

    Pipeline:
      1. Full-spectrum standardize the input (per-sample mu/sd over the
         600 bins) -- applies uniformly to BOTH 5a and 5b so the ablation
         only differs in the patch encoder.
      2. Reshape to patches (B, 60, 10) and run patch_encoder -> (B, 60, 96).
      3. Prepend CLS -> (B, 61, 96).
      4. Add learned positional embedding, run transformer + final LN, head
         on the CLS token.
    """

    def __init__(self, patch_encoder: nn.Module, *, arch: str):
        super().__init__()
        self.arch = arch
        self.patch_encoder = patch_encoder
        self.transformer: nn.TransformerEncoder  # assigned by builder
        self.final_ln: nn.LayerNorm
        self.head: nn.Sequential
        self.cls_token: nn.Parameter
        self.pos_embed: nn.Embedding

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        """spectrum: (B, OMEGA_GRID_LEN).
        Returns (B, 3) head output in STANDARDIZED target space."""
        B = spectrum.shape[0]
        # 1. Full-spectrum standardization, per-sample, over all 600 bins.
        mu = spectrum.mean(dim=-1, keepdim=True)
        sd = spectrum.std(dim=-1, keepdim=True).clamp_min(1e-6)
        x = (spectrum - mu) / sd
        # 2. Reshape into patches and encode.
        patches = x.reshape(B, N_PATCHES, PATCH_SIZE)
        patch_emb = self.patch_encoder(patches)                          # (B, 60, 96)
        # 3. Prepend CLS.
        cls = self.cls_token.expand(B, -1, -1)                           # (B, 1, 96)
        tokens = torch.cat([cls, patch_emb], dim=1)                      # (B, 61, 96)
        # 4. Position + transformer + head.
        pos_idx = torch.arange(SEQ_LEN, device=tokens.device)
        tokens = tokens + self.pos_embed(pos_idx)
        out = self.transformer(tokens)
        out = self.final_ln(out)
        return self.head(out[:, 0])                                      # (B, 3)


# ---------------------------------------------------------------------------
# Builder with seed-paired init.
# ---------------------------------------------------------------------------

def build_spectral_transformer(
    arch: str,
    seed: int,
    *,
    sigma: float = RFF_SIGMA,
) -> SpectralTransformer:
    """Build a 5a or 5b SpectralTransformer with seed-paired init.

    See module docstring for the seed-pairing protocol (Phase 1: shared
    modules under default generator; Phase 2: arch-specific patch encoder,
    with 5b's RFF buffer on a separate seeded generator at seed + 10_000).
    """
    if not _TORCH_OK:
        raise RuntimeError("PyTorch not available")
    if arch not in ("5a", "5b"):
        raise ValueError(f"arch must be '5a' or '5b'; got {arch!r}")

    torch.manual_seed(seed)
    # MPS RNG, if available; harmless on CPU/CUDA-only setups.
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # --- Phase 1: SHARED modules, fixed construction order. -----------
    # Construction order matters: every shared init draws from the default
    # generator in this exact sequence so 5a@seed == 5b@seed bit-for-bit.
    transformer = _make_transformer_encoder()
    final_ln = nn.LayerNorm(EMBED_DIM)
    head = _make_mlp_head()
    cls_token = nn.Parameter(torch.randn(1, 1, EMBED_DIM))
    pos_embed = nn.Embedding(SEQ_LEN, EMBED_DIM)

    # --- Phase 2: arch-specific patch encoder. ------------------------
    if arch == "5a":
        patch_encoder: nn.Module = LearnedPatchProjection()
    else:  # 5b
        rff_gen = torch.Generator()
        rff_gen.manual_seed(int(seed) + RFF_BUFFER_SEED_OFFSET)
        B = torch.randn(RFF_NUM_FREQS, PATCH_SIZE, generator=rff_gen) * sigma
        patch_encoder = RFFPatchEncoder(B)

    # Assemble.
    model = SpectralTransformer(patch_encoder, arch=arch)
    model.transformer = transformer
    model.final_ln = final_ln
    model.head = head
    model.register_parameter("cls_token", cls_token)
    model.add_module("pos_embed", pos_embed)
    return model


# ---------------------------------------------------------------------------
# Target standardization (auxiliary heads).
# ---------------------------------------------------------------------------

@dataclass
class TargetStats:
    """Mean/std for auxiliary-target standardization.

    log_M is left in natural units (already zero-ish-mean, unit-ish-variance
    from inspection). omega_Q and log_Gamma_Q are standardized so the 0.1
    auxiliary-loss weight is scale-invariant.
    """
    omega_Q_mean: float
    omega_Q_std: float
    log_Gamma_Q_mean: float
    log_Gamma_Q_std: float

    @classmethod
    def from_train(cls, omega_Q: np.ndarray, Gamma_Q: np.ndarray) -> "TargetStats":
        log_Gamma = np.log(np.clip(Gamma_Q, 1e-9, None))
        return cls(
            omega_Q_mean=float(np.mean(omega_Q)),
            omega_Q_std=float(np.std(omega_Q) + 1e-9),
            log_Gamma_Q_mean=float(np.mean(log_Gamma)),
            log_Gamma_Q_std=float(np.std(log_Gamma) + 1e-9),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "omega_Q_mean": self.omega_Q_mean,
            "omega_Q_std": self.omega_Q_std,
            "log_Gamma_Q_mean": self.log_Gamma_Q_mean,
            "log_Gamma_Q_std": self.log_Gamma_Q_std,
        }

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "TargetStats":
        return cls(**d)


def standardize_targets(M: np.ndarray, omega_Q: np.ndarray, Gamma_Q: np.ndarray,
                        stats: TargetStats) -> np.ndarray:
    """Stack (log_M, omega_Q_std, log_Gamma_Q_std) into a (N, 3) array."""
    log_M = np.log(np.clip(M, 1e-9, None))
    omega_std = (omega_Q - stats.omega_Q_mean) / stats.omega_Q_std
    log_Gamma_std = (np.log(np.clip(Gamma_Q, 1e-9, None)) - stats.log_Gamma_Q_mean) / stats.log_Gamma_Q_std
    return np.stack([log_M, omega_std, log_Gamma_std], axis=1).astype(np.float32)


def unstandardize_outputs(out_std: np.ndarray, stats: TargetStats) -> dict[str, np.ndarray]:
    """Invert standardization. Returns dict with M, omega_Q, Gamma_Q (in natural units)."""
    log_M = out_std[:, 0]
    omega_Q = out_std[:, 1] * stats.omega_Q_std + stats.omega_Q_mean
    log_Gamma_Q = out_std[:, 2] * stats.log_Gamma_Q_std + stats.log_Gamma_Q_mean
    return {
        "M": np.exp(log_M),
        "omega_Q": omega_Q,
        "Gamma_Q": np.exp(log_Gamma_Q),
    }


# ---------------------------------------------------------------------------
# BaseModel wrapper for evaluation harness.
# ---------------------------------------------------------------------------

class SpectralTransformerModel(BaseModel):
    """BaseModel adapter wrapping a trained nn.Module + target stats.

    The training loop lives in `scripts/train_spectral_transformer.py`.
    This wrapper is for *inference / eval* — it loads a checkpoint and
    exposes the BaseModel.predict_* interface so the evaluation harness
    can treat ST runs uniformly with the DHO baselines.

    Construction patterns:
      * `SpectralTransformerModel.from_checkpoint(path)` for eval.
      * Direct `SpectralTransformerModel(net, stats, name=...)` after
        training (training script uses this).
    """

    def __init__(self, net: SpectralTransformer, stats: TargetStats,
                 name: str, device: str = "cpu"):
        self.net = net.to(device).eval()
        self.stats = stats
        self.name = name
        self.device = device

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path, device: str = "cpu") -> "SpectralTransformerModel":
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        arch = ckpt["arch"]
        seed = int(ckpt["seed"])
        stats = TargetStats.from_dict(ckpt["target_stats"])
        net = build_spectral_transformer(arch, seed)
        net.load_state_dict(ckpt["state_dict"])
        name = ckpt.get("name", f"spectral_transformer_{arch}")
        return cls(net=net, stats=stats, name=name, device=device)

    def fit(self, train_ds, val_ds) -> None:
        # Training is done by scripts/train_spectral_transformer.py;
        # the wrapper is for inference. Raising makes it obvious if
        # someone tries to call .fit on the eval-side wrapper.
        raise RuntimeError(
            "SpectralTransformerModel is an inference wrapper. "
            "Use scripts/train_spectral_transformer.py to train, then "
            "construct via from_checkpoint()."
        )

    @torch.no_grad()
    def _forward_batch(self, batch: dict[str, Any]) -> np.ndarray:
        spec = batch["spectrum"]
        if hasattr(spec, "numpy"):
            spec = spec.numpy()
        spec_t = torch.as_tensor(np.asarray(spec, dtype=np.float32), device=self.device)
        out_std = self.net(spec_t).cpu().numpy()
        return out_std

    def predict_M(self, batch: dict[str, Any]) -> np.ndarray:
        out_std = self._forward_batch(batch)
        natural = unstandardize_outputs(out_std, self.stats)
        return natural["M"].astype(np.float64)

    def predict_omega_Q(self, batch: dict[str, Any]) -> np.ndarray:
        out_std = self._forward_batch(batch)
        natural = unstandardize_outputs(out_std, self.stats)
        return natural["omega_Q"].astype(np.float64)

    def predict_Gamma_Q(self, batch: dict[str, Any]) -> np.ndarray:
        out_std = self._forward_batch(batch)
        natural = unstandardize_outputs(out_std, self.stats)
        return natural["Gamma_Q"].astype(np.float64)


# ---------------------------------------------------------------------------
# Convenience: parameter counting (used by training script + tests).
# ---------------------------------------------------------------------------

def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def count_buffer_elements(model: nn.Module) -> int:
    return int(sum(b.numel() for b in model.buffers()))
