"""Load A4 (v8) checkpoints and expose a uniform prediction interface.

Every downstream task (A3b, A5, A6, A8, A9, A10, A14) needs "give me this
model's predictions on these spectra". Fusion additionally needs the condition
vector, so the batch dict carries T_K / c_pct / E_kVcm and spectrum-only models
simply ignore them.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from a4_train_v8 import build_model, uses_conditions  # noqa: E402
from sbc.models.nonlinear_conditions_mlp import normalize_conditions  # noqa: E402
from sbc.models.spectral_transformer import TargetStats, unstandardize_outputs  # noqa: E402

CKPT_DIR = ROOT / "results" / "revision" / "A4" / "checkpoints"


class A4Model:
    """Inference wrapper around one A4 checkpoint."""

    def __init__(self, net, stats: TargetStats, arch: str, protocol: str, seed: int,
                 device: str = "cpu"):
        self.net = net.to(device).eval()
        self.stats = stats
        self.arch = arch
        self.protocol = protocol
        self.seed = seed
        self.device = device
        self.name = f"{arch}_{protocol}_seed{seed}"

    @torch.no_grad()
    def _raw(self, batch: dict[str, Any], chunk: int = 2048) -> np.ndarray:
        spec = np.asarray(batch["spectrum"], dtype=np.float32)
        outs = []
        cond_np = None
        if uses_conditions(self.arch):
            cond_np = normalize_conditions(np.asarray(batch["T_K"], dtype=np.float32),
                                           np.asarray(batch["c_pct"], dtype=np.float32),
                                           np.asarray(batch["E_kVcm"], dtype=np.float32))
        for i in range(0, len(spec), chunk):
            x = torch.from_numpy(spec[i:i + chunk]).to(self.device)
            if cond_np is None:
                outs.append(self.net(x).cpu().numpy())
            else:
                c = torch.from_numpy(cond_np[i:i + chunk]).to(self.device)
                outs.append(self.net(x, c).cpu().numpy())
        return np.concatenate(outs, axis=0)

    def predict_all(self, batch) -> dict[str, np.ndarray]:
        return unstandardize_outputs(self._raw(batch), self.stats)

    def predict_M(self, batch) -> np.ndarray:
        return self.predict_all(batch)["M"].astype(np.float64)

    def predict_logM(self, batch) -> np.ndarray:
        return self._raw(batch)[:, 0].astype(np.float64)

    def predict_omega_Q(self, batch) -> np.ndarray:
        return self.predict_all(batch)["omega_Q"].astype(np.float64)

    def predict_Gamma_Q(self, batch) -> np.ndarray:
        return self.predict_all(batch)["Gamma_Q"].astype(np.float64)


def load_a4_model(arch: str, protocol: str, seed: int, device: str = "cpu",
                  ckpt_dir: Path = CKPT_DIR) -> A4Model:
    path = ckpt_dir / f"{arch}_{protocol}_seed{seed}.pt"
    ck = torch.load(path, map_location=device, weights_only=False)
    net = build_model(ck["arch"], int(ck["seed"]))
    net.load_state_dict(ck["state_dict"])
    return A4Model(net, TargetStats.from_dict(ck["target_stats"]),
                   ck["arch"], ck.get("protocol", protocol), int(ck["seed"]), device)


def available(ckpt_dir: Path = CKPT_DIR):
    out = []
    for p in sorted(ckpt_dir.glob("*.pt")):
        arch, protocol, seed = p.stem.rsplit("_", 2)[0], p.stem.rsplit("_", 2)[1], p.stem.rsplit("_", 2)[2]
        out.append((arch, protocol, int(seed.replace("seed", ""))))
    return out
