"""A7 - channel-ablation datasets, including the round-19 shared-alpha setting.

Four latent settings, dataset seeds {0, 1, 2}:

    xi1_only        beta*xi2 forced to 0 (no frequency channel)
    xi2_only        xi1 forced to 0 for every mode (no linewidth channel)
    both            the standard Phase 1 latent model
    alpha_per_mode  both channels, but the disorder amplitude alpha is drawn
                    INDEPENDENTLY per mode instead of once per spectrum

No generator source is modified. Each setting is produced by constructing the
LatentDraw directly: alpha and xi1 enter the multiplier only as the product
alpha*xi1[mode], so an independent per-mode amplitude is obtained by setting
alpha = 1 and folding a per-mode draw into xi1.
"""

from __future__ import annotations

import argparse, hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sbc.data.augmentations import SpectrumState, sample_aug_params  # noqa: E402
from sbc.data.latent_perturbations import (  # noqa: E402
    MODES, LatentDraw, sample_alpha, sample_beta_xi2,
)
from sbc.data.sampling import draw_samples  # noqa: E402
import dataset_paths as DS  # noqa: E402
from sbc.data.spectrum_generator import DEFAULT_OMEGA_GRID, generate_spectrum  # noqa: E402

CFG = ROOT / "configs" / "augmentation_realistic.yaml"
DATA = ROOT / "data"
SETTINGS = ("xi1_only", "xi2_only", "both", "alpha_per_mode")
SPLITS = [("train", 5000, None), ("val", 1000, None), ("stress_base", 500, None)]
REPL_TUPLES, REPL_DRAWS = 100, 20


def make_latent(setting, rng):
    if setting == "alpha_per_mode":
        xi1 = {m: float(sample_alpha(rng) * rng.standard_normal()) for m in MODES}
        return LatentDraw(alpha=1.0, xi1=xi1, beta_xi2=sample_beta_xi2(rng))
    a = sample_alpha(rng)
    xi1 = {m: float(rng.standard_normal()) for m in MODES}
    b = sample_beta_xi2(rng)
    if setting == "xi1_only":
        b = 0.0
    elif setting == "xi2_only":
        xi1 = {m: 0.0 for m in MODES}
    return LatentDraw(alpha=a, xi1=xi1, beta_xi2=b)


def build(setting, seed, cfg, cfg_text, out):
    grid = DEFAULT_OMEGA_GRID
    master = np.random.default_rng(20260910 + seed)
    recs = []
    for name, n, pin in SPLITS:
        s_rng = np.random.default_rng(master.integers(0, 2**63 - 1))
        l_rng = np.random.default_rng(master.integers(0, 2**63 - 1))
        for s in draw_samples(n, s_rng, pin=pin):
            recs.append((name, -1, 0, s, make_latent(setting, l_rng),
                         np.random.default_rng(master.integers(0, 2**63 - 1))))
    # matching replicate set
    s_rng = np.random.default_rng(master.integers(0, 2**63 - 1))
    l_rng = np.random.default_rng(master.integers(0, 2**63 - 1))
    for tidx, s in enumerate(draw_samples(REPL_TUPLES, s_rng)):
        for d in range(REPL_DRAWS):
            recs.append(("replicate", tidx, d, s, make_latent(setting, l_rng),
                         np.random.default_rng(master.integers(0, 2**63 - 1))))

    n = len(recs)
    arr = dict(spectra_clean=np.empty((n, grid.size), np.float32),
               T_K=np.empty(n, np.float32), c_pct=np.empty(n, np.float32),
               E_kVcm=np.empty(n, np.float32), omega_Q=np.empty(n, np.float32),
               Gamma_Q=np.empty(n, np.float32), M=np.empty(n, np.float32),
               stratum=np.empty(n, "<U24"), split=np.empty(n, "<U16"),
               aug_params_json=np.empty(n, "<U1024"), latent_json=np.empty(n, "<U256"),
               tuple_id=np.empty(n, np.int32), draw_id=np.empty(n, np.int32))
    for i, (split, tidx, d, s, lat, arng) in enumerate(recs):
        s.T_K = float(np.float32(s.T_K)); s.c_pct = float(np.float32(s.c_pct))
        s.E_kVcm = float(np.float32(s.E_kVcm))
        cl = generate_spectrum(s.T_K, s.c_pct, s.E_kVcm, grid, latent=lat)
        arr["spectra_clean"][i] = cl["spectrum"].astype(np.float32)
        arr["T_K"][i], arr["c_pct"][i], arr["E_kVcm"][i] = s.T_K, s.c_pct, s.E_kVcm
        arr["omega_Q"][i], arr["Gamma_Q"][i], arr["M"][i] = cl["omega_Q"], cl["Gamma_Q"], cl["M"]
        arr["stratum"][i], arr["split"][i] = s.stratum, split
        arr["tuple_id"][i], arr["draw_id"][i] = tidx, d
        arr["latent_json"][i] = json.dumps(lat.to_dict(), separators=(",", ":"))
        st = SpectrumState(omega_grid=grid, spectrum=cl["spectrum"], modes=cl["modes"],
                           omega_Q=cl["omega_Q"], Gamma_Q=cl["Gamma_Q"],
                           T_K=s.T_K, c_pct=s.c_pct, E_kVcm=s.E_kVcm)
        arr["aug_params_json"][i] = json.dumps(sample_aug_params(st, cfg, arng),
                                               separators=(",", ":"))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, omega_grid=grid.astype(np.float64), **arr,
                        setting=np.array(setting, "<U20"),
                        dataset_seed=np.array(seed, np.int32),
                        augmentation_config_yaml=np.array(cfg_text[:4096], "<U4096"),
                        generator_file_sha256=np.array(hashlib.sha256(
                            (ROOT / "src" / "data"
                             / "spectrum_generator.py").read_bytes()).hexdigest(), "<U64"),
                        schema_version=np.array(2, np.int32))
    return out.stat().st_size / 1024**2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    cfg_text = CFG.read_text(); cfg = yaml.safe_load(cfg_text)
    for setting in args.settings:
        for seed in args.seeds:
            out = DS.ablation(setting, seed)
            mb = build(setting, seed, cfg, cfg_text, out)
            print(f"  {setting:15s} seed {seed}: {mb:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
