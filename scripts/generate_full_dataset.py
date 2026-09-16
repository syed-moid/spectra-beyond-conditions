"""Generate the Paper 1 full dataset.

Phase 0 (default):  data/full_dataset/dataset.npz       (deterministic M)
Phase 1 (--phase1): data/full_dataset_phase1/dataset.npz (latent-microphysics)

Splits:
    train         : 5000  (physics-aware sampler, c-exclusion at 1%)
    val           : 1000  (same physics-aware sampler as train)
    holdout_T     :  300  (T pinned = 600; (c, E) per stratum, no c-exclusion)
    holdout_c     :  300  (c pinned = 1;   (T, E) per stratum)
    holdout_E     :  300  (E pinned = 4;   (T, c) per stratum, no c-exclusion)
    stress_base   :  500  (independent draws from same physics-aware sampler as train)

Total: 7400 base spectra. Augmented spectra are NOT stored — Dataset
replays them at runtime from the per-spectrum aug_params at any severity.

Phase 1 (Session 8). Each spectrum draws one seeded LatentDraw (per-mode
scalar Gamma perturbation xi1 + soft-mode omega shift xi2). The REALIZED
spectrum and REALIZED (M, omega_Q, Gamma_Q) targets are stored, plus the
latent draw as `latent_json` for audit and for the runtime replay path
(state_from_clean reconstructs the realized clean spectrum from the latent).
The latent affects M only through the realized soft-mode (omega, Gamma) via
the validated composite merit(). Phase 0 is left intact for the live
NL-cond-floor comparison.

Reference augmented spectra: 10 first-train samples x 4 severities are
computed and stored in `replay_test_reference` (regenerated with the
Session-8 Delta-omega fix; the prior Phase-0 references are stale at E!=0).

Held-out test sets MUST NEVER be used for training or hyperparameter search.

Schema invariant (Session 5): coordinate arrays on the replay path are
float64 (omega_grid); spectra are float32. New replay-path arrays must be
float64; end-product arrays float32.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.augmentations import (  # noqa: E402
    apply_aug_params, sample_aug_params, state_from_clean, SpectrumState,
)
from src.data.latent_perturbations import sample_latents  # noqa: E402
from src.data.sampling import draw_samples  # noqa: E402
from src.data.spectrum_generator import DEFAULT_OMEGA_GRID, generate_spectrum  # noqa: E402

MASTER_SEED: int = 20260517
PHASE1_MASTER_SEED: int = 20260520
CFG_PATH: Path = ROOT / "configs" / "augmentation_realistic.yaml"

SPLITS: list[tuple[str, int, dict | None]] = [
    ("train",       5000, None),
    ("val",         1000, None),
    ("holdout_T",    300, {"T_K": 600.0}),
    ("holdout_c",    300, {"c_pct": 1.0}),
    ("holdout_E",    300, {"E_kVcm": 4.0}),
    ("stress_base",  500, None),
]

REPLAY_SEVERITIES: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
REPLAY_TEST_SIZE: int = 10  # first 10 train indices


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", action="store_true",
                    help="Inject per-spectrum latent-microphysics perturbations.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output .npz path. Defaults per phase.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Master seed. Defaults per phase.")
    args = ap.parse_args()

    phase1 = bool(args.phase1)
    master_seed = args.seed if args.seed is not None else (PHASE1_MASTER_SEED if phase1 else MASTER_SEED)
    if args.out is not None:
        out_path = Path(args.out)
    else:
        out_path = ROOT / "data" / ("full_dataset_phase1" if phase1 else "full_dataset") / "dataset.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema_version = 2 if phase1 else 1

    print(f"Phase {'1 (latent-microphysics)' if phase1 else '0 (deterministic M)'}; "
          f"seed={master_seed}; out={out_path}")

    cfg_text = CFG_PATH.read_text()
    cfg = yaml.safe_load(cfg_text)

    master_rng = np.random.default_rng(master_seed)
    omega_grid = DEFAULT_OMEGA_GRID

    N_total = sum(n for _, n, _ in SPLITS)
    n_omega = omega_grid.size

    spectra = np.empty((N_total, n_omega), dtype=np.float32)
    T_K = np.empty(N_total, dtype=np.float32)
    c_pct = np.empty(N_total, dtype=np.float32)
    E_kVcm = np.empty(N_total, dtype=np.float32)
    omega_Q = np.empty(N_total, dtype=np.float32)
    Gamma_Q = np.empty(N_total, dtype=np.float32)
    M = np.empty(N_total, dtype=np.float32)
    stratum = np.empty(N_total, dtype="<U24")
    split_arr = np.empty(N_total, dtype="<U16")
    aug_json = np.empty(N_total, dtype="<U1024")
    latent_json = np.empty(N_total, dtype="<U256")   # "" for Phase 0
    latents: list = [None] * N_total                 # in-memory, for replay ref

    cursor = 0
    train_start = 0
    for split_name, n, pin in SPLITS:
        print(f"  generating {split_name} (n={n}, pin={pin}) ...", flush=True)
        sampler_rng = np.random.default_rng(master_rng.integers(0, 2 ** 63 - 1))
        # Phase 1: dedicated per-split latent RNG (only consumes master stream
        # when phase1, so Phase-0 reproducibility is preserved bit-for-bit).
        latent_rng = (np.random.default_rng(master_rng.integers(0, 2 ** 63 - 1))
                      if phase1 else None)
        samples = draw_samples(n, sampler_rng, pin=pin)
        for i, s in enumerate(samples):
            idx = cursor + i
            latent = sample_latents(latent_rng) if phase1 else None
            clean = generate_spectrum(s.T_K, s.c_pct, s.E_kVcm, omega_grid, latent=latent)
            spectra[idx] = clean["spectrum"].astype(np.float32)
            T_K[idx] = s.T_K
            c_pct[idx] = s.c_pct
            E_kVcm[idx] = s.E_kVcm
            omega_Q[idx] = clean["omega_Q"]
            Gamma_Q[idx] = clean["Gamma_Q"]
            M[idx] = clean["M"]
            stratum[idx] = s.stratum
            split_arr[idx] = split_name
            latents[idx] = latent
            latent_json[idx] = json.dumps(latent.to_dict(), separators=(",", ":")) if latent else ""

            # Sample augmentation params at severity=1 against the REALIZED clean.
            aug_rng = np.random.default_rng(master_rng.integers(0, 2 ** 63 - 1))
            light_state = SpectrumState(
                omega_grid=omega_grid,
                spectrum=clean["spectrum"],
                modes=clean["modes"],
                omega_Q=clean["omega_Q"],
                Gamma_Q=clean["Gamma_Q"],
                T_K=s.T_K, c_pct=s.c_pct, E_kVcm=s.E_kVcm,
            )
            params = sample_aug_params(light_state, cfg, aug_rng)
            aug_json[idx] = json.dumps(params, separators=(",", ":"))

        if split_name == "train":
            train_start = cursor
        cursor += n

    assert cursor == N_total

    # Replay reference: regenerated WITH the per-spectrum latent (Phase 1) so the
    # realized clean is reconstructed before augmentation.
    print(f"  building replay reference (n={REPLAY_TEST_SIZE} x {len(REPLAY_SEVERITIES)}) ...", flush=True)
    ref_indices = np.arange(train_start, train_start + REPLAY_TEST_SIZE, dtype=np.int32)
    ref_block = np.empty((REPLAY_TEST_SIZE, len(REPLAY_SEVERITIES), n_omega), dtype=np.float32)
    for j, idx in enumerate(ref_indices):
        params = json.loads(str(aug_json[idx]))
        for k, sev in enumerate(REPLAY_SEVERITIES):
            state = state_from_clean(
                float(T_K[idx]), float(c_pct[idx]), float(E_kVcm[idx]),
                omega_grid=omega_grid, latent=latents[idx],
            )
            state = apply_aug_params(state, params, sev, noise_rng=None)
            ref_block[j, k] = state.spectrum.astype(np.float32)

    save_kwargs = dict(
        omega_grid=omega_grid.astype(np.float64),
        spectra_clean=spectra,
        T_K=T_K, c_pct=c_pct, E_kVcm=E_kVcm,
        omega_Q=omega_Q, Gamma_Q=Gamma_Q, M=M,
        stratum=stratum, split=split_arr,
        aug_params_json=aug_json,
        replay_test_reference=ref_block,
        replay_test_indices=ref_indices,
        replay_test_severities=np.array(REPLAY_SEVERITIES, dtype=np.float32),
        master_seed=np.array(master_seed, dtype=np.int64),
        augmentation_config_yaml=np.array(cfg_text[:4096], dtype="<U4096"),
        generator_git_sha=np.array(_git_sha(), dtype="<U40"),
        schema_version=np.array(schema_version, dtype=np.int32),
    )
    if phase1:
        save_kwargs["latent_json"] = latent_json
    np.savez_compressed(out_path, **save_kwargs)

    size_mb = out_path.stat().st_size / (1024 ** 2)
    print(f"\nWrote {out_path} ({size_mb:.1f} MiB)  schema_version={schema_version}")
    print(f"Total samples: {N_total}")
    for split_name, n, _ in SPLITS:
        sub_mask = split_arr == split_name
        sub_M = M[sub_mask]
        print(f"  {split_name:>12}  n={n:>4}  M[{sub_M.min():.3g}, {sub_M.max():.3g}] med={np.median(sub_M):.3g}")


if __name__ == "__main__":
    main()
