"""Regenerate replicate_eval_v8/augmented_blocks.npz from dataset_clean.npz.

The replicate evaluation set stores its augmented spectra at s in {0.5, 1, 2, 4}
as a separate, gitignored file so the tracked/deposited artefact stays small.
Those blocks are fully determined by dataset_clean.npz: the conditions, the
latent draw, the severity-1 augmentation parameters and the stored noise seed.
This script rebuilds them bit-for-bit.

    python scripts/revision/regenerate_augmented_blocks.py [--verify]

--verify rebuilds into memory and compares against the existing blocks file
instead of writing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sbc.data.augmentations import apply_aug_params, state_from_clean  # noqa: E402
from sbc.data.latent_perturbations import LatentDraw  # noqa: E402
import dataset_paths as DS

DEFAULT_DIR = DS.REPLICATE_DIR


def build_blocks(clean_path: Path) -> dict[str, np.ndarray]:
    z = np.load(clean_path, allow_pickle=False)
    grid = z["omega_grid"]
    sevs = [float(s) for s in z["augmented_severities"]]
    T, c, E = z["T_K"], z["c_pct"], z["E_kVcm"]
    params = [json.loads(str(s)) for s in z["aug_params_json"]]
    lat = [LatentDraw.from_dict(json.loads(str(s))) for s in z["latent_json"]]
    n = len(T)
    out = {f"spectra_aug_s{s:g}": np.empty((n, grid.size), dtype=np.float32) for s in sevs}
    for i in range(n):
        for s in sevs:
            st = state_from_clean(float(T[i]), float(c[i]), float(E[i]),
                                  omega_grid=grid, latent=lat[i])
            st = apply_aug_params(st, params[i], s, noise_rng=None)
            out[f"spectra_aug_s{s:g}"][i] = st.spectrum.astype(np.float32)
    out["augmented_severities"] = np.array(sevs, dtype=np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    clean = args.dir / "dataset_clean.npz"
    blocks = args.dir / "augmented_blocks.npz"

    built = build_blocks(clean)
    if args.verify:
        ref = np.load(blocks, allow_pickle=False)
        ok = True
        for k, v in built.items():
            same = np.array_equal(ref[k], v)
            ok &= same
            print(f"  {k:22s} identical: {same}")
        print(f"\nALL BLOCKS BIT-IDENTICAL: {ok}")
        return 0 if ok else 1
    np.savez_compressed(blocks, **built)
    print(f"wrote {blocks} ({blocks.stat().st_size / 1024**2:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
