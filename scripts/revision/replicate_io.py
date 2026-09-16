"""Loader for the replicate evaluation set.

The augmented severity blocks live in a separate, gitignored file. Consumers
call `load_replicate()` and get the blocks either from disk (fast path) or
regenerated in memory (bit-identical, ~40 s), without caring which.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import dataset_paths as DS

ROOT = Path(__file__).resolve().parents[2]
REPLICATE_DIR = DS.REPLICATE_DIR


def clean_path(directory: Path = REPLICATE_DIR) -> Path:
    """The clean-spectra file, under either layout.

    v8 split the set into `dataset_clean.npz` plus a separate, gitignored
    `augmented_blocks.npz`. v9 writes one self-contained `dataset.npz` holding
    the clean spectra and the augmented severity blocks together. Both are
    accepted so the same consumers work against either realization.
    """
    for name in ("dataset_clean.npz", "dataset.npz"):
        p = directory / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"no dataset_clean.npz or dataset.npz in {directory}")


def load_replicate(directory: Path = REPLICATE_DIR, want_blocks: bool = True):
    """Return (clean_npz, blocks_dict). `blocks_dict` is None when want_blocks is False."""
    path = clean_path(directory)
    clean = np.load(path, allow_pickle=False)
    if not want_blocks:
        return clean, None

    # v9: blocks live in the same file
    blocks = {k: clean[k] for k in clean.files if k.startswith("spectra_aug_s")}
    if blocks:
        return clean, blocks

    # v8: separate blocks file, else regenerate (bit-identical, ~40 s)
    blocks_path = directory / "augmented_blocks.npz"
    if blocks_path.exists():
        z = np.load(blocks_path, allow_pickle=False)
        return clean, {k: z[k] for k in z.files if k.startswith("spectra_aug_s")}
    from regenerate_augmented_blocks import build_blocks
    built = build_blocks(path)
    return clean, {k: v for k, v in built.items() if k.startswith("spectra_aug_s")}


def block_key(severity: float) -> str:
    return f"spectra_aug_s{severity:g}"
