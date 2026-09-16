"""Single source of truth for which dataset realization the analysis reads.

Every script that consumes generated spectra or reads a trained checkpoint
imports this module instead of hardcoding a path. The version is taken from
the `INS_DATASET_VERSION` environment variable and defaults to the current
realization, so a resumed pipeline cannot silently read a superseded set.

Provenance fields are derived from `VERSION`: no script writes a literal
version string. Round-33 note: v9 result directories carry the version in the
directory name, so a v9 run never overwrites the v8 record. The mapping is
declared in `_REV`, not derived, because the v8 names predate the convention.

To reproduce a v8-era result:  INS_DATASET_VERSION=v8 python <script>.py
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REV = ROOT / "results" / "revision"

VERSION = os.environ.get("INS_DATASET_VERSION", "v9")
if VERSION not in ("v7", "v8", "v9", "v10"):
    raise ValueError(f"INS_DATASET_VERSION={VERSION!r}; expected v7, v8, v9 or v10")

# --- datasets -------------------------------------------------------------
FULL = DATA / f"full_dataset_phase1_{VERSION}" / "dataset.npz"
HOLDOUT_C_EXTRAP = DATA / f"holdout_c_extrap_{VERSION}" / "dataset.npz"
REPLICATE_DIR = DATA / f"replicate_eval_{VERSION}"


def _replicate_file() -> Path:
    """Clean-spectra file of the replicate set, under either layout.

    v8 wrote `dataset_clean.npz` beside a separate `augmented_blocks.npz`;
    v9 writes one self-contained `dataset.npz`. Checked in that order.
    """
    for name in ("dataset_clean.npz", "dataset.npz"):
        p = REPLICATE_DIR / name
        if p.exists():
            return p
    return REPLICATE_DIR / "dataset.npz"


REPLICATE = _replicate_file()

# Untouched evaluation sets (round 42/43). Generated with fresh seeds and never
# used for checkpoint selection or architecture tuning; `val` is the selection
# set. Present from v10 onward.
TEST = DATA / f"test_{VERSION}" / "dataset.npz"
REPLICATE_TEST_DIR = DATA / f"replicate_test_{VERSION}"
REPLICATE_TEST = REPLICATE_TEST_DIR / "dataset.npz"


def ablation(setting: str, seed: int) -> Path:
    """Dataset for one A7 channel-ablation setting. v8 sets carry no version tag."""
    stem = f"ablation_{setting}_seed{seed}" if VERSION == "v8" \
        else f"ablation_{VERSION}_{setting}_seed{seed}"
    return DATA / stem / "dataset.npz"


# --- result directories ---------------------------------------------------
# v8 names predate the versioned convention, so the map is explicit per version.
_REV = {
    "v8": {k: k for k in (
        "A1", "A3", "A4_extended", "A5", "A7", "A17", "A19",
        "A18b_tilt_trained", "A12", "A6", "A8", "A9", "A10", "A14",
        "A18", "A18b")},
    "v9": {
        "A1": "A1_v9", "A3": "A3_v9", "A4_extended": "A4_v9_extended",
        "A5": "A5_v9", "A7": "A7_v9", "A17": "A17_v9", "A19": "A19_v9",
        "A18b_tilt_trained": "A18b_v9_tilt", "A12": "A12_v9",
        "A6": "A6_v9", "A8": "A8_v9", "A9": "A9_v9", "A10": "A10_v9",
        "A14": "A14_v9", "A18": "A18_v9", "A18b": "A18b_v9",
    },
    "v10": {
        "A1": "A1_v10", "A3": "A3_v10", "A4": "A4_v10", "A4_extended": "A4_v10_extended",
        "A5": "A5_v10", "A7": "A7_v10", "A17": "A17_v10", "A19": "A19_v10",
        "A18b_tilt_trained": "A18b_v10_tilt", "A12": "A12_v10", "A21": "A21_v10", "A22": "A22_v10", "A23": "A23_v10", "A24": "A24_v10",
        "A6": "A6_v10", "A8": "A8_v10", "A9": "A9_v10", "A10": "A10_v10",
        "A14": "A14_v10", "A18": "A18_v10", "A18b": "A18b_v10",
    },
}


def rev(task: str) -> Path:
    """Result directory for `task` under the active dataset version."""
    try:
        return REV / _REV[VERSION][task]
    except KeyError:
        raise KeyError(
            f"no {VERSION} result directory declared for {task!r}; "
            f"add it to dataset_paths._REV") from None


# Tuned architecture variants (cnn_kernel45, tf_patch60) were trained into A19 on
# v9 but into A17 on v10, because the v10 sweep trained every variant -- base,
# tuned and ablated -- into one directory. Six scripts hardcoded the A19 path and
# each failed the same way on v10. Resolve through here instead.
TUNED_CKPT_TASKS = ("A19", "A17")


def tuned_ckpt(name: str, seed: int) -> Path:
    """Checkpoint for a tuned architecture variant, wherever this version put it."""
    tried = []
    for task in TUNED_CKPT_TASKS:
        try:
            d = rev(task) / "checkpoints"
        except KeyError:
            continue
        q = d / f"{name}_v7_seed{seed}.pt"
        tried.append(q)
        if q.exists():
            return q
    raise FileNotFoundError(
        f"no {VERSION} checkpoint for tuned variant {name!r} seed {seed}; looked in "
        + ", ".join(str(t.parent.name) for t in tried))


def tuned_ckpt_dir(name: str, seed: int = 42) -> Path:
    """Directory holding the tuned variants for this version."""
    return tuned_ckpt(name, seed).parent


def provenance() -> dict:
    """Version fields every sidecar and run CSV should carry."""
    return {"dataset_version": VERSION, "dataset": str(FULL.relative_to(ROOT))}
