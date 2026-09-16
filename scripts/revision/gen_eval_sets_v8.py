"""Generate the round-2 evaluation sets from the corrected (v8) generator.

Two sets, both written with the same schema `InsSpectraDataset` reads:

  holdout_c_extrap_v8  300 spectra pinned at c = 2.5%, above the 2% training
                       maximum, so it is a genuine extrapolation set (unlike
                       holdout_c, whose c = 1% is interior to the training
                       range). T and E drawn from the same four-component
                       mixture as the other hold-outs.

  replicate_eval_v8    250 condition tuples drawn from the same mixture as val,
                       each with 20 independent latent realizations (5000
                       spectra). Clean spectra plus augmented copies at
                       s in {0.5, 1, 2, 4} with fixed per-spectrum augmentation
                       seeds, so severity can be evaluated without regenerating.
                       This is the set that makes the A5 within-condition
                       shuffle and the A14 within-condition ranking possible:
                       the main dataset has exactly one realization per tuple.

Both sets carry `tuple_id` and `draw_id` so replicates can be grouped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sbc.data.augmentations import (  # noqa: E402
    SpectrumState, apply_aug_params, sample_aug_params, state_from_clean,
)
from sbc.data.latent_perturbations import sample_latents  # noqa: E402
from sbc.data.sampling import draw_samples  # noqa: E402
from sbc.data.spectrum_generator import DEFAULT_OMEGA_GRID, generate_spectrum  # noqa: E402
import dataset_paths as DS  # noqa: E402

CFG_PATH = ROOT / "configs" / "augmentation_realistic.yaml"
DATA = ROOT / "data"

HOLDOUT_C_EXTRAP_SEED = 20260906
HOLDOUT_C_EXTRAP_PCT = 2.5
HOLDOUT_C_EXTRAP_N = 300

REPLICATE_SEED = 20260905
REPLICATE_N_TUPLES = 250
REPLICATE_N_DRAWS = 20
REPLICATE_SEVERITIES = (0.5, 1.0, 2.0, 4.0)


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def _gen_sha():
    p = ROOT / "src" / "data" / "spectrum_generator.py"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _build(records, cfg, cfg_text, master_seed, split_name, severities, out_path,
           dataset_version=None):
    """records: list of (tuple_id, draw_id, Sample, latent_rng_draw_fn)."""
    dataset_version = dataset_version or DS.VERSION
    grid = DEFAULT_OMEGA_GRID
    n = len(records)
    n_om = grid.size
    arr = dict(
        spectra_clean=np.empty((n, n_om), dtype=np.float32),
        T_K=np.empty(n, dtype=np.float32), c_pct=np.empty(n, dtype=np.float32),
        E_kVcm=np.empty(n, dtype=np.float32),
        omega_Q=np.empty(n, dtype=np.float32), Gamma_Q=np.empty(n, dtype=np.float32),
        M=np.empty(n, dtype=np.float32),
        stratum=np.empty(n, dtype="<U24"), split=np.empty(n, dtype="<U24"),
        aug_params_json=np.empty(n, dtype="<U1024"), latent_json=np.empty(n, dtype="<U256"),
        tuple_id=np.empty(n, dtype=np.int32), draw_id=np.empty(n, dtype=np.int32),
    )
    aug_blocks = {s: np.empty((n, n_om), dtype=np.float32) for s in severities}

    for i, (tid, did, s, latent, aug_rng) in enumerate(records):
        # Round the conditions to float32 BEFORE generating. The stored T/c/E
        # arrays are float32, and InsSpectraDataset.get_augmented regenerates
        # the clean state from those stored values; matching here makes the
        # pre-computed augmented blocks bit-identical to a runtime replay.
        s.T_K = float(np.float32(s.T_K))
        s.c_pct = float(np.float32(s.c_pct))
        s.E_kVcm = float(np.float32(s.E_kVcm))
        clean = generate_spectrum(s.T_K, s.c_pct, s.E_kVcm, grid, latent=latent)
        arr["spectra_clean"][i] = clean["spectrum"].astype(np.float32)
        arr["T_K"][i], arr["c_pct"][i], arr["E_kVcm"][i] = s.T_K, s.c_pct, s.E_kVcm
        arr["omega_Q"][i] = clean["omega_Q"]
        arr["Gamma_Q"][i] = clean["Gamma_Q"]
        arr["M"][i] = clean["M"]
        arr["stratum"][i] = s.stratum
        arr["split"][i] = split_name
        arr["tuple_id"][i], arr["draw_id"][i] = tid, did
        arr["latent_json"][i] = json.dumps(latent.to_dict(), separators=(",", ":"))
        light = SpectrumState(omega_grid=grid, spectrum=clean["spectrum"], modes=clean["modes"],
                              omega_Q=clean["omega_Q"], Gamma_Q=clean["Gamma_Q"],
                              T_K=s.T_K, c_pct=s.c_pct, E_kVcm=s.E_kVcm)
        params = sample_aug_params(light, cfg, aug_rng)
        arr["aug_params_json"][i] = json.dumps(params, separators=(",", ":"))
        for sev in severities:
            st = state_from_clean(s.T_K, s.c_pct, s.E_kVcm, omega_grid=grid, latent=latent)
            st = apply_aug_params(st, params, sev, noise_rng=None)
            aug_blocks[sev][i] = st.spectrum.astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save = dict(arr)
    save["omega_grid"] = grid.astype(np.float64)
    for sev in severities:
        save[f"spectra_aug_s{sev:g}"] = aug_blocks[sev]
    save["augmented_severities"] = np.array(list(severities), dtype=np.float32)
    save["master_seed"] = np.array(master_seed, dtype=np.int64)
    save["augmentation_config_yaml"] = np.array(cfg_text[:4096], dtype="<U4096")
    save["generator_git_sha"] = np.array(_git_sha(), dtype="<U40")
    save["generator_file_sha256"] = np.array(_gen_sha(), dtype="<U64")
    save["dataset_version"] = np.array(dataset_version, dtype="<U8")
    save["schema_version"] = np.array(2, dtype=np.int32)
    np.savez_compressed(out_path, **save)
    return out_path.stat().st_size / (1024 ** 2)


def build_holdout_c_extrap(cfg, cfg_text):
    rng = np.random.default_rng(HOLDOUT_C_EXTRAP_SEED)
    sampler_rng = np.random.default_rng(rng.integers(0, 2 ** 63 - 1))
    latent_rng = np.random.default_rng(rng.integers(0, 2 ** 63 - 1))
    samples = draw_samples(HOLDOUT_C_EXTRAP_N, sampler_rng, pin={"c_pct": HOLDOUT_C_EXTRAP_PCT})
    records = []
    for i, s in enumerate(samples):
        records.append((i, 0, s, sample_latents(latent_rng),
                        np.random.default_rng(rng.integers(0, 2 ** 63 - 1))))
    out = DATA / f"holdout_c_extrap_{DS.VERSION}" / "dataset.npz"
    mb = _build(records, cfg, cfg_text, HOLDOUT_C_EXTRAP_SEED, "holdout_c_extrap",
                REPLICATE_SEVERITIES, out)
    print(f"  holdout_c_extrap_{DS.VERSION}: n={len(records)} c={HOLDOUT_C_EXTRAP_PCT}%  {mb:.1f} MiB")
    return out


def build_replicate_eval(cfg, cfg_text):
    rng = np.random.default_rng(REPLICATE_SEED)
    sampler_rng = np.random.default_rng(rng.integers(0, 2 ** 63 - 1))
    latent_rng = np.random.default_rng(rng.integers(0, 2 ** 63 - 1))
    tuples = draw_samples(REPLICATE_N_TUPLES, sampler_rng)
    assert all(t.T_K <= 600.0 for t in tuples)
    records = []
    for tid, s in enumerate(tuples):
        for did in range(REPLICATE_N_DRAWS):
            records.append((tid, did, s, sample_latents(latent_rng),
                            np.random.default_rng(rng.integers(0, 2 ** 63 - 1))))
    out = DATA / f"replicate_eval_{DS.VERSION}" / "dataset.npz"
    mb = _build(records, cfg, cfg_text, REPLICATE_SEED, "replicate_eval",
                REPLICATE_SEVERITIES, out)
    print(f"  replicate_eval_{DS.VERSION}: {REPLICATE_N_TUPLES} tuples x {REPLICATE_N_DRAWS} draws "
          f"= {len(records)} spectra  {mb:.1f} MiB")
    return out


# --------------------------------------------------------------------------- #
# Untouched evaluation sets (round 42/43). Fresh seeds, drawn from the same
# condition mixture as val/replicate_eval, and never used for checkpoint
# selection or architecture tuning. `val` remains the selection set; headline
# numbers are reported on these.
# --------------------------------------------------------------------------- #
TEST_SEED = 20260912
REPLICATE_TEST_SEED = 20260913
TEST_N = 2000


def build_test(cfg, cfg_text):
    rng = np.random.default_rng(TEST_SEED)
    sampler_rng = np.random.default_rng(rng.integers(0, 2 ** 63 - 1))
    latent_rng = np.random.default_rng(rng.integers(0, 2 ** 63 - 1))
    samples = draw_samples(TEST_N, sampler_rng)
    records = [(i, 0, s, sample_latents(latent_rng),
                np.random.default_rng(rng.integers(0, 2 ** 63 - 1)))
               for i, s in enumerate(samples)]
    out = DATA / f"test_{DS.VERSION}" / "dataset.npz"
    mb = _build(records, cfg, cfg_text, TEST_SEED, "test", REPLICATE_SEVERITIES, out)
    print(f"  test_{DS.VERSION}: n={len(records)}  seed={TEST_SEED}  {mb:.1f} MiB")
    return out


def build_replicate_test(cfg, cfg_text):
    rng = np.random.default_rng(REPLICATE_TEST_SEED)
    sampler_rng = np.random.default_rng(rng.integers(0, 2 ** 63 - 1))
    latent_rng = np.random.default_rng(rng.integers(0, 2 ** 63 - 1))
    tuples = draw_samples(REPLICATE_N_TUPLES, sampler_rng)
    assert all(t.T_K <= 600.0 for t in tuples)
    records = []
    for tid, s in enumerate(tuples):
        for did in range(REPLICATE_N_DRAWS):
            records.append((tid, did, s, sample_latents(latent_rng),
                            np.random.default_rng(rng.integers(0, 2 ** 63 - 1))))
    out = DATA / f"replicate_test_{DS.VERSION}" / "dataset.npz"
    mb = _build(records, cfg, cfg_text, REPLICATE_TEST_SEED, "replicate_test",
                REPLICATE_SEVERITIES, out)
    print(f"  replicate_test_{DS.VERSION}: {REPLICATE_N_TUPLES} tuples x {REPLICATE_N_DRAWS} "
          f"= {len(records)} spectra  seed={REPLICATE_TEST_SEED}  {mb:.1f} MiB")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=("both", "holdout", "replicate", "test",
                                        "replicate_test", "untouched"), default="both")
    args = ap.parse_args()
    cfg_text = CFG_PATH.read_text()
    cfg = yaml.safe_load(cfg_text)
    print(f"generator sha256 = {_gen_sha()[:16]}…  HEAD = {_git_sha()[:8]}  "
          f"{datetime.now(timezone.utc).isoformat()}")
    if args.which in ("both", "holdout"):
        build_holdout_c_extrap(cfg, cfg_text)
    if args.which in ("untouched", "test"):
        build_test(cfg, cfg_text)
    if args.which in ("untouched", "replicate_test"):
        build_replicate_test(cfg, cfg_text)
    if args.which in ("both", "replicate"):
        build_replicate_eval(cfg, cfg_text)


if __name__ == "__main__":
    main()
