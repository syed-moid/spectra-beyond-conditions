"""A7 post-processing: the seven summary tables built from `training_runs.csv`.

Reconstructed (round 36) because the v8 copies were produced ad hoc and the code
was not kept. Validation against the v8 files is in `reports/A7_postprocess.md`.

Outputs, in the order they are produced:

  training_runs_dedup.csv        first occurrence per (setting, dataset_seed, arch, seed)
  mps_reproducibility_pairs.csv  the discarded duplicates, paired against the kept run
  marginal_check.csv             realized linewidth-multiplier deviation, per setting
  cross_mode_correlation.csv     the same quantity correlated across modes
  oracle_per_setting.csv         metadata-only (conditional-median) floor, per setting
  improvement_over_oracle.csv    model MAE against that floor, with paired-bootstrap CIs
  rho_per_setting.csv            within-tuple rank correlation on the replicate split
  variance_decomposition.csv     dataset-seed vs training-seed components

Conventions that the v8 files do not pin down are stated in the report and, where
they matter, emitted as extra columns rather than resolved silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skewnorm, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from a4_eval import load_a4_model  # noqa: E402
from sbc.data.dataset import InsSpectraDataset  # noqa: E402
from sbc.data.latent_perturbations import (  # noqa: E402
    ALPHA_MU_LOG, ALPHA_SIGMA_LOG, BETA_XI2_HIGH, BETA_XI2_LOW, BETA_XI2_SKEW,
    GAMMA_FLOOR, MODES, _BETA_XI2_LOC, _BETA_XI2_SCALE,
)
from sbc.data.merit import ALPHA_DEFAULT, K_B_meV_per_K, T_C_K  # noqa: E402
from sbc.data.spectrum_generator import Gamma_Q, omega_Q  # noqa: E402
import dataset_paths as DS  # noqa: E402

KEY = ["setting", "dataset_seed", "arch", "seed"]
CELL = ["setting", "dataset_seed", "arch"]
LATENT_SETTINGS = ("both", "alpha_per_mode")   # the only ones whose alpha structure differs


# --------------------------------------------------------------------------- #
# latent-level checks (no model, no Monte Carlo: exact)
# --------------------------------------------------------------------------- #
def multiplier_deviation(latents, mode="soft"):
    """|max(GAMMA_FLOOR, 1 + alpha*xi1[mode]) - 1|, the realized deviation of the
    linewidth multiplier from 1 after the floor is applied."""
    a = np.array([d["alpha"] for d in latents])
    x = np.array([d["xi1"][mode] for d in latents])
    return np.abs(np.maximum(GAMMA_FLOOR, 1.0 + a * x) - 1.0), (1.0 + a * x)


def latent_tables(settings, dataset_seed):
    marg, cross = [], []
    for setting in settings:
        npz = DS.ablation(setting, dataset_seed)
        if not npz.exists():
            continue
        z = np.load(npz, allow_pickle=False)
        lat = [json.loads(s) for s in z["latent_json"]]
        dev, raw = multiplier_deviation(lat, "soft")
        marg.append({"setting": setting, "n": len(dev),
                     "P50": float(np.percentile(dev, 50)),
                     "P90": float(np.percentile(dev, 90)),
                     "P95": float(np.percentile(dev, 95)),
                     "floor_hit": float((raw < GAMMA_FLOOR).mean()),
                     "mean": float(dev.mean()),
                     "dataset_seed": dataset_seed})
        per = {m: multiplier_deviation(lat, m)[0] for m in MODES}
        cross.append({"setting": setting,
                      "pearson_soft_acoustic": float(np.corrcoef(per["soft"], per["acoustic"])[0, 1]),
                      "pearson_soft_optical": float(np.corrcoef(per["soft"], per["optical"])[0, 1]),
                      "spearman_soft_acoustic": float(spearmanr(per["soft"], per["acoustic"]).statistic),
                      "dataset_seed": dataset_seed})
    return pd.DataFrame(marg), pd.DataFrame(cross)


# --------------------------------------------------------------------------- #
# oracle
# --------------------------------------------------------------------------- #
def merit_vec(om, gm, T, E):
    return ((om / gm) * np.exp(-ALPHA_DEFAULT * np.abs((T - T_C_K) / T_C_K))
            * np.exp(-om / (2.0 * K_B_meV_per_K * T))
            * (1.0 / (1.0 + (gm / (om / 2.0)) ** 2))
            * (1.0 + 0.05 * E / (1.0 + E)))


def conditional_logM(setting, T, c_pct, E, n, rng):
    """log M draws at one condition tuple under `setting`'s latent model.

    Only the soft mode enters M, so only alpha*xi1[soft] and beta*xi2 matter.
    Mirrors a7_generate_ablation.make_latent: `alpha_per_mode` folds the
    per-mode amplitude into xi1 with alpha = 1, which leaves the soft-mode
    product alpha*xi1 distributed exactly as in `both`.
    """
    om0 = omega_Q(T, c_pct / 100.0, E)
    gm0 = Gamma_Q(T, c_pct / 100.0, E)
    if setting == "xi2_only":
        ax = np.zeros(n)
    else:
        ax = rng.lognormal(ALPHA_MU_LOG, ALPHA_SIGMA_LOG, n) * rng.standard_normal(n)
    if setting == "xi1_only":
        bx2 = np.zeros(n)
    else:
        raw = skewnorm.rvs(a=BETA_XI2_SKEW, size=n, random_state=rng)
        bx2 = np.clip(raw * _BETA_XI2_SCALE + _BETA_XI2_LOC, BETA_XI2_LOW, BETA_XI2_HIGH)
    om = om0 * (1.0 + bx2)
    gm = gm0 * np.maximum(GAMMA_FLOOR, 1.0 + ax)
    return np.log(np.clip(merit_vec(om, gm, T, E), 1e-9, None))


def oracle_table(cells, n_draws, seed):
    rows = []
    for setting, ds_seed in cells:
        npz = DS.ablation(setting, ds_seed)
        if not npz.exists():
            continue
        z = np.load(npz, allow_pickle=False)
        m = z["split"] == "val"
        y = np.log(np.clip(z["M"][m].astype(float), 1e-9, None))
        T, c, E = (z["T_K"][m].astype(float), z["c_pct"][m].astype(float),
                   z["E_kVcm"][m].astype(float))
        rng = np.random.default_rng(seed)
        med = np.array([np.median(conditional_logM(setting, T[i], c[i], E[i], n_draws, rng))
                        for i in range(len(y))])
        rows.append({"setting": setting, "dataset_seed": ds_seed, "n": int(len(y)),
                     "oracle_MAE": float(np.abs(y - med).mean()),
                     "marginal_median_MAE": float(np.abs(y - np.median(y)).mean())})
        # per-row oracle error is reused by the bootstrap
        rows[-1]["_err"] = np.abs(y - med)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# model-dependent tables
# --------------------------------------------------------------------------- #
def _models(setting, ds_seed, arch, seeds, ckpt_root, device):
    d = ckpt_root / f"{setting}_seed{ds_seed}"
    return [load_a4_model(arch, "v7", s, device=device, ckpt_dir=d) for s in seeds]


def _batch(ds):
    return {"T_K": ds._T_K.astype(float), "c_pct": ds._c_pct.astype(float),
            "E_kVcm": ds._E_kVcm.astype(float)}


def tuple_rho(tid, y, p):
    out = []
    for t in np.unique(tid):
        m = tid == t
        r = spearmanr(y[m], p[m]).statistic if np.ptp(p[m]) > 0 else 0.0
        out.append(0.0 if not np.isfinite(r) else float(r))
    return float(np.mean(out))


def improvement_table(dedup, oracle, ckpt_root, seeds, n_boot, seed, device):
    """Model MAE against the oracle floor, with a paired bootstrap over val rows.

    The point estimate keeps the v8 convention: the mean of the per-seed val
    MAEs, which equals the row-mean of the per-seed absolute errors, so the
    bootstrap resamples that same quantity paired with the oracle's errors.
    """
    rng = np.random.default_rng(seed)
    orc = oracle.set_index(["setting", "dataset_seed"])
    rows = []
    for (setting, ds_seed, arch), g in dedup.groupby(CELL, sort=False):
        if (setting, ds_seed) not in orc.index:
            continue
        o = orc.loc[(setting, ds_seed)]
        npz = DS.ablation(setting, ds_seed)
        va = InsSpectraDataset(npz, "val", severity=1.0, as_torch=False)
        X = np.stack([va.get_augmented(i, 1.0) for i in range(len(va))]).astype(np.float32)
        y = np.log(np.clip(va._M.astype(float), 1e-9, None))
        b = _batch(va)
        errs = [np.abs(m.predict_logM({"spectrum": X, **b}) - y)
                for m in _models(setting, ds_seed, arch, sorted(g.seed.unique()), ckpt_root, device)]
        e_model = np.mean(errs, axis=0)
        e_orac = np.asarray(o["_err"])
        idx = rng.integers(0, len(y), size=(n_boot, len(y)))
        imp = 1.0 - e_model[idx].mean(1) / e_orac[idx].mean(1)
        rows.append({
            "setting": setting, "dataset_seed": ds_seed, "arch": arch,
            "best_val_MAE_logM": float(g.best_val_MAE_logM.mean()),
            "n": int(o["n"]), "oracle_MAE": float(o["oracle_MAE"]),
            "marginal_median_MAE": float(o["marginal_median_MAE"]),
            "improvement_over_oracle": float(1.0 - e_model.mean() / e_orac.mean()),
            "improvement_ci_lo": float(np.percentile(imp, 2.5)),
            "improvement_ci_hi": float(np.percentile(imp, 97.5)),
            "n_boot": int(n_boot),
        })
    return pd.DataFrame(rows)


def rho_table(dedup, ckpt_root, device):
    """Within-tuple rank correlation on the setting's own replicate split.

    v8 reports one `rho_logM`/`MAE_logM` per cell without recording whether it
    is the seed ensemble or the mean over seeds. Both are emitted; the ensemble
    is the primary column, matching how rho is reported elsewhere (A14, A19).
    """
    rows = []
    for (setting, ds_seed, arch), g in dedup.groupby(CELL, sort=False):
        npz = DS.ablation(setting, ds_seed)
        z = np.load(npz, allow_pickle=False)
        tid = z["tuple_id"][z["split"] == "replicate"]
        rp = InsSpectraDataset(npz, "replicate", severity=1.0, as_torch=False)
        X = np.stack([rp.get_augmented(i, 1.0) for i in range(len(rp))]).astype(np.float32)
        y = np.log(np.clip(rp._M.astype(float), 1e-9, None))
        om_t = rp._omega_Q.astype(float)
        gm_t = np.log(np.clip(rp._Gamma_Q.astype(float), 1e-9, None))
        b = _batch(rp)
        ms = _models(setting, ds_seed, arch, sorted(g.seed.unique()), ckpt_root, device)
        P = [m.predict_all({"spectrum": X, **b}) for m in ms]
        logM = [np.log(np.clip(p["M"], 1e-9, None)) for p in P]
        ens = np.mean(logM, axis=0)
        om_e = np.mean([p["omega_Q"] for p in P], axis=0)
        gm_e = np.log(np.clip(np.mean([p["Gamma_Q"] for p in P], axis=0), 1e-9, None))
        rows.append({
            "setting": setting, "dataset_seed": ds_seed, "arch": arch,
            "rho_logM": tuple_rho(tid, y, ens),
            "rho_logGamma": tuple_rho(tid, gm_t, gm_e),
            "rho_omega0": tuple_rho(tid, om_t, om_e),
            "MAE_logM": float(np.abs(ens - y).mean()),
            "rho_logM_seedmean": float(np.mean([tuple_rho(tid, y, p) for p in logM])),
            "MAE_logM_seedmean": float(np.mean([np.abs(p - y).mean() for p in logM])),
            "n_tuples": int(len(np.unique(tid))), "n": int(len(y)),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# variance decomposition
# --------------------------------------------------------------------------- #
def variance_table(dedup):
    """Dataset-seed vs training-seed components on the 3x3 grid.

    Estimator (round-37 decision 2): nested between/within with ddof = 1.
    `var_dataset_seed` is the variance of the three dataset-seed means, each
    taken over its training seeds; `var_training_seed` is the mean of the
    within-dataset-seed variances. Fractions are of their sum, reported with
    the cell count.

    The round-36 two-way random-effects spec was withdrawn: on a 3x3 grid with
    no replication it leaves 4 residual d.o.f. and returns negative components
    wherever the true dataset-seed term is near zero. Its values are retained
    in `*_twoway` columns as a diagnostic only.
    """
    rows = []
    for (setting, arch), g in dedup.groupby(["setting", "arch"], sort=False):
        piv = g.pivot_table(index="dataset_seed", columns="seed", values="best_val_MAE_logM")
        y = piv.values
        r, c = y.shape
        if r < 2 or c < 2 or not np.isfinite(y).all():
            continue
        v_ds = float(y.mean(1).var(ddof=1))
        v_tr = float(np.mean(y.var(axis=1, ddof=1)))
        tot = v_ds + v_tr

        gm = y.mean()
        rm, cm = y.mean(1), y.mean(0)
        MSA = c * ((rm - gm) ** 2).sum() / (r - 1)
        MSB = r * ((cm - gm) ** 2).sum() / (c - 1)
        MSE = ((y - rm[:, None] - cm[None, :] + gm) ** 2).sum() / ((r - 1) * (c - 1))

        rows.append({
            "setting": setting, "arch": arch,
            "n_dataset_seeds": int(r), "n_training_seeds": int(c), "n_cells": int(y.size),
            "var_dataset_seed": v_ds, "var_training_seed": v_tr,
            "frac_dataset_seed": v_ds / tot if tot else np.nan,
            "frac_training_seed": v_tr / tot if tot else np.nan,
            "sd_dataset_seed": float(np.sqrt(v_ds)), "sd_training_seed": float(np.sqrt(v_tr)),
            "estimator": "nested between/within, ddof=1",
            "var_dataset_seed_twoway": float((MSA - MSE) / c),
            "var_training_seed_twoway": float((MSB - MSE) / r),
            "var_residual_twoway": float(MSE),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=None,
                    help="A7 result directory (default: the active version's)")
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--n-draws", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--latent-dataset-seed", type=int, default=0,
                    help="dataset seed used for the latent marginal / cross-mode tables")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--skip-models", action="store_true",
                    help="tables that need no checkpoint only")
    args = ap.parse_args()

    out = args.dir or DS.rev("A7")
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(out / "training_runs.csv")
    t0 = time.time()

    dedup = raw.drop_duplicates(subset=KEY, keep="first").reset_index(drop=True)
    dedup.to_csv(out / "training_runs_dedup.csv", index=False)

    pair_rows = []
    for k, d in raw[raw.duplicated(subset=KEY, keep=False)].groupby(KEY, sort=False):
        d = d.sort_index()
        pair_rows.append(dict(zip(KEY, k),
                              mae1=d.best_val_MAE_logM.iloc[0], mae2=d.best_val_MAE_logM.iloc[1],
                              abs_diff=abs(d.best_val_MAE_logM.iloc[0] - d.best_val_MAE_logM.iloc[1]),
                              epoch1=d.best_epoch.iloc[0], epoch2=d.best_epoch.iloc[1],
                              epoch_diff=d.best_epoch.iloc[0] - d.best_epoch.iloc[1]))
    # The pairs only exist where a cell was trained more than once. A clean run
    # produces none, so the frame is built with its columns declared: an empty
    # CSV with no header cannot be read back by any consumer.
    MPS_COLS = KEY + ["mae1", "mae2", "abs_diff", "epoch1", "epoch2", "epoch_diff"]
    pd.DataFrame(pair_rows, columns=MPS_COLS).to_csv(
        out / "mps_reproducibility_pairs.csv", index=False)
    print(f"  dedup {len(raw)} -> {len(dedup)} runs; {len(pair_rows)} duplicate pairs")

    settings = [s for s in LATENT_SETTINGS if s in set(dedup.setting)]
    marg, cross = latent_tables(settings, args.latent_dataset_seed)
    marg.to_csv(out / "marginal_check.csv", index=False)
    cross.to_csv(out / "cross_mode_correlation.csv", index=False)
    print(f"  latent tables on dataset seed {args.latent_dataset_seed}: {len(marg)} settings")

    cells = sorted({(r.setting, int(r.dataset_seed)) for r in dedup.itertuples()})
    orc = oracle_table(cells, args.n_draws, args.seed)
    orc.drop(columns=["_err"]).to_csv(out / "oracle_per_setting.csv", index=False)
    print(f"  oracle: {len(orc)} cells, {args.n_draws} draws/tuple")

    variance_table(dedup).to_csv(out / "variance_decomposition.csv", index=False)

    if not args.skip_models:
        ckpt_root = out / "checkpoints"
        imp = improvement_table(dedup, orc, ckpt_root, sorted(dedup.seed.unique()),
                                args.n_boot, args.seed, args.device)
        imp.to_csv(out / "improvement_over_oracle.csv", index=False)
        print(f"  improvement: {len(imp)} cells with {args.n_boot}-sample paired bootstrap")
        rho_table(dedup, ckpt_root, args.device).to_csv(out / "rho_per_setting.csv", index=False)
        print(f"  rho: {len(imp)} cells on the replicate split")

    (out / "run_sidecar_postprocess.json").write_text(json.dumps({
        "task": "A7_postprocess",
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_commit": git_sha(),
        "dataset_version": DS.VERSION,
        "seed": args.seed, "n_draws": args.n_draws, "n_boot": args.n_boot,
        "latent_dataset_seed": args.latent_dataset_seed,
        "skip_models": bool(args.skip_models),
        "wall_time_sec": round(time.time() - t0, 1),
        "date_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    try:
        shown = out.relative_to(ROOT)
    except ValueError:
        shown = out
    print(f"  wrote {shown} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
